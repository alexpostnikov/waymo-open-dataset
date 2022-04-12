import torch
from tqdm.auto import tqdm
import torch.nn as nn
import numpy as np
from einops import rearrange
from test.visualize import vis_cur_and_fut
import wandb
from waymo_open_dataset.protos import motion_submission_pb2

# from test_train import get_ade_from_pred_speed_with_mask, get_speed_ade_with_mask
def create_subm(model,  loader):
    model.load_state_dict(torch.load("/home/jovyan/waymo-open-dataset/model-seed-0-epoch-0.pt", map_location="cuda"))
    # print(model)
    motion_challenge_submission = motion_submission_pb2.MotionChallengeSubmission()
    motion_challenge_submission.account_name = "AlexPostnik"
    authors = "postnikov,gamaynov"
    motion_challenge_submission.authors.extend(authors.split(","))
    motion_challenge_submission.submission_type = (
        motion_submission_pb2.MotionChallengeSubmission.SubmissionType.MOTION_PREDICTION
    )
    motion_challenge_submission.unique_method_name = "iab"
    model.eval()
    RES = {}
    with torch.no_grad():
        pbar = tqdm(loader, total=int(22000*128//479*150))
        for chank, data in enumerate(pbar):
            logits, confidences, goals, goal_vector, rot_mat, rot_mat_inv = model(data)

            logits = logits.cpu().numpy()
            confidences = confidences.cpu().numpy()
            mask = data["state/tracks_to_predict"].reshape(-1,128)>0
            agent_id = data["state/id"].cpu()[mask].numpy()
            scenario_id = data["scenario/id"]  #?
            # scenario_id = [sc.numpy().tobytes().decode("utf-8") for sc in scenario_id]
            scenarios_id = []
            for bn, scenario in enumerate(scenario_id):
                [scenarios_id.append(scenario) for i in range((mask.nonzero()[:,0] == bn).sum())]
            # next(iter(test_loader))["scenario/id"][0].numpy().tobytes().decode("utf-8")
            # center = center.cpu().numpy()
            # yaw = yaw.cpu().numpy()
            for p, conf, aid, sid in zip(
                    logits, confidences, agent_id, scenarios_id,
            ):
                if sid not in RES:
                    RES[sid] = []

                RES[sid].append(
                    {"aid": aid, "conf": conf, "pred": p}
                )

        selector = np.arange(14, 80 + 1, 5)
        for scenario_id, data in tqdm(RES.items()):
            scenario_predictions = motion_challenge_submission.scenario_predictions.add()
            scenario_predictions.scenario_id = scenario_id
            prediction_set = scenario_predictions.single_predictions

            for d in data:
                predictions = prediction_set.predictions.add()
                predictions.object_id = int(d["aid"])

                # y = d["yaw"]
                # rot_matrix = np.array([
                #     [np.cos(y), -np.sin(y)],
                #     [np.sin(y), np.cos(y)],
                # ])

                for i in np.argsort(-d["conf"]):
                    scored_trajectory = predictions.trajectories.add()
                    scored_trajectory.confidence = d["conf"][i]

                    trajectory = scored_trajectory.trajectory

                    p = d["pred"][selector][i]  #@ rot_matrix + d["center"]

                    trajectory.center_x.extend(p[:, 0])
                    trajectory.center_y.extend(p[:, 1])

        with open("file", "wb") as f:
            f.write(motion_challenge_submission.SerializeToString())

def train(model, loader, optimizer, num_ep=10):
    losses = torch.rand(0)
    for epoch in range(num_ep):  # loop over the dataset multiple times
        pbar = tqdm(loader)
        for chank, data in enumerate(pbar):
            optimizer.zero_grad()
            outputs = model(data)

            loss = get_ade_from_pred_speed_with_mask(data, outputs).mean()
            # loss = ade_loss(data, outputs)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                speed_ade = get_speed_ade_with_mask(data, outputs.clone())
                lin_ade = get_ade_from_pred_speed_with_mask(data, outputs.clone())
                losses = torch.cat([losses, torch.tensor([loss.detach().item()])], 0)
                pbar.set_description("ep %s chank %s" % (epoch, chank))
                pbar.set_postfix({"loss": losses.mean().item(),
                                  "median": speed_ade.median().item(),
                                  "max": speed_ade.max().item(),
                                  "lin_ade": lin_ade.mean().item()})
                if len(losses) > 500:
                    losses = losses[100:]


def train_multymodal(model, loaders, optimizer, num_ep=10, checkpointer=None, logger=None,
                     use_every_nth_prediction=1, scheduler=None):

    train_loader, test_loader = loaders
    for epoch in range(num_ep):  # loop over the dataset multiple times
        model.train()
        train_losses = train_epoch(epoch, logger, model, optimizer, train_loader, use_every_nth_prediction, scheduler)
        # test_losses = test_epoch(epoch, logger, model, test_loader)
        checkpointer.save(epoch, train_losses.mean().item())


def goal_vector_to_gmm(predictions,  rot_mat_inv, out_modes=6):

    bs = predictions.shape[0]
    mean_predictions = predictions[:, :2 * out_modes].reshape(bs, out_modes, 2)
    cov_predictions = 3 * torch.eye(2).unsqueeze(0).unsqueeze(0).repeat(bs, out_modes, 1, 1).to(
        mean_predictions.device)
    probs_predictions = predictions[:, -out_modes:]

    probs_predictions = torch.sigmoid(probs_predictions)
    mix = torch.distributions.Categorical(probs_predictions)

    scale_tril = torch.tril(((cov_predictions ** 2) + 1e-6))
    scale_tril[:, :, 1, 0] = cov_predictions[:, :, 1, 0]

    distr = torch.distributions.multivariate_normal.MultivariateNormal(mean_predictions, scale_tril=scale_tril)

    gmm = torch.distributions.MixtureSameFamily(mix, distr)
    return gmm

def train_epoch(epoch, logger, model, optimizer, train_loader, use_every_nth_prediction=1, scheduler=None):
    losses = torch.rand(0)
    mades = torch.rand(0)
    mfdes = torch.rand(0)
    pbar = tqdm(train_loader)
    for chank, data in enumerate(pbar):
        optimizer.zero_grad()
        poses, confs, goals, goal_vector,  rot_mat, rot_mat_inv = model(data)
        gmm = goal_vector_to_gmm(goal_vector,  rot_mat_inv)
        mask = data["state/tracks_to_predict"]
        valid = data["state/future/valid"].reshape(-1, 128, 80)[mask > 0].to(poses.device)[:, use_every_nth_prediction - 1::use_every_nth_prediction]
        fut_path = get_future(data).to(poses.device).permute(0, 2, 1, 3)[mask > 0]

        fut_ext = torch.cat([fut_path, torch.ones_like(fut_path[:, :, :1])], -1)
        fut_path = torch.bmm(rot_mat, fut_ext.permute(0,2,1)).permute(0, 2, 1)[:, use_every_nth_prediction - 1::use_every_nth_prediction, :2]
        goal_nll = -gmm.log_prob(fut_path[:, -1])[valid[:, -1] > 0]

        m_ades = (torch.norm((fut_path.unsqueeze(2) - poses), dim=-1)*valid.unsqueeze(2)).mean(1).min(-1).values.mean()
        m_fdes = (torch.norm((fut_path[:, -1].unsqueeze(1) - poses[:, -1]), dim=-1) * valid[:, -1].unsqueeze(1)).min(-1).values
        m_fdes = m_fdes[m_fdes > 0].mean()

        fut_path_masked = fut_path.unsqueeze(2) * valid.unsqueeze(2).unsqueeze(2)
        pred_masked = poses * valid.unsqueeze(2).unsqueeze(2)
        loss_nll = -log_likelihood(fut_path_masked, pred_masked, confs).mean()
        m_ade = m_ades.mean()
        m_fde = m_fdes.mean()
        loss = 0.01 * m_ade + 1 * loss_nll + 1 * goal_nll.mean() + 0.001 * m_fde
        loss.backward()

        optimizer.step()
        my_lr = [0]
        if scheduler is not None:
            scheduler.step()
            my_lr = scheduler.get_last_lr()
            # logger.log({})

        with torch.no_grad():
            losses = torch.cat([losses, torch.tensor([loss_nll.detach().item()])], 0)
            mades = torch.cat([mades, torch.tensor([m_ade.detach().item()])], 0)
            mfdes = torch.cat([mfdes, torch.tensor([m_fde.detach().item()])], 0)
            pbar.set_description("ep %s chank %s" % (epoch, chank))
            pbar.set_postfix({"loss": losses.mean().item(), "m_ade": mades.mean().item(), 'fde': mfdes.mean().item()
                              # "predicted": (mask[:, :, ::use_every_nth_prediction].sum(2)/poses.shape[2]).sum().item()
                              })
            logger.log({"loss": loss_nll,
                        "min_ade": m_ade.item(),
                        "min_fde": m_fde.item(),
                        "goal_nll": goal_nll.mean().item(),
                        "LR": my_lr[0]})
            if len(losses) > 500:
                losses = losses[100:]

        if (chank % 200) == 0:
            poses = apply_tr(poses, rot_mat_inv)
            image = vis_cur_and_fut(data, poses, confs=confs)
            images = wandb.Image(image, caption="Top: Output, Bottom: Input")
            wandb.log({"examples": images})
    return losses


def apply_tr(poses, tr):
    poses_exp = torch.cat([poses, torch.ones_like(poses[..., :1])], dim=-1)
    bs, times, modes, datadim = poses_exp.shape
    poses_exp = torch.bmm(tr, rearrange(poses_exp, "bs times  modes  datadim -> bs  datadim  (times  modes) "))
    poses_exp = rearrange(poses_exp, "bs  datadim  (times  modes)  -> bs  times  modes  datadim",
                          times=times, modes=modes)[..., :2]
    return poses_exp

def uni_ade_poses(data, predictions, use_every_nth_prediction=1):
    bs, num_ped, future_steps, _ = predictions.shape
    gt_fut = get_future(data).to(predictions.device)[:, use_every_nth_prediction-1::use_every_nth_prediction]
    mask = data["state/tracks_to_predict"].reshape(-1, 128)  # .repeat(1, 1, 80) > 0)
    gt_fut = gt_fut.permute(0, 2, 1, 3)[mask > 0]
    # assert gt_fut.shape == torch.Size([bs, future_steps, num_ped, 2])

    cur = get_current(data).to(predictions.device)
    cur = cur.permute(0, 2, 1, 3)[mask > 0]  # [:,0,:]
    # assert cur.shape == torch.Size([bs, 1, num_ped, 2])
    gt_fut = gt_fut - cur
    fut_valid = data["state/future/valid"].reshape(-1, 128, 80)[mask > 0].unsqueeze(2).to(predictions.device)
    fut_valid = fut_valid[:, use_every_nth_prediction-1::use_every_nth_prediction]
    err = (predictions[:, 0] - gt_fut)[fut_valid[:,:,0]>0]
    dist = torch.norm(err, dim=-1)
    return dist


def goals_nll(gmm, data):
    gt_fut = get_future(data).to(gmm.device)[:, -1]
    mask = data["state/tracks_to_predict"].reshape(-1, 128)  # .repeat(1, 1, 80) > 0)
    gt_fut = gt_fut[mask > 0]

def uni_fde_poses(data, predictions):
    bs, num_ped, future_steps, _ = predictions.shape
    gt_fut = get_future(data).to(predictions.device)[:, -1]
    mask = data["state/tracks_to_predict"].reshape(-1, 128)  # .repeat(1, 1, 80) > 0)
    gt_fut = gt_fut[mask > 0]
    # assert gt_fut.shape == torch.Size([bs, future_steps, num_ped, 2])

    cur = get_current(data).to(predictions.device)
    cur = cur.permute(0, 2, 1, 3)[mask > 0]  # [:,0,:]
    # assert cur.shape == torch.Size([bs, 1, num_ped, 2])
    gt_fut = gt_fut - cur[:, 0]
    fut_valid = data["state/future/valid"].reshape(-1, 128, 80)[mask > 0].unsqueeze(2).to(predictions.device)
    err = (predictions[:, 0, -1] - gt_fut) * fut_valid[:, -1]
    dist = torch.norm(err, dim=-1)
    return dist[dist>0]


def get_future(data):
    bs = data["state/future/x"].shape[0]
    gt_fut = torch.cat([data["state/future/x"].reshape(bs, 128, 80, 1), data["state/future/y"].reshape(bs, 128, 80, 1)],
                       -1)
    gt_fut = gt_fut.permute(0, 2, 1, 3)
    # bs, 80, 128, 2
    return gt_fut


def get_current(data):
    cur = torch.cat([data["state/current/x"].reshape(-1, 1, 128, 1), data["state/current/y"].reshape(-1, 1, 128, 1)],
                    -1)
    return cur


def get_future_speed(data, num_ped=128, future_steps=80):
    bs = data["state/future/x"].shape[0]
    gt_fut = get_future(data)
    assert gt_fut.shape == torch.Size([bs, future_steps, num_ped, 2])
    cur = get_current(data)
    assert cur.shape == torch.Size([bs, 1, num_ped, 2])
    gt_fut[:, 1:, :, :] = gt_fut[:, 1:, :, :] - gt_fut[:, :-1, :, :]
    gt_fut[:, 0:1] = gt_fut[:, 0:1] - cur
    return gt_fut


def get_valid_data_mask(data, check_fut=1, check_cur=0, check_past=0):
    bs = data["state/future/x"].shape[0]
    fut_valid = torch.ones([bs, 128, 80]) > 0
    if check_fut:
        fut_valid *= data["state/future/valid"].reshape(bs, 128, -1) > 0
    if check_cur:
        fut_valid *= (data["state/current/valid"].reshape(bs, 128, -1) > 0)
    if check_past:
        fut_valid *= (data["state/past/valid"].reshape(bs, 128, 10).sum(2) == 10).reshape(bs, 128, 1) > 0
    return fut_valid


def pred_to_future(data, pred, num_ped=128, future_steps=80):
    bs = data["state/future/x"].shape[0]
    pred_poses = pred.clone()
    cur = get_current(data).reshape(-1, 128, 2)
    assert pred.shape == torch.Size([bs, num_ped, future_steps, 2])
    pred_poses[:, :, 0] += cur.to(pred.device)
    pred_poses = torch.cumsum(pred_poses, 2)
    return pred_poses


def get_speed_ade_with_mask(data, pred, num_ped=128, future_steps=80):
    bs = data["state/future/x"].shape[0]
    assert pred.shape == torch.Size([bs, num_ped, future_steps, 2])
    gt_fut = get_future(data)
    assert gt_fut.shape == torch.Size([bs, future_steps, num_ped, 2])

    cur = get_current(data)
    assert cur.shape == torch.Size([bs, 1, num_ped, 2])
    gt_fut_speed = get_future_speed(data)
    dist = torch.norm(pred.permute(0, 2, 1, 3) - gt_fut_speed.cuda(), dim=3)
    valid = get_valid_data_mask(data)
    mask = data["state/tracks_to_predict"].reshape(-1, 128, 1).repeat(1, 1, 80) * valid
    mask = mask.permute(0, 2, 1)
    dist_masked = dist[mask > 0]
    #     dist = dist[dist<500]
    return dist_masked


def get_ade_from_pred_speed_with_mask(data, pred, num_ped=128, future_steps=80):
    bs = data["state/future/x"].shape[0]
    assert pred.shape == torch.Size([bs, num_ped, future_steps, 2])
    cur = get_current(data)
    assert cur.shape == torch.Size([bs, 1, num_ped, 2])
    loss = pred + 0
    loss[:, :, 0] = pred[:, :, 0] + cur[:, 0].to(pred.device)
    loss = torch.cumsum(loss, dim=2)
    gt_fut = get_future(data).to(pred.device)
    assert gt_fut.shape == torch.Size([bs, future_steps, num_ped, 2])
    dist = torch.norm((loss.permute(0, 2, 1, 3) - gt_fut), dim=3)
    valid = get_valid_data_mask(data)
    mask = data["state/tracks_to_predict"].reshape(-1, 128, 1).repeat(1, 1, 80) > 0 * valid > 0
    mask = mask.permute(0, 2, 1)
    dist_masked = dist[mask > 0]
    return dist_masked


def ade_loss(data, pred):
    ade = get_speed_ade_with_mask(data, pred)
    return ade.mean()


def log_likelihood(ground_truth, predicted, weights, sigma=1.0):
    """Calculates log-likelihood of the ground_truth trajectory
    under the factorized gaussian mixture parametrized by predicted trajectories, weights and sigma.
    Please follow the link below for the metric formulation:
    https://github.com/yandex-research/shifts/blob/195b3214ff41e5b6c197ea7ef3e38552361f29fb/sdc/ysdc_dataset_api/evaluation/log_likelihood_based_metrics.pdf

    Args:
        ground_truth (np.ndarray): ground truth trajectory, (n_timestamps, 2)
        predicted (np.ndarray): predicted trajectories, (n_modes, n_timestamps, 2)
        weights (np.ndarray): confidence weights associated with trajectories, (n_modes,)
        sigma (float, optional): distribution standart deviation. Defaults to 1.0.

    Returns:
        float: calculated log-likelihood
    """
    #     assert_weights_near_one(weights)
    #     assert_weights_non_negative(weights)
    #     print(ground_truth.shape,  predicted.shape)
    displacement_norms_squared = torch.sum((ground_truth - predicted) ** 2, dim=-1)
    #     print(displacement_norms_squared)

    displacement_norms_squared = torch.clamp(displacement_norms_squared, max=1e6)
    normalizing_const = np.log(2 * np.pi * sigma ** 2)
    lse_args = torch.log(weights + 1e-6) - torch.sum(
        normalizing_const + 0.5 * displacement_norms_squared.permute(0, 2, 1) / sigma ** 2, dim=-1)
    if ground_truth.ndim == 4:
        max_arg = lse_args.max(1).values.reshape(-1, 1)
    else:
        max_arg = lse_args.max()

    ll = torch.log(torch.sum(torch.exp(lse_args - max_arg), -1) + 1e-6) + max_arg.reshape(-1)
    return ll


def pytorch_neg_multi_log_likelihood_batch(data, logits, confidences, use_every_nth_prediction=1):
    """
    Compute a negative log-likelihood for the multi-modal scenario.
    Args:
        gt (Tensor): array of shape (bs)x(time)x(2D coords)
        logits (Tensor): array of shape (bs)x(modes)x(time)x(2D coords)
        confidences (Tensor): array of shape (bs)x(modes) with a confidence for each mode in each sample
        avails (Tensor): array of shape (bs)x(time) with the availability for each gt timestep
    Returns:
        Tensor: negative log-likelihood for this example, a single float number
    """

    bs, num_ped = logits.shape[:2]
    # convert to (batch_size, num_modes, future_len, num_coords)
    gt_fut = get_future(data).to(logits.device)
    gt = gt_fut.permute(0, 2, 1, 3)[data["state/tracks_to_predict"].reshape(-1, 128, 1)[:, :, 0] > 0]
    # gt = rearrange(gt_fut, "bs timestemps num_peds data_dim -> (bs num_peds) timestemps data_dim")[:,
    #      ::use_every_nth_prediction]
    # gt = torch.unsqueeze(gt, 1)  # add modes
    valid = get_valid_data_mask(data)
    mask = data["state/tracks_to_predict"].reshape(-1, 128, 1).repeat(1, 1, 80) * valid
    avails = mask.permute(0, 2, 1)
    avails = rearrange(avails, "bs timestemps num_peds -> (bs num_peds) timestemps")

    logits = rearrange(logits, "bs num_peds timestemps modes data_dim -> (bs num_peds) modes timestemps   data_dim")
    cur = get_current(data)
    cur = cur[:, 0][data["state/tracks_to_predict"].reshape(-1, 128, 1)[:, :, 0] > 0]
    cur = cur.unsqueeze(1).unsqueeze(1)
    assert cur.shape == torch.Size([bs, 1, num_ped, 2])
    cur = rearrange(cur, "bs time num_peds data -> (bs num_peds) 1 time data").to(
        logits.device)  # [:, :, ::use_every_nth_prediction]
    logits_moved = logits + cur
    # confidences = rearrange(confidences, "bs num_peds modes -> (bs num_peds) modes")
    # error (batch_size, num_modes, future_len)
    # print(gt.shape, logits_moved.shape, avails.shape)
    error = torch.sum(
        ((gt.unsqueeze(1) - logits_moved) * valid[data["state/tracks_to_predict"].reshape(-1, 128)>0].unsqueeze(1).unsqueeze(-1).cuda()) ** 2, dim=-1
    )  # reduce coords and use availability

    with np.errstate(
            divide="ignore"
    ):  # when confidence is 0 log goes to -inf, but we're fine with it
        # error (batch_size, num_modes)
        error = nn.functional.log_softmax(confidences, dim=1) - 0.5 * torch.sum(
            error, dim=-1
        )  # reduce time

    # error (batch_size, num_modes)
    error = -torch.logsumexp(error, dim=-1, keepdim=True)

    return torch.mean(error)
