import torch
from tqdm.auto import tqdm
import torch.nn as nn
import numpy as np
from einops import rearrange
from test.visualize import vis_cur_and_fut
import wandb
# from test_train import get_ade_from_pred_speed_with_mask, get_speed_ade_with_mask


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
                     use_every_nth_prediction=1):

    train_loader, test_loader = loaders
    for epoch in range(num_ep):  # loop over the dataset multiple times
        train_losses = train_epoch(epoch, logger, model, optimizer, train_loader, use_every_nth_prediction)
        # test_losses = test_epoch(epoch, logger, model, test_loader)
        checkpointer.save(epoch, train_losses.mean().item())


def test_epoch(epoch, logger, model, test_loader, max_chanks=100):
    losses = torch.rand(0)
    pbar = tqdm(test_loader)
    model.eval()
    for chank, data in enumerate(pbar):
        if chank > max_chanks:
            break
        poses, confs = model(data)
        # poses -> bs, num_peds, times, modes, 2
        loss_nll = pytorch_neg_multi_log_likelihood_batch(data, poses, confs).mean()
        uni_ades = []
        for mode in range(poses.shape[3]):
            ades = uni_ade_poses(data, poses[:, :, :, mode])
            # calc uni ade. 1.pred = pred+cur, 2. torch.norm(pred, gt), 3. apply mask
            uni_ades.append(ades)
        m_ades = torch.stack(uni_ades, dim=1)
        m_ades = torch.min(m_ades, dim=1).values

        valid = get_valid_data_mask(data)
        mask = (data["state/tracks_to_predict"].reshape(-1, 128, 1).repeat(1, 1, 80) > 0) * (valid > 0)
        m_ade = m_ades[mask > 0].mean()

        with torch.no_grad():
            losses = torch.cat([losses, torch.tensor([loss_nll.detach().item()])], 0)
            pbar.set_description("ep %s chank %s" % (epoch, chank))
            pbar.set_postfix({"loss": losses.mean().item(), "m_ade": m_ade.item()})
            logger.log({"loss": loss_nll,
                        "min_ade": m_ade.item()})
            if len(losses) > 500:
                losses = losses[100:]
    return losses

def train_epoch(epoch, logger, model, optimizer, train_loader, use_every_nth_prediction=1):
    losses = torch.rand(0)
    mades = torch.rand(0)
    pbar = tqdm(train_loader)
    for chank, data in enumerate(pbar):
        optimizer.zero_grad()
        poses, confs = model(data)
        bs = poses.shape[0]
        # poses -> bs, num_peds, times, modes, 2
        # loss_nll = pytorch_neg_multi_log_likelihood_batch(data, poses, confs, use_every_nth_prediction=use_every_nth_prediction).mean()

        uni_ades = []
        for mode in range(poses.shape[3]):
            ades = uni_ade_poses(data, poses[:, :, :,
                                       mode], use_every_nth_prediction=use_every_nth_prediction)  # calc uni ade. 1.pred = pred+cur, 2. torch.norm(pred, gt), 3. apply mask
            uni_ades.append(ades)
        m_ades = torch.stack(uni_ades, dim=1)
        m_ades = torch.min(m_ades, dim=1).values

        valid = get_valid_data_mask(data)
        mask = (data["state/tracks_to_predict"].reshape(-1, 128, 1).repeat(1, 1, 80) > 0) * (valid > 0)
        gt_fut = get_future(data).to(poses.device)
        gt = rearrange(gt_fut, "bs timestemps num_peds data_dim -> (bs num_peds) timestemps data_dim")[:, ::use_every_nth_prediction]
        cur = get_current(data).to(poses.device)
        poses_moved = cur.permute(0, 2, 1, 3).unsqueeze(3).repeat(1, 1, 1, 6, 1) + poses
        (mask.sum(2) == 80).reshape(-1)
        # log_likelihood(gt.unsqueeze(1), poses_moved.reshape(512, 4, 6, 2).permute(0, 2, 1, 3), confs.reshape(512, 6))
        loss_nll = -log_likelihood(gt.unsqueeze(1), poses_moved.reshape(bs*128, -1, 6, 2).permute(0, 2, 1, 3),
                                   confs.reshape(bs*128, 6))[
            (mask.sum(2) == 80).reshape(-1)].mean()
        m_ade = m_ades[mask[:, :, ::use_every_nth_prediction] > 0].mean()
        loss = m_ade + loss_nll
        loss.backward()

        optimizer.step()
        with torch.no_grad():
            losses = torch.cat([losses, torch.tensor([loss_nll.detach().item()])], 0)
            mades = torch.cat([mades, torch.tensor([m_ade.detach().item()])], 0)
            pbar.set_description("ep %s chank %s" % (epoch, chank))
            pbar.set_postfix({"loss": losses.mean().item(), "m_ade": mades.mean().item(),
                              "predicted": (mask[:, :, ::use_every_nth_prediction].sum(2)/poses.shape[2]).sum().item()})
            logger.log({"loss": loss_nll,
                        "min_ade": m_ade.item()})
            if len(losses) > 500:
                losses = losses[100:]
        image = vis_cur_and_fut(data, poses)
        if (chank % 200) == 0:
            images = wandb.Image(image, caption="Top: Output, Bottom: Input")
            wandb.log({"examples": images})
    return losses


def uni_ade_poses(data, predictions, use_every_nth_prediction=1):
    bs, num_ped, future_steps, _ = predictions.shape
    gt_fut = get_future(data).to(predictions.device)[:, ::use_every_nth_prediction]
    assert gt_fut.shape == torch.Size([bs, future_steps, num_ped, 2])

    cur = get_current(data).to(predictions.device)
    assert cur.shape == torch.Size([bs, 1, num_ped, 2])
    pred_moved = predictions + cur.permute(0, 2, 1, 3)
    dist = torch.norm((pred_moved - gt_fut.permute(0, 2, 1, 3)), dim=-1)
    return dist


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
    fut_valid = torch.ones([bs,128,80])>0
    if check_fut:
        fut_valid *= data["state/future/valid"].reshape(bs, 128, -1)>0
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
        normalizing_const + 0.5 * displacement_norms_squared / sigma ** 2, dim=-1)
    if ground_truth.ndim == 4:
        max_arg = lse_args.max(1).values.reshape(-1,1)
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
    gt = rearrange(gt_fut, "bs timestemps num_peds data_dim -> (bs num_peds) timestemps data_dim")[:, ::use_every_nth_prediction]
    gt = torch.unsqueeze(gt, 1)  # add modes
    valid = get_valid_data_mask(data)
    mask = data["state/tracks_to_predict"].reshape(-1, 128, 1).repeat(1, 1, 80) * valid
    avails = mask.permute(0, 2, 1)
    avails = rearrange(avails, "bs timestemps num_peds -> (bs num_peds) timestemps")
    avails = avails[:, None, :, None].to(logits.device)[:, :, ::use_every_nth_prediction]  # add modes and cords

    logits = rearrange(logits, "bs num_peds timestemps modes data_dim -> (bs num_peds) modes timestemps   data_dim")
    cur = get_current(data)
    assert cur.shape == torch.Size([bs, 1, num_ped, 2])
    cur = rearrange(cur, "bs time num_peds data -> (bs num_peds) 1 time data").to(logits.device)#[:, :, ::use_every_nth_prediction]
    logits_moved = logits + cur
    confidences = rearrange(confidences, "bs num_peds modes -> (bs num_peds) modes")
    # error (batch_size, num_modes, future_len)
    # print(gt.shape, logits_moved.shape, avails.shape)
    error = torch.sum(
        ((gt - logits_moved) * avails) ** 2, dim=-1
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
