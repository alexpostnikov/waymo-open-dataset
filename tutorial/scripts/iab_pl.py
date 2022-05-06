import pytorch_lightning as pl
import torch
import torch.nn as nn
import os
from scripts.dataloaders import context_description, CustomImageDataset, d_collate_fn
from scripts.pointNet import PointNetfeat
import torchvision.models as models
from scripts.iab import InitEmbedding, Encoder, DecoderGoals, DecoderTraj3
import numpy as np
import math
from torch.optim.lr_scheduler import LambdaLR
from scripts.train import preprocess_batch, get_future, log_likelihood
from scripts.rgb_loader import RgbLoader
import pathlib
from read_map_ds import WaymoDataset

def get_cosine_with_hard_restarts_schedule_with_warmup_with_min(
        optimizer: torch.optim.Optimizer, num_warmup_steps: int, num_training_steps: int, num_cycles: int = 1, last_epoch: int = -1,
        minimal_coef: float = 0.2
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, with several hard restarts, after a warmup period during which it increases
    linearly between 0 and the initial lr set in the optimizer.

    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
            The total number of training steps.
        num_cycles (:obj:`int`, `optional`, defaults to 1):
            The number of hard restarts to use.
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        if progress >= 1.0:
            return 0.0
        return max(minimal_coef, 0.5 * (1.0 + math.cos(math.pi * ((float(num_cycles) * progress) % 1.0))))

    return LambdaLR(optimizer, lr_lambda, last_epoch)

class AttPredictorPecNetWithTypeD3(pl.LightningModule):
    def __init__(self, config=None, wandb_logger=None):
        super().__init__()
        self.config = config
        self.use_gt_goals = False
        self.wandb_logger = wandb_logger
        self.use_points = use_points = config.exp_use_points
        if self.use_points:
            self.pointNet = PointNetfeat(global_feat=True)
        self.use_vis = use_vis = config.exp_use_vis
        if self.use_vis:
            self.visual = models.resnet18(pretrained=True)
            self.visual.fc = torch.nn.Linear(512, 1024)
        out_modes = 6
        embed_dim = config.exp_embed_dim
        use_rec = config.exp_use_rec
        inp_dim = config.exp_inp_dim
        num_blocks = config.exp_num_blocks
        dr_rate = 0.1
        out_dim = 2
        out_horiz = 80
#         self.tr_rgb_loader = RgbLoader(config.train_index_path)
#         self.val_rgb_loader = RgbLoader(config.val_index_path)
        self.latent = nn.Parameter(torch.rand(out_modes, embed_dim + 16), requires_grad=True)
        self.embeder = InitEmbedding(inp_dim=5, out_dim=embed_dim, use_recurrent=use_rec)
        self.encoder = Encoder(inp_dim, embed_dim, num_blocks, use_vis=use_vis, use_points=use_points, dr_rate=dr_rate)
        self.decoder_goals = DecoderGoals(embed_dim, 12, out_modes, dr_rate=dr_rate, use_recurrent=use_rec)
        self.decoder_trajs = DecoderTraj3(embed_dim, 12, out_modes, out_dim, out_horiz, dr_rate, use_recurrent=use_rec)

        self.learning_rate = self.config.exp_lr
        self.batch_size = self.config.exp_batch_size

    def forward(self, batch_unpacked):
        ## xyz emb
        masks, rot_mat, rot_mat_inv, state_masked, xyz_personal, maps = batch_unpacked
        bsr = state_masked.shape[0]
        agent_h_emb = self.embeder(state_masked)
        # pointnet embedder
        xyz_emb = None
        if self.use_points:
            xyz_emb, _, _ = self.pointNet(xyz_personal.permute(0, 2, 1))

        img_emb = None
        if self.use_vis:
            maps = maps[:, :3] #.to(self.latent.device) #cuda()
            img_emb = self.visual(maps) #.to(self.latent.device).cuda()
        x = self.latent.unsqueeze(0).repeat(bsr, 1, 1)
        x = self.encoder(x, img_emb, agent_h_emb, xyz_emb)
        goal_vector = self.decoder_goals(x)
        predictions, confidences = self.decoder_trajs(x, goal_vector)

        return predictions.permute(0, 2, 1, 3), confidences, goal_vector,  rot_mat, rot_mat_inv

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1,  gamma=1.)
        # lr_scheduler = get_cosine_with_hard_restarts_schedule_with_warmup_with_min(
        #     optimizer=optimizer,
        #     num_warmup_steps=20,
        #     num_training_steps=(22000 * 128 / self.config.exp_batch_size) * self.config.exp_num_epochs,
        #     num_cycles=self.config.exp_num_epochs, minimal_coef=0.8)
        lr_scheduler_config = {
            # REQUIRED: The scheduler instance
            "scheduler": lr_scheduler,
            # The unit of the scheduler's step size, could also be 'step'.
            # 'epoch' updates the scheduler on epoch end whereas 'step'
            # updates it after a optimizer update.
            "interval": "epoch",
            # How many epochs/steps should pass between calls to
            # `scheduler.step()`. 1 corresponds to updating the learning
            # rate after every epoch/step.
            "frequency": 1,
            # Metric to to monitor for schedulers like `ReduceLROnPlateau`
            # "monitor": "val_loss",

            # If set to `True`, will enforce that the value specified 'monitor'
            # is available when the scheduler is updated, thus stopping
            # training if not found. If set to `False`, it will only produce a warning
            # "strict": True,
            # If using the `LearningRateMonitor` callback to monitor the
            # learning rate progress, this keyword can be used to specify
            # a custom logged name
            # "name": None,
        }
        return [optimizer], [lr_scheduler_config]

    def training_step(self, batch, batch_idx):
        # todo from config
        use_every_nth_prediction = 1

        batch_unpacked = preprocess_batch(batch, self.use_points, self.use_vis)

        poses, confs, goals_local, rot_mat, rot_mat_inv = self(batch_unpacked)
        loss_goals, loss_nll, m_ade, m_fde = self.get_losses(batch, confs, goals_local, poses, rot_mat,
                                                      use_every_nth_prediction)

        loss = 0.01 * m_ade + 1 * loss_nll + 1 * loss_goals + 0.1 * m_fde

        # my_lr = [0]
        my_lr = self.lr_schedulers().get_last_lr()


        self.log("train_m_ade", m_ade, prog_bar=True)
        self.log("train_m_fde", m_fde, prog_bar=True)
        if self.wandb_logger:
            self.wandb_logger.log({"loss": loss_nll,
                        "min_ade": m_ade.item(),
                        "min_fde": m_fde.item(),
                        "lr": my_lr[0]})


        # x, y = batch
        # y_hat = self(x)
        # loss = F.cross_entropy(y_hat, y)
        return loss

    def get_losses(self, batch, confs, goals_local, poses, rot_mat, use_every_nth_prediction):
        mask = batch["state/tracks_to_predict"]
        valid = batch["state/future/valid"].reshape(-1, 128, 80)[mask > 0].to(poses.device)[:,
                use_every_nth_prediction - 1::use_every_nth_prediction]
        fut_path = get_future(batch).to(poses.device).permute(0, 2, 1, 3)[mask > 0]
        fut_ext = torch.cat([fut_path, torch.ones_like(fut_path[:, :, :1])], -1)
        fut_path = torch.bmm(rot_mat, fut_ext.permute(0, 2, 1)).permute(0, 2, 1)[:,
                   use_every_nth_prediction - 1::use_every_nth_prediction, :2]
        m_ades = (torch.norm((fut_path.unsqueeze(2) - poses), dim=-1) * valid.unsqueeze(2)).mean(1).min(
            -1).values.mean()
        m_fdes = (torch.norm((fut_path[:, -1].unsqueeze(1) - goals_local.reshape(-1, 6, 2)), dim=-1) * valid[:,
                                                                                                       -1].unsqueeze(
            1)).min(
            -1).values
        m_fdes = m_fdes[m_fdes > 0]
        if len(m_fdes) > 0:
            m_fde = m_fdes.mean()
        else:
            m_fde = torch.tensor([0.]).to(m_fdes.device)
        fut_path_masked = fut_path.unsqueeze(2) * valid.unsqueeze(2).unsqueeze(2)
        pred_masked = poses * valid.unsqueeze(2).unsqueeze(2)
        selector = np.arange(4, 80 + 1, 5)
        loss_nll = -log_likelihood(fut_path_masked[:, selector], pred_masked[:, selector], confs).mean() \
                   - 0.1 * log_likelihood(fut_path_masked, pred_masked, confs).mean()
        goals_masked = (valid.unsqueeze(2).unsqueeze(2)[:, -1] * goals_local.reshape(-1, 6, 2))
        loss_goals = -log_likelihood(fut_path_masked[:, -1:], goals_masked.unsqueeze(1), confs).mean()
        m_ade = m_ades.mean()
        return loss_goals, loss_nll, m_ade, m_fde

    def validation_step(self, batch, batch_idx):
        use_every_nth_prediction = 1
    
        batch_unpacked = preprocess_batch(batch, self.use_points, self.use_vis)

        poses, confs, goals_local, rot_mat, rot_mat_inv = self(batch_unpacked)
        loss_goals, loss_nll, m_ade, m_fde = self.get_losses(batch, confs, goals_local, poses, rot_mat,
                                                             use_every_nth_prediction)

        loss = 0.01 * m_ade + 1 * loss_nll + 1 * loss_goals + 0.1 * m_fde
        self.log("val_loss", loss)
        self.log("val_m_ade", m_ade)
        self.log("val_m_fde", m_fde)
        self.log("val_loss_goals", loss_goals)
        self.log("val_loss_nll", loss_nll)
        return loss

    def test_step(self, batch, batch_idx):
        pass

    def predict_step(self, batch, batch_idx, rgb_loader):

        batch["rgbs"] = torch.tensor(rgb_loader.load_batch_rgb(batch, prefix="").astype(np.float32))

        batch_unpacked = preprocess_batch(batch, self.use_points, self.use_vis)

        pred = self(batch_unpacked)
        return pred

    def train_dataloader(self):
        # ds_path = pathlib.Path("/home/jovyan/uncompressed/tf_example/")
        ds_path = self.config.dir_data
        assert pathlib.Path(ds_path).exists()

        index_file = "training_mapstyle/index_file.txt"
        # join
        index_file = pathlib.Path(ds_path) / index_file
        # join the path with the index file
        train_dataset = WaymoDataset(ds_path, index_file, rgb_index_path="/home/jovyan/rendered/train/index.pkl", rgb_prefix="/home/jovyan/")

        
        # train_tfrecord_path = os.path.join(self.config.dir_data, "training/training_tfexample.*-of-01000")
        # train_dataset = CustomImageDataset(train_tfrecord_path, context_description)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size,
                                                   num_workers=self.config.exp_num_workers, collate_fn=d_collate_fn)
        print("train_loader done")
        return train_loader

    def val_dataloader(self):
        ds_path = self.config.dir_data
        index_file = "val_mapstyle/index_file.txt"
        # join
        index_file = pathlib.Path(ds_path) / index_file
        # join the path with the index file
        val_dataset = WaymoDataset(ds_path, index_file,rgb_index_path="/home/jovyan/rendered/val/index.pkl", rgb_prefix="/home/jovyan/")

        
        # train_tfrecord_path = os.path.join(self.config.dir_data, "training/training_tfexample.*-of-01000")
        # train_dataset = CustomImageDataset(train_tfrecord_path, context_description)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.batch_size,
                                                   num_workers=self.config.exp_num_workers, collate_fn=d_collate_fn)
        print("val_loader done")
        return val_loader

    def test_dataloader(self):
        return None

    def predict_dataloader(self):
        return None
