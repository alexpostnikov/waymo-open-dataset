# Data location. Please edit.

# A tfrecord containing tf.Example protos as downloaded from the Waymo dataset
# webpage.

# Replace this path with your own tfrecords.
import sys
from pathlib import Path

sys.path.append(str(Path.cwd().parent)) 
import tensorflow as tf

from scripts.iab import AttPredictorPecNet
from scripts.train import train_multymodal, create_subm
from scripts.visualize import vis_cur_and_fut
from scripts.train import get_speed_ade_with_mask, get_ade_from_pred_speed_with_mask
from scripts.config import build_parser
from scripts.models import Checkpointer
from scripts.dataloaders import context_description, CustomImageDataset, d_collate_fn
from scripts.rgb_loader import  RgbLoader

tf.get_logger().setLevel('ERROR')

import torch

from tqdm.auto import tqdm
import torch.utils.data

import matplotlib.pyplot as plt
import numpy as np
import wandb
import random
import torch.optim as optim
from transformers import get_cosine_with_hard_restarts_schedule_with_warmup
# torch.backends.cudnn.benchmark = True

parser = build_parser()
config = parser.parse_args()
random.seed(config.np_seed)
torch.manual_seed(config.torch_seed)
np.random.seed(config.np_seed)

# Example field definition

# Features of other agents.

# from pathlib import Path
import os

wandb.init(project="waymo22", entity="aleksey-postnikov", name=config.exp_name)
wandb.config = {
    "learning_rate": config.exp_lr,
    "epochs": config.exp_num_epochs,
    "batch_size": config.exp_batch_size
}



batch_size = config.exp_batch_size

train_tfrecord_path = os.path.join(config.dir_data, "training/training_tfexample.*-of-01000")
test_path = os.path.join(config.dir_data, "validation/validation_tfexample.tfrecord-*-of-00150")
train_dataset = CustomImageDataset(train_tfrecord_path, context_description)
test_dataset = CustomImageDataset(test_path, context_description)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                           num_workers=config.exp_num_workers, collate_fn=d_collate_fn)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                          num_workers=config.exp_num_workers, collate_fn=d_collate_fn)


device = "cuda"

net = AttPredictorPecNet(inp_dim=config.exp_inp_dim, embed_dim=config.exp_embed_dim, num_blocks=config.exp_num_blocks,
                         out_modes=6, use_vis=config.exp_use_vis, use_rec=config.exp_use_rec,
                         use_points=config.exp_use_points, out_horiz=80//config.use_every_nth_prediction)
net = torch.nn.DataParallel(net)


optimizer = optim.Adam(net.parameters(), lr=wandb.config["learning_rate"])

from transformers import get_cosine_with_hard_restarts_schedule_with_warmup
from torch.optim.lr_scheduler import LambdaLR
from torch.optim import Optimizer
import math

def get_cosine_with_hard_restarts_schedule_with_warmup_with_min(
    optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int, num_cycles: int = 1, last_epoch: int = -1, minimal_coef: float = 0.2
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
# return LambdaLR(optimizer, lr_lambda, last_epoch)



scheduler = get_cosine_with_hard_restarts_schedule_with_warmup_with_min(
            optimizer=optimizer,
            num_warmup_steps=20,
            num_training_steps=(22000*128 / config.exp_batch_size) * wandb.config["epochs"],
            num_cycles=wandb.config["epochs"], minimal_coef=0.5) 
net = net.to(device)

checkpointer = Checkpointer(model=net, torch_seed=0, ckpt_dir=config.dir_checkpoint, checkpoint_frequency=1)
net = checkpointer.load(config.epoch_to_load)



def overfit_test(model, loader, optimizer):
    losses = torch.rand(0)
    pbar = tqdm(loader)
    data = next(iter(pbar))
    pbar = tqdm(range(200))
    for chank in pbar:
        optimizer.zero_grad(set_to_none=True)
        outputs = model(data)

        loss = get_ade_from_pred_speed_with_mask(data, outputs).mean()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            speed_ade = get_speed_ade_with_mask(data, outputs.clone())
            lin_ade = get_ade_from_pred_speed_with_mask(data, outputs.clone())
            losses = torch.cat([losses, torch.tensor([loss.detach().item()])], 0)
            pbar.set_description("ep %s chank %s" % (0, chank))
            pbar.set_postfix({"loss": losses.mean().item(),
                              "median": speed_ade.median().item(),
                              "max": speed_ade.max().item(),
                              "lin_ade": lin_ade.mean().item()})
            if len(losses) > 500:
                losses = losses[100:]
    im = vis_cur_and_fut(data, outputs)
    plt.imshow(im)

def main():
    
    rgb_loader = RgbLoader(config.train_index_path)
    train_multymodal(net, (train_loader, test_loader), optimizer, checkpointer=checkpointer,
                     num_ep=wandb.config["epochs"],
                     logger=wandb, use_every_nth_prediction=config.use_every_nth_prediction, scheduler=scheduler,
                     rgb_loader=rgb_loader)


if __name__ == "__main__":
    
    main()
    # import cProfile, pstats
    # data = next(iter(train_loader))
    # profiler = cProfile.Profile()
    # profiler.enable()
    # for i in tqdm(range(20)):
    #     net(data)
    # # main()
    # profiler.disable()
    # cProfile.run('')
    # stats = pstats.Stats(profiler)
    # stats.dump_stats('here')
    #
