import sys
from pathlib import Path

sys.path.append(str(Path.cwd().parent))
import tensorflow as tf

from scripts.iab import AttPredictorPecNet
from scripts.train import train_multymodal
from scripts.config import build_parser
from scripts.models import Checkpointer
from scripts.dataloaders import context_description, CustomImageDataset, d_collate_fn
from scripts.rgb_loader import RgbLoader

tf.get_logger().setLevel('ERROR')

import torch.utils.data


import numpy as np
import wandb
import random
import torch.optim as optim
from transformers import get_cosine_with_hard_restarts_schedule_with_warmup



parser = build_parser()
config = parser.parse_args()
random.seed(config.np_seed)
torch.manual_seed(config.torch_seed)
np.random.seed(config.np_seed)

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
                         use_points=config.exp_use_points, out_horiz=80 // config.use_every_nth_prediction)
net = torch.nn.DataParallel(net)

optimizer = optim.Adam(net.parameters(), lr=wandb.config["learning_rate"])
scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=20,
    num_training_steps=(22000 * 128 / config.exp_batch_size) * wandb.config["epochs"],
    num_cycles=wandb.config["epochs"])
net = net.to(device)

checkpointer = Checkpointer(model=net, torch_seed=0, ckpt_dir=config.dir_checkpoint, checkpoint_frequency=1)
net = checkpointer.load(config.epoch_to_load)


def main():
    rgb_loader = RgbLoader(config.train_index_path)
    train_multymodal(net, (train_loader, test_loader), optimizer, checkpointer=checkpointer,
                     num_ep=wandb.config["epochs"],
                     logger=wandb, use_every_nth_prediction=config.use_every_nth_prediction, scheduler=scheduler,
                     rgb_loader=rgb_loader)


if __name__ == "__main__":
    main()
