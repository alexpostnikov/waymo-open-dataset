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
# net = torch.nn.DataParallel(net)


optimizer = optim.Adam(net.parameters(), lr=wandb.config["learning_rate"])
scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=20,
            num_training_steps=(22000*128 / config.exp_batch_size) * wandb.config["epochs"],
            num_cycles=wandb.config["epochs"]) 
net = net.to(device)

checkpointer = Checkpointer(model=net, torch_seed=0, ckpt_dir=config.dir_checkpoint, checkpoint_frequency=1)
net = checkpointer.load(config.epoch_to_load)



def overfit_test(model, loader, optimizer):
    losses = torch.rand(0)
    pbar = tqdm(loader)
    data = next(iter(pbar))
    pbar = tqdm(range(200))
    for chank in pbar:
        optimizer.zero_grad()
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
