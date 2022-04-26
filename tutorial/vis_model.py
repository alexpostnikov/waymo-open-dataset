import sys
from pathlib import Path

sys.path.append(str(Path.cwd().parent))
import tensorflow as tf
from scripts.iab import AttPredictorPecNet
from scripts.visualize import vis_cur_and_fut
from scripts.config import build_parser
from scripts.dataloaders import context_description, CustomImageDataset, d_collate_fn
from scripts.rgb_loader import RgbLoader

tf.get_logger().setLevel('ERROR')

import torch

from tqdm.auto import tqdm
import torch.utils.data

import matplotlib.pyplot as plt
import numpy as np
import wandb
import random
from scripts.train import preprocess_batch, apply_tr



parser = build_parser()
config = parser.parse_args()
random.seed(config.np_seed)
torch.manual_seed(config.torch_seed)
np.random.seed(config.np_seed)

import os


batch_size = 1

train_tfrecord_path = os.path.join(config.dir_data, "training/training_tfexample.*-of-01000")

train_dataset = CustomImageDataset(train_tfrecord_path, context_description)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                           num_workers=config.exp_num_workers, collate_fn=d_collate_fn)

device = "cuda"

net = AttPredictorPecNet(inp_dim=config.exp_inp_dim, embed_dim=config.exp_embed_dim, num_blocks=config.exp_num_blocks,
                         out_modes=6, use_vis=config.exp_use_vis, use_rec=config.exp_use_rec,
                         use_points=config.exp_use_points, out_horiz=80 // config.use_every_nth_prediction)
net = torch.nn.DataParallel(net)

net = net.to(device)
net.eval()

def main():

    net.load_state_dict(
        torch.load("/media/robot/hdd1/waymo-open-dataset/tutorial/checkpoints/model-seed-0-epoch-2.pt", map_location="cuda"))
    print("model loaded")

    rgb_loader = RgbLoader(config.train_index_path)
    # use tqdm to iterate data
    for i, data in tqdm(enumerate(train_loader)):
        # add rgb to data
        data["rgbs"] = torch.tensor(rgb_loader.load_batch_rgb(data, prefix="").astype(np.float32))
        # preprocess_batch
        data = preprocess_batch(data, net.module.use_points, net.module.use_vis)
        # get predictions
        poses, confs, goals_local, rot_mat, rot_mat_inv = net(data)
        poses = apply_tr(poses, rot_mat_inv)
        image = vis_cur_and_fut(data, poses.detach().cpu(), confs=confs.detach().cpu())
        # visualize
        plt.imshow(image)
        plt.show()


    # poses = apply_tr(poses, rot_mat_inv)
    # image = vis_cur_and_fut(data, poses.detach().cpu(), confs=confs.detach().cpu())
    # images = wandb.Image(image, caption="Top: Output, Bottom: Input")
    # wandb.log({"examples": images})
if __name__ == "__main__":
    main()