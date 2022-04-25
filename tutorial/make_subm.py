import sys
from pathlib import Path

sys.path.append(str(Path.cwd().parent))
import tensorflow as tf

from scripts.iab import AttPredictorPecNet
from scripts.train import create_subm
from scripts.visualize import vis_cur_and_fut
from scripts.train import get_speed_ade_with_mask, get_ade_from_pred_speed_with_mask
from scripts.config import build_parser
from scripts.models import Checkpointer
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
import torch.optim as optim
from transformers import get_cosine_with_hard_restarts_schedule_with_warmup

parser = build_parser()
config = parser.parse_args()
random.seed(config.np_seed)
torch.manual_seed(config.torch_seed)
np.random.seed(config.np_seed)

import os


batch_size = config.exp_batch_size

test_path = os.path.join(config.dir_data, "validation/validation_tfexample.tfrecord-*-of-00150")
test_dataset = CustomImageDataset(test_path, context_description)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                          num_workers=config.exp_num_workers, collate_fn=d_collate_fn)

device = "cuda"

net = AttPredictorPecNet(inp_dim=config.exp_inp_dim, embed_dim=config.exp_embed_dim, num_blocks=config.exp_num_blocks,
                         out_modes=6, use_vis=config.exp_use_vis, use_rec=config.exp_use_rec,
                         use_points=config.exp_use_points, out_horiz=80 // config.use_every_nth_prediction)
net = torch.nn.DataParallel(net)

net = net.to(device)

checkpointer = Checkpointer(model=net, torch_seed=0, ckpt_dir=config.dir_checkpoint, checkpoint_frequency=1)
net = checkpointer.load(config.epoch_to_load)

def main():
    try:
        net.load_state_dict(
            torch.load("/home/jovyan/waymo-open-dataset/tutorial/checkpoints/model-seed-0-epoch-6.pt", map_location="cuda"))
        print("model loaded")
    except:
        print("fake subm!!!")
        print("fake subm!!!")
        print("fake subm!!!")
    rgb_loader = RgbLoader(config.test_index_path)
    create_subm(net, test_loader, rgb_loader, out_file=config.subm_file_path)


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
