import sys
from pathlib import Path
import glob
import pickle
import multiprocessing

sys.path.append(str(Path.cwd().parent))
# import tensorflow as tf
from scripts.iab import AttPredictorPecNet, AttPredictorPecNetWithType, AttPredictorPecNetWithTypeD3
from scripts.visualize import vis_cur_and_fut
from scripts.config import build_parser
from scripts.dataloaders import context_description, CustomImageDataset, d_collate_fn
from scripts.rgb_loader import RgbLoader
import math
# tf.get_logger().setLevel('ERROR')

import torch

from tqdm.auto import tqdm
import torch.utils.data

import matplotlib.pyplot as plt
import numpy as np
import wandb
import random
from scripts.train import preprocess_batch, apply_tr, get_future

random.seed(0)
torch.manual_seed(0)
np.random.seed(0)

import os
from scripts.iab_pl import AttPredictorPecNetWithTypeD3
from scripts.visualize import visualize_one_step_with_future, vis_cur_and_fut
import pathlib
from read_map_ds import WaymoDataset

class Config:
    def __init__(self):
        self.exp_use_points = 0
        self.exp_use_vis = 1
        self.exp_embed_dim = 256
        self.exp_use_rec = 0
        self.exp_inp_dim = 128
        self.exp_num_blocks = 4
        self.exp_lr = 1e-5
        self.exp_batch_size = 1
        self.dir_data = "/home/jovyan/uncompressed/tf_example"
        self.exp_num_workers = 0

config = Config()

# model = AttPredictorPecNetWithTypeD3(config=config, wandb_logger=None)

ds_path = config.dir_data
assert pathlib.Path(ds_path).exists()

index_file = "training_mapstyle/index_file.txt"
index_file = pathlib.Path(ds_path) / index_file
train_dataset = WaymoDataset(ds_path, index_file, rgb_index_path="/home/jovyan/rendered/train/index.pkl", rgb_prefix="/home/jovyan/")


train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.exp_batch_size,
                                           num_workers=config.exp_num_workers, collate_fn=d_collate_fn)


size_pixels = 224
import uuid
def fig_canvas_image(fig):
    """Returns a [H, W, 3] uint8 np.array image from fig.canvas.tostring_rgb()."""
    # Just enough margin in the figure to display xticks and yticks.
    fig.subplots_adjust(
        left=0.08, bottom=0.08, right=0.98, top=0.98, wspace=0.0, hspace=0.0)
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    return data.reshape(fig.canvas.get_width_height()[::-1] + (3,))


def create_figure_and_axes(size_pixels):
    """Initializes a unique figure and axes for plotting."""
    fig, ax = plt.subplots(1, 1, num=uuid.uuid4())

    # Sets output image to pixel resolution.
    dpi = 100
    size_inches = size_pixels / dpi
    fig.set_size_inches([size_inches, size_inches])
    fig.set_dpi(dpi)
    fig.set_facecolor('white')
    ax.set_facecolor('white')
    ax.xaxis.label.set_color('black')
    ax.tick_params(axis='x', colors='black')
    ax.yaxis.label.set_color('black')
    ax.tick_params(axis='y', colors='black')
    fig.set_tight_layout(True)
    ax.grid(False)
    ax.set_axis_off()
    ax.margins(x=0, y=0, tight=True)
    return fig, ax

def plot_scene_my(data, ag_id):
    fig, ax = create_figure_and_axes(size_pixels=size_pixels)
    yaw = data["state/current/bbox_yaw"][0,ag_id]
    rot_matrix = np.array(
                    [
                        [np.cos(yaw), -np.sin(yaw)],
                        [np.sin(yaw), np.cos(yaw)],
                    ]
                )
    hist = np.stack([data['state/past/x'].reshape(128, 10), data['state/past/y'].reshape(128,10)], axis=-1)
    cur = np.stack([data['state/current/x'].reshape(128), data['state/current/y'].reshape(128)], -1)
    state = np.concatenate([hist, cur.reshape(128,1,2)], 1)
    state = np.flip(state, 1)
    hist_start = state[ag_id][0]
    pix_to_m = 224 / 40
    shift_meters = np.array([224//6, 224//2])
    is_valid = np.concatenate([data['state/past/valid'].reshape(128,10), data['state/current/valid'].reshape(128,1)], -1)
    is_valid = np.flip(is_valid, 1)
    
    agent_is_valid = is_valid[ag_id]#.reshape(11,1)
    hist_img = ((state[ag_id] - hist_start)[agent_is_valid>0])  @ rot_matrix * pix_to_m + shift_meters
    xy_lines = (data["roadgraph_samples/xyz"].reshape(20000, 3)[:, :2] - hist_start) @ rot_matrix * pix_to_m + shift_meters
    others_img = (state - hist_start) @ rot_matrix * pix_to_m + shift_meters
    
    
    for i in range(11):
        ax.plot(others_img[:, i, 0], others_img[:,i, 1], "*g", markersize=4-(i*0.3),  alpha= 1 - i/20)

    for i in range(len(hist_img)-1):
        ax.plot(hist_img[:i+2, 0][i:], hist_img[:i+2:, 1][i:], "b", linewidth=6,  alpha=1 - i/13)

    ax.plot(xy_lines[:, 0], xy_lines[:,1], "ok", markersize=2.0)
    ax.plot(xy_lines[:, 0], xy_lines[:,1], "*k", markersize=1.)
    plt.xlim(0, 224)
    plt.ylim(0, 224)
    image = fig_canvas_image(fig)
    plt.close(fig)
    return image

def create_raster(data):
                rgb_d = {}
                for i in (data["state/tracks_to_predict"]>0).nonzero():
                    sid = data["scenario/id"].numpy().tobytes().decode("utf-8")
                    aid = data["state/id"][0][i[1]]
                    name = str(sid)+str(float(aid))
                    img = plot_scene_my(data, i[1])[np.newaxis]
                    rgb_d[name] = img
                return name, img

class RgbRendererMy:
    def __init__(self, index_path="renderedMy40/index.pkl", rgb_file_name_base="renderedMy40/rgb", train_loader=None):
        self.index_path = index_path
        self.rgb_file_name_base = rgb_file_name_base
        self.train_loader = train_loader

    def save_dict(self, di_, filename_=None):
        if filename_ is None:
            filename_ = self.index_path
        with open(filename_, 'ab') as f:
            pickle.dump(di_, f)

    def save_dataset(self, train_loader=None, index_file_name=None, rgb_file_name_base=None, rgb_file_name_index=None):
        
        if not train_loader:
            train_loader=self.train_loader
        if index_file_name is None:
            index_file_name = self.index_path
        if rgb_file_name_base is None:
            rgb_file_name_base = self.rgb_file_name_base
        
        if not rgb_file_name_index:
            rgb_file_name_index = len(glob.glob(self.rgb_file_name_base+"*"))
        print(f" staring index {rgb_file_name_index}")
        rgb_holder = {}

        index_dict = {}

        tli = iter(train_loader)
        for data in tqdm(tli):
            file_index = rgb_file_name_base + str(rgb_file_name_index) + ".npz"            
            name, img = create_raster(data)            
            d = {name: file_index}

            index_dict = {**d, **index_dict}

            rgb_holder = {**rgb_holder, **{name: img}}
            if len(rgb_holder) >= 200:
                np.savez_compressed(rgb_file_name_base + str(rgb_file_name_index), rgb=rgb_holder,
                         names=np.array(list(index_dict.keys())))
                rgb_file_name_index += 1
                rgb_holder = {}
                self.save_dict(index_dict, index_file_name)
                index_dict = {}
        self.save_dict(index_dict, index_file_name)
        np.savez_compressed(rgb_file_name_base + str(rgb_file_name_index), rgb=rgb_holder, names=np.array(list(index_dict.keys())))
        return

lnsp =np.linspace(start=0, stop=len(train_dataset), num=12)
dls = []
rends = []
indexes = []

for i in range(1, len(lnsp)):
    print(math.ceil(lnsp[i-1]), math.ceil(lnsp[i]))
    ds = torch.utils.data.Subset(train_dataset, range(math.ceil(lnsp[i-1]), math.ceil(lnsp[i])))
    dl = torch.utils.data.DataLoader(ds, batch_size=config.exp_batch_size,
                                           num_workers=config.exp_num_workers, collate_fn=d_collate_fn)
    
    rend = RgbRendererMy(index_path=f"renderedMy40/index+{i}.pkl", train_loader=dl) #rgb_file_name_index=
    rends.append(rend)
    indexes.append(i*1e7)
    dls.append(dl)
s = 0


def st(inp):
    rend, rgb_file_name_index = inp
    rend.save_dataset(rgb_file_name_index=rgb_file_name_index)

with multiprocessing.Pool(12) as p:
     result = p.map(st, zip(rends, indexes))        





