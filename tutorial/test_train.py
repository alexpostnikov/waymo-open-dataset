# Data location. Please edit.

# A tfrecord containing tf.Example protos as downloaded from the Waymo dataset
# webpage.

# Replace this path with your own tfrecords.

import tensorflow as tf

from test.models import Model, SimplModel, MultyModel
from test.train import train, train_multymodal
from test.visualize import vis_cur_and_fut
from test.train import get_speed_ade_with_mask, get_ade_from_pred_speed_with_mask
from test.config import build_parser
import argparse

tf.get_logger().setLevel('ERROR')

import glob
from itertools import chain
import torch
from tfrecord.torch.dataset import TFRecordDataset
from torch.utils.data.dataset import T_co
from tqdm.auto import tqdm
import torch.utils.data

import matplotlib.pyplot as plt
import numpy as np
import wandb
import random
import torch.optim as optim

parser = build_parser()
config = parser.parse_args()
random.seed(config.np_seed)
torch.manual_seed(config.torch_seed)
np.random.seed(config.np_seed)

FILENAME = '/media/robot/hdd/waymo_dataset/tf_example/training/training_tfexample.tfrecord-00000-of-01000'
DATASET_FOLDER = '/path/to/waymo_open_dataset_motion_v_1_1_0/uncompressed'

# Example field definition
roadgraph_features = {
    'roadgraph_samples/dir':
        tf.io.FixedLenFeature([20000, 3], tf.float32, default_value=None),
    'roadgraph_samples/id':
        tf.io.FixedLenFeature([20000, 1], tf.int64, default_value=None),
    'roadgraph_samples/type':
        tf.io.FixedLenFeature([20000, 1], tf.int64, default_value=None),
    'roadgraph_samples/valid':
        tf.io.FixedLenFeature([20000, 1], tf.int64, default_value=None),
    'roadgraph_samples/xyz':
        tf.io.FixedLenFeature([20000, 3], tf.float32, default_value=None),
}

# Features of other agents.
state_features = {
    'state/id':
        tf.io.FixedLenFeature([128], tf.float32, default_value=None),
    'state/type':
        tf.io.FixedLenFeature([128], tf.float32, default_value=None),
    'state/is_sdc':
        tf.io.FixedLenFeature([128], tf.int64, default_value=None),
    'state/tracks_to_predict':
        tf.io.FixedLenFeature([128], tf.int64, default_value=None),
    'state/current/bbox_yaw':
        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    'state/current/height':
        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    'state/current/length':
        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    'state/current/timestamp_micros':
        tf.io.FixedLenFeature([128, 1], tf.int64, default_value=None),
    'state/current/valid':
        tf.io.FixedLenFeature([128, 1], tf.int64, default_value=None),
    'state/current/vel_yaw':
        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    'state/current/velocity_x':
        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    'state/current/velocity_y':
        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    'state/current/width':
        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    'state/current/x':
        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    'state/current/y':
        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    'state/current/z':
        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    'state/future/bbox_yaw':
        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/future/height':
        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/future/length':
        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/future/timestamp_micros':
        tf.io.FixedLenFeature([128, 80], tf.int64, default_value=None),
    'state/future/valid':
        tf.io.FixedLenFeature([128, 80], tf.int64, default_value=None),
    'state/future/vel_yaw':
        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/future/velocity_x':
        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/future/velocity_y':
        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/future/width':
        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/future/x':
        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/future/y':
        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/future/z':
        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/past/bbox_yaw':
        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    'state/past/height':
        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    'state/past/length':
        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    'state/past/timestamp_micros':
        tf.io.FixedLenFeature([128, 10], tf.int64, default_value=None),
    'state/past/valid':
        tf.io.FixedLenFeature([128, 10], tf.int64, default_value=None),
    'state/past/vel_yaw':
        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    'state/past/velocity_x':
        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    'state/past/velocity_y':
        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    'state/past/width':
        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    'state/past/x':
        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    'state/past/y':
        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    'state/past/z':
        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
}

traffic_light_features = {
    'traffic_light_state/current/state':
        tf.io.FixedLenFeature([1, 16], tf.int64, default_value=None),
    'traffic_light_state/current/valid':
        tf.io.FixedLenFeature([1, 16], tf.int64, default_value=None),
    'traffic_light_state/current/x':
        tf.io.FixedLenFeature([1, 16], tf.float32, default_value=None),
    'traffic_light_state/current/y':
        tf.io.FixedLenFeature([1, 16], tf.float32, default_value=None),
    'traffic_light_state/current/z':
        tf.io.FixedLenFeature([1, 16], tf.float32, default_value=None),
    'traffic_light_state/past/state':
        tf.io.FixedLenFeature([10, 16], tf.int64, default_value=None),
    'traffic_light_state/past/valid':
        tf.io.FixedLenFeature([10, 16], tf.int64, default_value=None),
    'traffic_light_state/past/x':
        tf.io.FixedLenFeature([10, 16], tf.float32, default_value=None),
    'traffic_light_state/past/y':
        tf.io.FixedLenFeature([10, 16], tf.float32, default_value=None),
    'traffic_light_state/past/z':
        tf.io.FixedLenFeature([10, 16], tf.float32, default_value=None),
}

context_description = {
    'roadgraph_samples/xyz': "float",
    "state/current/x": 'float',
    "state/current/y": 'float',
    "state/past/x": 'float',
    "state/past/y": 'float',
    "state/current/velocity_x": 'float',
    "state/current/velocity_y": 'float',
    "state/future/x": 'float',
    "state/future/y": 'float',
    "state/future/valid": 'int',
    "state/current/valid": "int",
    "state/past/valid": "int",
    "state/tracks_to_predict": "int",
}

# from pathlib import Path
import os


class CustomImageDataset(torch.utils.data.IterableDataset):
    def __init__(self, tf_dir, context_desription, transform=None, target_transform=None, ):
        # self.tf_dir = tf_dir
        self.context_desription = context_desription
        self.tf_files = glob.glob(tf_dir)
        self.transform = transform
        self.target_transform = target_transform
        self.cur_file_index = 0

    def __iter__(self):
        self.dataset = TFRecordDataset(self.tf_files[0], index_path=None, description=self.context_desription)
        self.iterator = iter(self.dataset)
        for file in self.tf_files[1:]:
            dataset = TFRecordDataset(file, index_path=None, description=self.context_desription)
            self.iterator = chain(self.iterator, iter(dataset))

        return self.iterator

    def __getitem__(self, index) -> T_co:
        pass

    def __next_file(self):
        if (self.cur_file_index + 1 < len(self.tf_files)):
            self.cur_file_index += 1
            self.dataset = TFRecordDataset(self.tf_files[self.cur_file_index],
                                           index_path=None,
                                           description=self.context_desription)
            self.iterator = iter(self.dataset)
            return
        raise StopIteration


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('integers', metavar='N', type=int, nargs='+', default=1,
                    help='an integer for the accumulator')
parser.add_argument('--sum', dest='accumulate', action='store_const',
                    const=sum, default=max,
                    help='sum the integers (default: find the max)')

wandb.init(project="waymo22", entity="aleksey-postnikov", name=config.exp_name)
wandb.config = {
    "learning_rate": config.exp_lr,
    "epochs": config.exp_num_epochs,
    "batch_size": config.exp_batch_size
}

batch_size = config.exp_batch_size

train_tfrecord_path = os.path.join(config.dir_data, "training/training_tfexample.*-of-01000")
test_path = os.path.join(config.dir_data, "testing/testing_tfexample.tfrecord-*-of-00150")
# "/media/robot/hdd/waymo_dataset/tf_example/training/"
train_dataset = CustomImageDataset(train_tfrecord_path, context_description)
test_dataset = CustomImageDataset(test_path, context_description)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=0)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=0)

device = "cuda"
net = MultyModel()
# net = SimplModel()
optimizer = optim.Adam(net.parameters(), lr=wandb.config["learning_rate"])
net = net.to(device)

# data = next(iter(dataset))


from test.models import Checkpointer

checkpointer = Checkpointer(model=net, torch_seed=0, ckpt_dir=config.dir_checkpoint, checkpoint_frequency=1)
checkpointer.load(config.epoch_to_load)


def overfit_test(model, loader, optimizer):
    losses = torch.rand(0)
    pbar = tqdm(loader)
    data = next(iter(pbar))
    pbar = tqdm(range(200))
    for chank in pbar:
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
            pbar.set_description("ep %s chank %s" % (0, chank))
            pbar.set_postfix({"loss": losses.mean().item(),
                              "median": speed_ade.median().item(),
                              "max": speed_ade.max().item(),
                              "lin_ade": lin_ade.mean().item()})
            if len(losses) > 500:
                losses = losses[100:]
    im = vis_cur_and_fut(data, outputs)
    plt.imshow(im)


train_multymodal(net, (train_loader, test_loader), optimizer, checkpointer=checkpointer, num_ep=wandb.config["epochs"], logger=wandb)
print("done")
print("done")
