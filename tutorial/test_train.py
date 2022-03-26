# Data location. Please edit.

# A tfrecord containing tf.Example protos as downloaded from the Waymo dataset
# webpage.

# Replace this path with your own tfrecords.

import tensorflow as tf

from tutorial.test.models import Model
from tutorial.test.train import train
from tutorial.test.visualize import vis_cur_and_fut

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

# torch.use_deterministic_algorithms(True)
import random
random.seed(0)
torch.manual_seed(0)
np.random.seed(0)

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


class CustomImageDataset(torch.utils.data.IterableDataset):

    def __init__(self, tf_dir, context_desription, transform=None, target_transform=None):
        self.tf_dir = tf_dir
        self.context_desription = context_desription
        self.tf_files = glob.glob("/media/robot/hdd/waymo_dataset/tf_example/training/training_tfexample.*-of-01000")
        self.transform = transform
        self.target_transform = target_transform
        self.cur_file_index = 0
        self.dataset = TFRecordDataset(self.tf_files[0], index_path=None, description=self.context_desription)
        self.iterator = iter(self.dataset)

    def __iter__(self):
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


def get_valid_data_mask(data):
    bs = data["state/future/x"].shape[0]
    fut_valid = data["state/future/valid"].reshape(bs, 128, -1) * (data["state/current/valid"].reshape(bs, 128, -1) > 0)
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
    mask = data["state/tracks_to_predict"].reshape(-1, 128, 1).repeat(1, 1, 80) * valid
    mask = mask.permute(0, 2, 1)
    dist_masked = dist[mask > 0]
    return dist_masked


def ade_loss(data, pred):
    ade = get_speed_ade_with_mask(data, pred)
    return ade.mean()


batch_size = 8

tfrecord_path = "/media/robot/hdd/waymo_dataset/tf_example/training/"
dataset = CustomImageDataset(tfrecord_path, context_description)
loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
import torch.optim as optim

device = "cuda"
net = Model()
optimizer = optim.Adam(net.parameters(), lr=3e-4)
net = net.to(device)

data = next(iter(dataset))

# images = visualize_all_agents_smooth(data)


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


# overfit_test(net, loader, optimizer)
train(net, loader, optimizer)
print("done")
print("done")
