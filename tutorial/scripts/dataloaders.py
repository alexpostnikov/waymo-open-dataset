import collections
import glob
from itertools import chain

import tensorflow as tf
import torch
import torch.utils
from tfrecord.torch import TFRecordDataset
import numpy as np
import re
from scripts.GAT import create_connectivity_graph
from typing import Tuple
np_str_obj_array_pattern = re.compile(r'[SaUO]')
string_classes = (str, bytes)

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
    'roadgraph_samples/xyz': "float",
    "roadgraph_samples/id": "int",
    "roadgraph_samples/type": "int",
    "roadgraph_samples/valid": "int",

    "scenario/id": "byte",
    "state/past/vel_yaw": "float",
    "state/current/vel_yaw": "float",
    "state/past/bbox_yaw": "float",
    "state/current/bbox_yaw": "float",
    "state/id" : "float",
    "state/type" : "float",
    "traffic_light_state/current/valid": "int",
    "state/current/width": "float",
    "state/current/length": "float",
    "traffic_light_state/current/state": "int",

}


class CustomImageDataset(torch.utils.data.IterableDataset):
    def __init__(self, tf_dir, context_desription, transform=None, target_transform=None):

        self.context_desription = context_desription

        self.tf_files = glob.glob(tf_dir)
        print(f"train dataset containing {len(self.tf_files)} files")

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

    def __getitem__(self, index):
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


def d_collate_fn(batch):
        r"""Puts each data field into a tensor with outer dimension batch size"""

        elem = batch[0]
        elem_type = type(elem)
        if isinstance(elem, torch.Tensor):
            out = None
            if torch.utils.data.get_worker_info() is not None:
                # If we're in a background process, concatenate directly into a
                # shared memory tensor to avoid an extra copy
                numel = sum(x.numel() for x in batch)
                storage = elem.storage()._new_shared(numel)
                out = elem.new(storage)
            try:
                return torch.stack(batch, 0, out=out)
            except:
                scenario_ids = [sc.numpy().tobytes().decode("utf-8") for sc in batch]
                return scenario_ids
                # scenarios_id = []
                # for bn, scenario in enumerate(scenario_id):
                #     [scenarios_id.append(scenario) for i in range((mask.nonzero()[:, 0] == bn).sum())]
        elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
                and elem_type.__name__ != 'string_':
            if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
                # array of string classes and object
                if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                    raise #TypeError(default_collate_err_msg_format.format(elem.dtype))
                if batch[0].flags["WRITEABLE"]:
                    return d_collate_fn([torch.as_tensor(b) for b in batch])
                else:
                    return d_collate_fn([torch.as_tensor(np.copy(b)) for b in batch])
            elif elem.shape == ():  # scalars
                return torch.as_tensor(batch)
        elif isinstance(elem, float):
            return torch.tensor(batch, dtype=torch.float64)
        elif isinstance(elem, int):
            return torch.tensor(batch)
        elif isinstance(elem, string_classes):
            return batch
        elif isinstance(elem, collections.abc.Mapping):
            return {key: d_collate_fn([d[key] for d in batch]) for key in elem}
        elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
            return elem_type(*(d_collate_fn(samples) for samples in zip(*batch)))
        elif isinstance(elem, collections.abc.Sequence):
            # check to make sure that the elements in batch have consistent size
            it = iter(batch)
            elem_size = len(next(it))
            if not all(len(elem) == elem_size for elem in it):
                raise RuntimeError('each element in list of batch should be of equal size')
            transposed = zip(*batch)
            return [d_collate_fn(samples) for samples in transposed]

        raise #TypeError(default_collate_err_msg_format.format(elem_type))


def prepare_data_for_gat(data: dict, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Prepare data for GAT model.

    :param data: dict of data
    :param device: device to use
    :return: tuple of (states, graph)
    """
    # from data get "state\type" with shape bs, 1

    state_type = data['state/type']
    bs = state_type.shape[0]
    # state_type shape bs, 1
    # from data get "state\current\x" and "state\current\y" and cat them
    state_cur = torch.cat([data["state/current/x"].unsqueeze(-1), data["state/current/y"].unsqueeze(-1)], dim=-1)
    # state_cur shape is (batch_size, 2*state_size)
    # unsqueeze to be able to concatenate with state_past
    state_cur = state_cur.unsqueeze(2)
    # from data get "state\past\x" and "state\past\y" and cat them
    state_past = torch.cat([data["state/past/x"].reshape(bs, -1, 10, 1), data["state/past/y"].reshape(bs, -1, 10, 1)], dim=-1)
    # state_past shape is (batch_size, num_obs, 2*state_size)
    # cat cur and past states
    state = torch.cat([state_cur, state_past], dim=2)
    # state_type unsqueeze and repeat to cat with state
    state_type = state_type.unsqueeze(2).repeat(1, 1, state.size(2)).unsqueeze(-1)
    # state_type shape is (batch_size, num_obs, 1)
    # cat state and state_type
    state = torch.cat([state, state_type], dim=-1)
    # state shape is (batch_size, num_obs, 2*state_size+1)
    state = state.reshape(bs, state.shape[1], -1)
    graph = create_connectivity_graph(state)
    return state.to(device), graph.to(device)


