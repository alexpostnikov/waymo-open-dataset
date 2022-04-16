import tensorflow as tf
from scripts.config import build_parser
from scripts.dataloaders import context_description, CustomImageDataset, d_collate_fn
from scripts.rgb_saver import RgbRenderer
tf.get_logger().setLevel('ERROR')
import torch.utils.data
import numpy as np
import random


parser = build_parser()
config = parser.parse_args()
random.seed(config.np_seed)
torch.manual_seed(config.torch_seed)
np.random.seed(config.np_seed)

# Example field definition

# Features of other agents.

# from pathlib import Path
import os



batch_size = config.exp_batch_size

train_tfrecord_path = os.path.join(config.dir_data, "training/training_tfexample.*-of-01000")
val_path = os.path.join(config.dir_data, "validation/validation_tfexample.tfrecord-*-of-00150")
train_dataset = CustomImageDataset(train_tfrecord_path, context_description)
val_dataset = CustomImageDataset(val_path, context_description)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                           num_workers=config.exp_num_workers, collate_fn=d_collate_fn)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,
                                          num_workers=config.exp_num_workers, collate_fn=d_collate_fn)

# index_path="rendered/train/index.pkl", rgb_file_name_base="rendered/train/rgb"

rgb_rend = RgbRenderer(index_path="rendered/train/index.pkl", rgb_file_name_base="rendered/train/rgb")
rgb_rend.save_dataset(train_loader)

rgb_rend = RgbRenderer(index_path="rendered/val/index.pkl", rgb_file_name_base="rendered/val/rgb")
rgb_rend.save_dataset(val_loader)


print("done")
print("done")
