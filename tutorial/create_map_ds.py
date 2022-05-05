import numpy as np
import tfrecord
from scripts.dataloaders import context_description
import glob
from tqdm import tqdm
import pickle
import pathlib




'''
1 file ~ 480 records
each recors is dict of numpy arrays
each numpy array is a batch of data


crete index file for each batch and save it to disk
index file is a list pathes to each batch

save batches to disk and save index file to disk

for each file in files create loader and save data to disk
'''

train_data_path = "/home/jovyan/uncompressed/tf_example/training/"
files = glob.glob(train_data_path + "*")

ds_path = "/home/jovyan/uncompressed/tf_example"
# check that ds_path exists
assert pathlib.Path(ds_path).exists()

index_path = "training_mapstyle/index_file.txt"
path_npz_files = "training_mapstyle/npz_files"


index_path = pathlib.Path(ds_path) / index_path
# check that index_path exists if not create it
# if not index_path.exists():
#     index_path.mkdir(parents=True)

path_npz_files = pathlib.Path(ds_path) / path_npz_files
# check that path_npz_files exists if not create it
if not path_npz_files.exists():
    path_npz_files.mkdir(parents=True)

# holder of dicts

indexes = {}

global_index = 0
for fn, file in enumerate(tqdm(files)):
    # stop after second file
    # if fn > 30:
        # break

    holder = {}
    loader = tfrecord.tfrecord_loader(file, None, context_description)
    path_batch = pathlib.Path(path_npz_files) / f"batch_{fn}.npz"
    # get relative to ds_path
    path_batch_relative = path_batch.relative_to(pathlib.Path(ds_path))
    for i, data in enumerate(loader):
        # save batch to disk

        # append data to holder
        # rgb_loader.load_rgb_by_name_file(data)
        holder[i] = data
        # save index file
        indexes[global_index] = [str(path_batch_relative), i]
        global_index += 1

    np.savez_compressed(path_batch, data=holder)
    # clear holder
    holder.clear()
# save index file

# save index file with pickle
with open(index_path, "wb") as f:
    pickle.dump(indexes, f)
