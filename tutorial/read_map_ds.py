import numpy as np
from tqdm import tqdm
import pickle
import pathlib




# # assert
# with open(index_file, 'rb') as f:
#     index = pickle.load(f)
#
# # index - list of path to file and index of frame
# # choose random index from index file
# read_files = {}
# indexes = np.random.choice(len(index), size=10, replace=False)
# # load data from pb file
# for i in tqdm(indexes):
#     path = index[i][0]
#     frame_idx = index[i][1]
#     # load np from path
#     if path not in read_files:
#         # join ds_path + path
#         path_glob = pathlib.Path(ds_path) / path
#         data = np.load(path_glob, allow_pickle=True)["data"].reshape(-1)[0]
#         read_files[path] = data
#     else:
#         data = read_files[path]
#     chunk = data[frame_idx]
# pass

import torch


class WaymoDataset(torch.utils.data.IterableDataset):
    def __init__(self, ds_path: str, index_file: str):
        self.ds_path = ds_path
        # assert self.ds_path.exists()
        assert pathlib.Path(self.ds_path).exists()
        self.index_file = index_file
        assert pathlib.Path(self.index_file).exists()
        self.index = None
        self.read_files = {}
        with open(index_file, 'rb') as f:
            self.index = pickle.load(f)

    def __getitem__(self, item):
        path = self.index[item][0]
        frame_idx = self.index[item][1]
        # load np from path
        if path not in self.read_files:
            # join ds_path + path
            path_glob = pathlib.Path(self.ds_path) / path
            data = np.load(path_glob, allow_pickle=True)["data"].reshape(-1)[0]
            self.read_files[path] = data
        else:
            data = self.read_files[path]
        chunk = data[frame_idx]
        return chunk

    def __len__(self):
        return len(self.index)

# main
if __name__ == "__main__":
    ds_path = pathlib.Path("/media/robot/hdd1/waymo_ds")
    assert pathlib.Path(ds_path).exists()

    index_file = "training_mapstyle/index_file.txt"
    # join
    index_file = pathlib.Path(ds_path) / index_file
    # join the path with the index file
    ds = WaymoDataset(ds_path, index_file)
    for i in tqdm(ds):
        pass
