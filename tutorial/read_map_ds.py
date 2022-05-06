import numpy as np

from tqdm import tqdm
import pickle
import pathlib
import torch
from scripts.rgb_loader import RgbLoader

class WaymoDataset(torch.utils.data.Dataset):
    def __init__(self, ds_path: str, index_file: str, rgb_index_path: str = None, rgb_prefix: str = ""):
        self.ds_path = ds_path
        # assert self.ds_path.exists()
        assert pathlib.Path(self.ds_path).exists()
        self.index_file = index_file
        assert pathlib.Path(self.index_file).exists()
        self.index = None
        self.read_files = {}
        with open(index_file, 'rb') as f:
            self.index = pickle.load(f)
        self.rgb_index_path = rgb_index_path
        if rgb_index_path is not None:
            assert pathlib.Path(rgb_index_path).exists()
            self.rgb_prefix = rgb_prefix
            self.rgb_loader = RgbLoader(rgb_index_path)


    def __getitem__(self, item):
        path = self.index[item][0]
        frame_idx = self.index[item][1]
        # load np from path
        if path not in self.read_files:
            # join ds_path + path
            path_glob = pathlib.Path(self.ds_path) / path
            data = np.load(path_glob, allow_pickle=True)["data"].reshape(-1)[0]
            
            new_data = {}
            for key, val in data.items():
                new_sub_d = {}
                for vkey in data[key].keys():
                    if "roadgraph" not in vkey:
#                         del data[key][vkey]
                        new_sub_d[vkey]  = data[key][vkey]
                new_data[key] = new_sub_d
            data = new_data
            self.read_files[path] = new_data
        else:
            data = self.read_files[path]

        chunk = data[frame_idx]
        if self.rgb_index_path:
            rgb = torch.tensor(
                self.rgb_loader.load_singlebatch_rgb(chunk, prefix=self.rgb_prefix).astype(np.float32))
            chunk["rgbs"] = np.zeros((8,rgb.shape[1],rgb.shape[2], rgb.shape[3]), dtype=np.float32)
            chunk["rgbs"][:rgb.shape[0]] = rgb
            chunk["rgbs"] = chunk["rgbs"][np.newaxis]
        chunk["file"] = path
        if len(self.read_files)>12:
            self.read_files.clear()
#             print("clearing")
        return chunk

    def __len__(self):
        return len(self.index)

# main
if __name__ == "__main__":
    ds_path = pathlib.Path("/home/jovyan/uncompressed/tf_example/")
    assert pathlib.Path(ds_path).exists()

    index_file = "training_mapstyle/index_file.txt"
    # join
    index_file = pathlib.Path(ds_path) / index_file
    # join the path with the index file
    ds = WaymoDataset(ds_path, index_file, rgb_index_path="/home/jovyan/rendered/train/index.pkl", rgb_prefix="/home/jovyan/")
    print(f"len(ds): {len(ds)}")
    # for i in tqdm(range(len(ds))):
    #     data = ds[i]
    from scripts.dataloaders import d_collate_fn
    from scripts.rgb_loader import RgbLoader

#     rgb_loader = RgbLoader()
    loader = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=0,
                                              num_workers=12, collate_fn=d_collate_fn)
    # use tqdm to iterate over the loader
    good = 0
    bad = 0
    for i, data in tqdm(enumerate(loader),total=len(loader)):
        pass
    print(good, bad)
