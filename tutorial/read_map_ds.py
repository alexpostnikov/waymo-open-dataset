import numpy as np

from tqdm import tqdm
import pickle
import pathlib
import torch


class WaymoDataset(torch.utils.data.Dataset):
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
        chunk["file"] = path
        if len(self.read_files)>100:
            self.read_files.clear()
            print("clearing")
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
    ds = WaymoDataset(ds_path, index_file)
    print(f"len(ds): {len(ds)}")
    # for i in tqdm(range(len(ds))):
    #     data = ds[i]
    from scripts.dataloaders import d_collate_fn
    from scripts.rgb_loader import RgbLoader

    rgb_loader = RgbLoader("/home/jovyan/rendered/train/index.pkl")
    loader = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=0,
                                              num_workers=12, collate_fn=d_collate_fn)
    # use tqdm to iterate over the loader
    good = 0
    bad = 0
    for i, data in tqdm(enumerate(loader),total=len(loader)):
        try:
            rgb = rgb_loader.load_batch_rgb(data, prefix="/home/jovyan/")
            good+=1
        except Exception as e:
            print(data["file"])
            bad+=1
    print(good, bad)
