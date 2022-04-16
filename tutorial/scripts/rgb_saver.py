from scripts.rasterization import rasterize_batch
from tqdm import tqdm
import numpy as np
from six.moves import cPickle as pickle


def save_dict(di_, filename_):
    with open(filename_, 'ab') as f:
        pickle.dump(di_, f)


def load_dict(filename_):
    out_d = {}
    with open(filename_, 'rb') as f:
        try:
            while 1:
                loaded_dict = pickle.load(f)
                out_d = {**out_d, **loaded_dict}
        except EOFError:
            pass

    return out_d


def get_rgb_file_names_from_batch(data):
    batch = data['scenario/id']
    try:
        scenario_ids = [sc.numpy().tobytes().decode("utf-8") for sc in batch]
    except Exception:
        scenario_ids = batch

    mask = (data["state/tracks_to_predict"] > 0)
    scenarios_id = []
    for bn, scenario in enumerate(scenario_ids):
        [scenarios_id.append(scenario) for _ in range((mask.nonzero()[:, 0] == bn).sum())]

    aids = data["state/id"][mask]
    names = [scenarios_id[i] + str(aids[i].item()) for i in range(len(aids))]
    return names


def create_rasters(data, file_ind):

    rasterized = rasterize_batch(data, True)
    summed_cur = np.concatenate(rasterized)[:, :, :, 3:14].sum(-1)
    summed_neigh = np.concatenate(rasterized)[:, :, :, 14:].sum(-1)
    rgb = np.concatenate(rasterized)[:, :, :, :3]
    rgb[:, :, :, 1] += -np.clip(summed_cur, 0, 200)
    rgb[:, :, :, 1] = np.clip(rgb[:, :, :, 1], 0, 255)
    rgb[:, :, :, 0] += -np.clip(summed_neigh, 0, 200)
    rgb[:, :, :, 0] = np.clip(rgb[:, :, :, 0], 0, 255)

    # prepare SIDs and AIDs
    names = get_rgb_file_names_from_batch(data)
    for_indexing = {}

    for name in names:
        for_indexing[name] = file_ind
    out_rgb = {}
    for i, name in enumerate(names):
        out_rgb[name] = 255 - rgb[i:i + 1]

    return out_rgb, for_indexing


class RgbRenderer:
    def __init__(self, index_path="rendered/index.pkl", rgb_file_name_base="rendered/rgb"):
        self.index_path = index_path
        self.rgb_file_name_base = rgb_file_name_base

    def save_dict(self, di_, filename_=None):
        if filename_ is None:
            filename_ = self.index_path
        with open(filename_, 'ab') as f:
            pickle.dump(di_, f)

    def save_dataset(self, train_loader, index_file_name=None, rgb_file_name_base=None, ):
        if index_file_name is None:
            index_file_name = self.index_path
        if rgb_file_name_base is None:
            rgb_file_name_base = self.rgb_file_name_base

        rgb_file_name_index = 0

        rgb_holder = {}

        index_dict = {}

        tli = iter(train_loader)
        for data in tqdm(tli):
            rasterized, d = create_rasters(data, rgb_file_name_base + str(rgb_file_name_index) + ".npz")
            index_dict = {**d, **index_dict}

            rgb_holder = {**rgb_holder, **rasterized}
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
