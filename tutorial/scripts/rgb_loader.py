import numpy as np
from six.moves import cPickle as pickle


class RgbLoader():
    def __init__(self, index_path="rendered/index.pkl"):
        self.index_path = index_path
        self.index_dict = self.load_dict()
        self.opened_files = {}
        self.rgbs = {}

    def load_dict(self, filename_=None):
        if filename_ is None:
            filename_ = self.index_path

        out_d = {}
        with open(filename_, 'rb') as f:
            try:
                lim = 1e4
                while 1 and lim > 0:
                    lim -= 1
                    loaded_dict = pickle.load(f)
                    out_d = {**out_d, **loaded_dict}
            except EOFError:
                pass
        return out_d

    def namefile_to_file_name_index(self, names, files):
        out_files = {}
        for i, (name, file) in enumerate(zip(names, files)):
            if file in out_files:
                out_files[file].append([name, i])
            else:
                out_files[file] = [[name, i]]
        return out_files

    def load_batch_rgb(self, data):
        batch = data['scenario/id']
        try:
            scenario_ids = [sc.cpu().numpy().tobytes().decode("utf-8") for sc in batch]
        except:
            scenario_ids = batch
        mask = (data["state/tracks_to_predict"] > 0)
        scenarios_id = []
        for bn, scenario in enumerate(scenario_ids):
            [scenarios_id.append(scenario) for i in range((mask.nonzero()[:, 0] == bn).sum())]

        aids = data["state/id"][mask]
        names = [scenarios_id[i] + str(aids[i].item()) for i in range(len(aids))]
        files = [self.index_dict[name] for name in names]
        file_name_index = self.namefile_to_file_name_index(names, files)
        batch_rgb = np.random.rand(len(aids), 224, 224, 3)
        for file, name_index in file_name_index.items():
            indexes = [ni[1] for ni in name_index]
            batch_rgb[indexes] = self.load_rgb_by_name_file(name_index, file)
        return batch_rgb

    def load_rgb_by_name_file(self, name_index, file_path):
        if file_path not in self.opened_files:
            self.opened_files[file_path] = 1
            self.rgbs = {**self.rgbs, **np.load(file_path, allow_pickle=True)["rgb"].reshape(-1)[0]}

        out = []
        for n_i in name_index:
            out.append(self.rgbs[n_i[0]])
        out = np.concatenate([out])
        return out[:, 0]
