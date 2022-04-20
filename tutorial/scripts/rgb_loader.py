import numpy as np
from six.moves import cPickle as pickle
import logging




class RgbLoader():
    def __init__(self, index_path="rendered/index.pkl"):
        # logging.basicConfig(filename='example.log', filemode='w', level=logging.DEBUG)
        # logging.info('------------------Started------------------')

        self.index_path = index_path
        self.index_dict = self.load_dict()
        self.opened_files = {}
        self.rgbs = {}

#     def load_dict(self, filename_=None):
#         if filename_ is None:
#             filename_ = self.index_path

#         out_d = {}
#         with open(filename_, 'rb') as f:
#             try:
#                 lim = 1e4
#                 while 1 and lim > 0:
#                     lim -= 1
#                     loaded_dict = pickle.load(f)
#                     out_d = {**out_d, **loaded_dict}
#             except EOFError:
#                 pass
#         return out_d
    
    def load_dict(self, filename_=None):
        if filename_ is None:
            filename_ = self.index_path
        out_d = {}
        with open(filename_, 'rb') as f:
            d = {}
            try:
                lim = 1e5
                dicts = []
                while 1 and lim > 0:
                    lim -= 1
                    loaded_dict = pickle.load(f)
                    dicts.append(loaded_dict)
                    # out_d = {**out_d, **loaded_dict}
                    
            except EOFError:
                pass
        d = {k:v for e in dicts for (k,v) in e.items()}
        print(f"loaded index file contains {1e5-lim} indexes to files")
        return d

    def namefile_to_file_name_index(self, names, files):
        out_files = {}
        for i, (name, file) in enumerate(zip(names, files)):
            if file in out_files:
                out_files[file].append([name, i])
            else:
                out_files[file] = [[name, i]]
        return out_files

    def load_batch_rgb(self, data, prefix="tutorial/"):
        batch = data['scenario/id']
        try:
            scenario_ids = [sc.cpu().numpy().tobytes().decode("utf-8") for sc in batch]
        except:
            scenario_ids = batch
        # logging.info(f'----load_batch_rgb() scenario_ids = {scenario_ids}')
        mask = (data["state/tracks_to_predict"] > 0)
        scenarios_id = []
        for bn, scenario in enumerate(scenario_ids):
            num_nonzero_in_each_batch = (mask.nonzero()[:, 0] == bn).sum()
            [scenarios_id.append(scenario) for _ in range(num_nonzero_in_each_batch)]
        
        aids = data["state/id"][mask]
        names = [scenarios_id[i] + str(aids[i].item()) for i in range(len(aids))]
        # logging.info(f'----load_batch_rgb() names = {names}')
        files = [self.index_dict[name] for name in names]
        file_name_index = self.namefile_to_file_name_index(names, files)
        batch_rgb = np.random.rand(len(aids), 224, 224, 3)
        for file, name_index in file_name_index.items():
            indexes = [ni[1] for ni in name_index]
            # logging.info(f' --------load_batch_rgb() name_index = {name_index}, prefix= {prefix}')
            batch_rgb[indexes] = self.load_rgb_by_name_file(name_index, file, prefix)
        return batch_rgb

    def load_rgb_by_name_file(self, name_index, file_path, prefix="tutorial/"):
        if file_path not in self.opened_files:
            # logging.info(f'----file_path: {prefix + file_path}')
            self.rgbs = {**self.rgbs, **np.load(prefix + file_path, allow_pickle=True)["rgb"].reshape(-1)[0]}
            self.opened_files[file_path] = 1
        out = []
        for n_i in name_index:
            # logging.info(f'--------load_rgb_by_name_file() n_i = {n_i}')
            out.append(self.rgbs[n_i[0]])
        out = np.concatenate([out])
        return out[:, 0]

