import dlib 
from torch.utils.data import Dataset, DataLoader
import json
import os
import pickle
from progressbar import *
from skimage import io
import torch

class Options():
    def __init__(self,
                 num_workers = 4,
                 datapath: str,
                 bs = 20,
                 test_overlapped = False,
                 data_augmentation_temporal = False,
                 normalise = 1,
                 frame_rate = 25,
                 frame_skip = 1,
                 min_timesteps = 2,
                 max_timesteps = 75,
                 mode_img = "mouth",
                 debug = False):
        
        self.num_workers = num_workers
        self.datapath = datapath
        self.bs = bs
        self.test_overlapped = test_overlapped
        self.data_augmentation_temporal = data_augmentation_temporal
        self.normalise = normalise
        self.frame_rate = frame_rate
        self.frame_skip = frame_skip
        self.min_timesteps = min_timesteps
        self.max_timesteps = max_timesteps
        self.mode_img = mode_img
        self.debug = debug

        

class MyDataset(Dataset):
    def __init__(self, opt, dataset_type = "train"):
        self.opt = opt
        self.dataset_type = dataset_type
        self.overlap_list = json.load(open(self.opt.overlap_list, "r").read())
        self.dataset = []

    def load_data(self):
        opt = self.opt
        unsorted_vocab = {}
        unsorted_vocab[' '] = True
        
        print(f"Loading data from {self.dataset_type} dataset")
        cache_path = '{}_{}.pkl'.format(self.dataset_type, "overlapped" if opt.test_overlapped else "unseen")
        if os.path.exists(cache_path):
            self.dataset, count_videos, self.vocab, self.vocab_mapping = pickle.load(open(cache_path, 'rb'))
        else:
            count_s, count_videos = 0, 0
            pbar = ProgressBar().start() # Change we don't really need it
            for dir_s in [paths for paths in os.listdir(opt.datapath) if paths.startwith("s")]:
                count_s += 1

                for dir_v in os.listdir(os.path.join(opt.datapath, dir_s)):
                    current_path = os.path.join(opt.datapath, dir_s, dir_v)

                    sub_file = os.path.join(opt.alignpath, dir_s, '{}.align'.format(dir_v))
                    flag_add = os.path.exists(sub_file)

                    if not os.path.exists(current_path) or len(os.listdir(current_path)) != 75:
                        flag_add = False
                    else:

                        size = io.imread(os.path.join(current_path, 'mouth_000.png')).shape
                        if size[0] != 50 or size[1] != 100:
                            flag_add = False
                    
                    
                    if flag_add:
                        d = {"s": dir_s, "v": dir_v, "words": [], "time_start": [], "time_end": []}
                        for line in open(sub_file, "r").read().splitlines():
                            token = line.split()

                            if token[2] != 'sil' and token[2] != 'sp':
                                d['words'].append(token[2])
                                d['time_start'].append(int(token[0]))
                                d['time_end'].append(int(token[1]))

                                for char in token[2]:
                                    unsorted_vocab[char] = True
                                
                                if (not opt.test_overlapped and (dir_s in ["s1", "s2", "s20", "s22"])) or (opt.test_overlapped and dir_v in self.overlap_list[dir_s].keys()):
                                    if self.dataset_type == "test":
                                        count_videos += 1
                                        d["mode"], d["flip"], d["test"] = 7, False, True
                                        self.dataset.append(d)

                                else:
                                    if self.dataset_type == "train":
                                        count_videos += 1
                                        d["test"] = False
                                        for flip in (False, True):
                                            if opt.use_words:
                                                for w_start in range(1,7):
                                                    d_i = d.copy()
                                                    d_i["flip"], d_i["mode"],d_i["w_start"] = flip, 1, w_start

                                                    d_i["w_end"] = w_start + d_i["mode"] - 1
                                                    frame_v_start = max(round(1 / 1000 * d["time_start"][d_i["w_start"] - 1]),1)
                                                    frame_v_end = min(round(1 / 1000 * d["time_end"][d_i["w_end"] - 1]),75)
                                                    if frame_v_end - frame_v_start + 1 >= 3:
                                                        self.dataset.append(d_i)


                                            d_i = d.copy()
                                            d_i["mode"], d_i["flip"] = 7, flip
                                            self.dataset.append(d_i)

                pbar.update(int(count_s / 33 * 100))
            pbar.finish()

            self.vocab = []
            for char in unsorted_vocab:
                self.vocab.append(char)

            self.vocab.sort()
            self.vocab_mapping = []

            for i, char in enumerate(self.vocab):
                self.vocab_mapping[char] = i + 1

            pickle.dump((self.dataset, count_videos, self.vocab, self.vocab_mapping), open(cache_path, 'wb'))

        print(f"{self.dataset_type}: n_videos = {count_videos}, n_samples = {len(self.dataset)}, vocab_size = {len(self.vocab)}")

        print("vocab: {}".format('|'.join(self.vocab)))
                                                                        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x = torch.zeros(3, self.opt.max_timesteps, 50, 100)

        d = self.dataset[idx]

        frames, y, sub = read_data(d, self.opt, self.vocab_mapping)
        x[:, :frames.size(1), :, :] = frames

        length = frames.size(1)

        return x, y, length, idx