import dlib 
from torch.utils.data import Dataset, DataLoader
import json
import os
import pickle
from progressbar import *
from skimage import io
import torch
import multiprocessing
import thre       
class MyDataset(Dataset):
    def __init__(self, **kwargs):
        self.opt = {
            "NUM_WORKERS":multiprocessing.cpu_count()/2, #Halve the cpu core
            "DATA_PATH":"./data/vid",
            "ALIGN_PATH":"./data/align",
            "BATCH_SIZE":20,
            "TEST_OVERLAPPED":False,
            "DATA_AUGMENTATION_TEMPORAL":False,
            "NORMALISE": True,
            "FRAME_RATE": 25, #FPS
            "FRAME_SKIP": 1,
            "MIN_TIMESTEPS": 2, #Min frames for filtering bad data
            "MAX_TIMESTEPS": 75, #Max number of frame per sub for preallocation
            "MODE_IMG":"mouth",
            "DEBUG": False,
            "DATASET_TYPE": "train",
            "USE_WORDS": True,
        }
        self.opt.update(kwargs)
        
        self.dataset_type = self.opt["DATASET_TYPE"]
        self.overlap_list = json.load(open(self.opt.overlap_list, "r").read())
        self.dataset = []
        self.dataset_val = []
        self.state = {}

    def load_data(self):
        opt = self.opt
        unsorted_vocab = {}
        unsorted_vocab[' '] = True
        
        #Iterate among speakers
        print(f"Loading data from {self.dataset_type} dataset")
        cache_path = '{}_{}.pkl'.format(self.dataset_type, "overlapped" if opt["TEST_OVERLAPPED"] else "unseen")
        if os.path.exists(cache_path):
            self.dataset, count_videos, self.vocab, self.vocab_mapping = pickle.load(open(cache_path, 'rb'))
        else:
            pbar = ProgressBar().start() # SHOWING PROGRESS, comment if not needed
            
            count_speakers, count_videos = 0, 0
            
            for speaker_dir in [paths for paths in os.listdir(opt["DATA_PATH"]) if paths.startswith("s")]:
                count_speakers += 1
                
                #get speaker videos
                for video_dir in os.listdir(os.path.join(opt["DATA_PATH"], speaker_dir)):
                    current_path = os.path.join(opt["DATA_PATH"], speaker_dir, video_dir) #create path for each video inside each speaker directory

                    #check if the sub of the current video has been made / check for align file
                    sub_file = os.path.join(opt["ALIGN_PATH"], speaker_dir, '{}.align'.format(video_dir))
                    need_to_add = os.path.exists(sub_file)

                    #check if the current video's frames exist and are the correct shape
                    if not os.path.exists(current_path) or len(os.listdir(current_path)) != 75:
                        need_to_add = False
                    else:

                        #read each frame/image size
                        size = io.imread(os.path.join(current_path, 'mouth_000.png')).shape
                        if size[0] != 50 or size[1] != 100:
                            need_to_add = False
                    
                    if need_to_add:
                        d = {"speaker": speaker_dir, "video": video_dir, "words": [], "time_start": [], "time_end": []}
                        
                        for line in open(sub_file, "r").read().splitlines():
                            line += 1
                            token = line.split()
                            
                            #remove silence and space in the sub
                            if token[2] != 'sil' and token[2] != 'sp':
                                d['words'].append(token[2])
                                d['time_start'].append(int(token[0]))
                                d['time_end'].append(int(token[1]))

                                #build vocab
                                for char in token[2]:
                                    unsorted_vocab[char] = True
                                
                        #append to subs data
                        if (not opt.test_overlapped and (speaker_dir in ["s1", "s2", "s20", "s22"])) or (opt.test_overlapped and video_dir in self.overlap_list[speaker_dir].keys()):
                            if self.dataset_type == "test":
                                count_videos += 1
                                d["mode"], d["flip"], d["test"] = 7, False, True
                                self.dataset.append(d)

                        else:
                            if self.dataset_type == "train":
                                count_videos += 1
                                d["test"] = False
                                for flip in (False, True):
                                    
                                    #add word instances
                                    if opt["USE_WORDS"]:
                                        for w_start in range(1,7):
                                            d_i = d.copy()
                                            d_i["flip"], d_i["mode"],d_i["w_start"] = flip, 1, w_start

                                            #all instances used were either whole sentences or individual words
                                            d_i["w_end"] = w_start + d_i["mode"] - 1
                                            # frame_v_start = max(round(1 / 1000 * d["time_start"][d_i["w_start"] - 1]),1)
                                            frame_v_start = max(round(75 / 3000 * d["time_start"][d_i["w_start"] - 1]),1)
                                            # frame_v_end = min(round(1 / 1000 * d["time_end"][d_i["w_end"] - 1]),75)
                                            frame_v_end = min(round(75 / 3000 * d["time_end"][d_i["w_end"] - 1]),75)
                                            if frame_v_end - frame_v_start + 1 >= 3:
                                                self.dataset.append(d_i)


                                    d_i = d.copy()
                                    d_i["mode"], d_i["flip"] = 7, flip
                                    self.dataset.append(d_i)

                pbar.update(int(count_speakers / 33 * 100))
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