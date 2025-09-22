import dlib 
from torch.utils.data import Dataset, DataLoader
import json
import os
import pickle

class MyDataset(Dataset):
    def __init__(self, opt, dataset_type = "train"):
        self.opt = opt
        self.dataset_type = dataset_type
        self.overlap_list = json.load(open(self.opt.overlap_list, "r").read())
        self.dataset = []

    def load_data(self):
        opt = sef.opt
        unsorted_vocab = {}
        unsorted_vocab[' '] = True
        
        print(f"Loading data from {self.dataset_type} dataset")
        cache_path = '{}_{}.pkl'.format(self.dataset_type, "overlapped" if opt.test_overlapped else "unseen")
        if os.path.exists(cache_path):
            self.dataset, count_v, self.vocab, self.vocab_mapping = pickle.load(open(cache_path, 'rb'))

        with open(in_dir, 'r') as 

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample, label