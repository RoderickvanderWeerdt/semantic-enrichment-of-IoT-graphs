#based on: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
from __future__ import print_function, division
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        embedding, target = sample['embedding'], sample['emb_targets']
        return {'embedding': torch.from_numpy(embedding),
                'target': torch.from_numpy(target)}





class Emb_Target_Dataset(Dataset):
    """Dataset containing the embedding and the class 'hot' or 'cold' based on the temperature at that time."""

    # def __init__(self, csv_file, train=True, transform=None, train_test_split=0.8, emb_id=0, target_column='https://interconnectproject.eu/example/Konstanz_TempC', embedding_column='emb19'):
    def __init__(self, csv_file, train=True, transform=None, train_test_split=0.8, emb_id=0, target_column='target_values_211', embedding_column='emb'):
        csv_file = pd.read_csv(csv_file, sep=",")
        if train:
            
            self.emb_targets = csv_file[:int(len(csv_file)*train_test_split)].reset_index()
        else:
            self.emb_targets = csv_file[int(len(csv_file)*train_test_split):].reset_index()

        self.transform = transform
        self.emb_id = emb_id
        self.target_column = target_column
        self.emb_column = embedding_column
        self.normalize_targets()


    def __len__(self):
        return len(self.emb_targets)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        target = self.emb_targets[self.target_column][idx]
        target = np.array(target)
        
        # embedding_column = 'emb'
        if self.emb_id > 0:
            embedding_column = self.emb_column + str(self.emb_id)
        else:
            embedding_column = self.emb_column
        sample = {'embedding': np.array([float(x.strip(' []')) for x in self.emb_targets[embedding_column][idx].split(',')]), 
                  'emb_targets': target 
                 }
        if self.transform:
            sample = self.transform(sample)

        return sample

    def normalize_targets(self):
        target_list = self.emb_targets[self.target_column]
        # self.target_min = min(target_list) 
        self.target_min = min(target_list) - 0.1 #added -0.1 so y will never be 0 (which causes divide by zero issues)
        if self.target_min < 0:
            self.target_min = self.target_min * -1
        self.target_max = max(target_list)
        target_list = (target_list + self.target_min) / (self.target_max+self.target_min)
        self.emb_targets[self.target_column] = target_list

    def denormalize_single(self, value, return_round=False):
        denormalized = (value * (self.target_max+self.target_min)) - self.target_min
        if return_round:
            return round(denormalized, 1)
        else:
            return denormalized

    def normalize_single(self, value):
        normalized = (value + self.target_min) / (self.target_max+self.target_min)
        return normalized