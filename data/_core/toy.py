from torch.utils.data import Dataset
import torch

"""
this contains functions needed for the toy dataset
"""


class ToyDataset(Dataset):
    def __len__(self):
        return 10000
    
    def __getitem__(self, i):
        return (torch.ones(10), torch.ones(2))