from typing import Optional
from torch.utils.data import Dataset, Subset, IterableDataset
import torch


class RandomSubset(Subset):
    def __init__(self, dataset: Dataset, num: int, seed=42):
        torch.manual_seed(seed)
        Subset.__init__(self, dataset, torch.randperm(len(dataset))[:num])


class LimitIterableDataset(IterableDataset):
    def __init__(self, dataset: Dataset, num: Optional[int] = None):
        self.dataset, self.num = dataset, num

    def __iter__(self):
        for (i, batch)in enumerate(self.dataset):
            if self.num is not None and i >= self.num:
                return
            yield batch