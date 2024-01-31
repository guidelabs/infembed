from typing import Optional
from torch.utils.data import Dataset, Subset, IterableDataset
import torch
import lightning as L


class RandomSubset(Subset):
    def __init__(self, dataset: Dataset, num: int, seed=42):
        torch.manual_seed(seed)
        Subset.__init__(self, dataset, torch.randperm(len(dataset))[:num])


class LimitIterableDataset(IterableDataset):
    def __init__(self, dataset: Dataset, num: Optional[int] = None):
        self.dataset, self.num = dataset, num

    def __iter__(self):
        for i, batch in enumerate(self.dataset):
            if self.num is not None and i >= self.num:
                return
            yield batch


class GenericDataModule(L.LightningDataModule):
    def __init__(
        self,
        train_dataloader=None,
        val_dataloader=None,
        test_dataloader=None,
        predict_dataloader=None,
    ):
        super().__init__()
        (
            self._train_dataloader,
            self._val_dataloader,
            self._test_dataloader,
            self._predict_dataloader,
        ) = (
            train_dataloader,
            val_dataloader,
            test_dataloader,
            predict_dataloader,
        )

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        return self._train_dataloader

    def val_dataloader(self):
        return self._val_dataloader

    def test_dataloader(self):
        return self._test_dataloader

    def predict_dataloader(self):
        return self._predict_dataloader