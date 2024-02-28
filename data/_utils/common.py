from typing import Callable, Dict, List, Optional
from torch.utils.data import Dataset, Subset, IterableDataset
import torch
import lightning as L


"""
helpers for data that don't fit anywhere else
"""


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
                print("breaking", i)
                return
            yield batch

    def __len__(self):
        return self.num


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


def default_batch_to_x(batch):
    return batch[:-1]


def default_batch_to_target(batch):
    return batch[-1]


def keep_pre(x):
    return x[:-1]


class ComposeCollateFn:
    def __init__(self, f, g):
        self.f, self.g = f, g

    def __call__(self, x):
        return self.f(self.g(x))


class ZipDataset(Dataset):
    """
    takes in sequence of datasets, and returns their zip.
    can act as `Dataset` if all of `dataset` is a `Dataset`
    """

    def __init__(self, datasets: List[Dataset]):
        self.datasets = datasets

    def __getitem__(self, i: int):
        return tuple(dataset[i] for dataset in self.datasets)

    def __len__(self):
        return len(next(iter(self.datasets)))


class ZipIterableDataset(IterableDataset):
    """
    takes in sequence of datasets, and returns their zip.
    can act as `Dataset` if all of `dataset` is a `Dataset`
    """

    def __init__(self, datasets: List[Dataset]):
        self.datasets = datasets

    def __iter__(self):
        return zip(*self.datasets)
    
    def __len__(self):
        min = None
        for dataset in self.datasets:
            try:
                _len = len(dataset)
            except TypeError:
                pass
            else:
                if min is None or _len < min:
                    min = _len
        return min
    

class MixDataset(Dataset):
    def __init__(self, dataset1: Dataset, dataset2: Dataset):
        # make total index, shuffle, and convert from total index to original
        torch.manual_seed(42)
        self.total_index = list(map(int, torch.randperm(len(dataset1) + len(dataset2))))
        self.dataset1, self.dataset2 = dataset1, dataset2
        print((torch.Tensor(self.total_index) < len(self.dataset1)).sum(), len(self.dataset1))
        print((torch.Tensor(self.total_index) >= len(self.dataset1)).sum(), len(self.dataset2))
    
    def __getitem__(self, i):
        _i = self.total_index[i]
        if _i < len(self.dataset1):
            return self.dataset1[_i]
        else:
            return self.dataset2[_i - len(self.dataset1)]
        
    def __len__(self):
        return len(self.total_index)
    

class MixIterableDataset(IterableDataset, MixDataset):
    def __iter__(self):
        dataset1_iter = iter(self.dataset1)
        dataset2_iter = iter(self.dataset2)
        for i in self.total_index:
            if i < len(self.dataset1):
                yield next(dataset1_iter)
            else:
                yield next(dataset2_iter)


class ReplicateDataset(Dataset):
    def __init__(self, dataset: Dataset, replicas: int):
        self.dataset, self.replicas = dataset, replicas

    def __getitem__(self, i):
        return self.dataset[i % len(self.dataset)]
    
    def __len__(self):
        return self.replicas * len(self.dataset)
    

class ReplicateIterableDataset(IterableDataset):
    def __init__(self, dataset: Dataset, replicas: int):
        self.dataset, self.replicas = dataset, replicas

    def __iter__(self):
        for _ in range(self.replicas):
            for x in self.dataset:
                yield x


class ZerosIterableDataset(IterableDataset):
    def __init__(self, dim: int):
        self.dim = dim

    def __iter__(self):
        while True:
            yield torch.zeros(self.dim)


def dict_batch_combiner(dicts):
    """
    returns dictionary with union of the items in dicts
    """
    d = {}
    for _d in dicts:
        d = {**d, **_d}
    return d


class ZipCollateFn:
    """
    applies sequence of collate functions to the examples from a `ZipDataset`, and
    applies a specified function to combine the different batches
    """

    def __init__(self, collate_fns, combiner: Callable):
        self.collate_fns = collate_fns
        self.combiner = combiner

    def __call__(self, examples):
        batches = [
            collate_fn(_examples)
            for (_examples, collate_fn) in zip(zip(*examples), self.collate_fns)
        ]
        return self.combiner(batches)


class DatasetFromText(IterableDataset):
    """
    takes in raw text, splits into chunks constituting different elements of a dataset
    """

    def __init__(self, path, text_size):
        self.path, self.text_size = path, text_size

    def __iter__(self):
        t = ""
        num_yield = 0
        num_line = 0
        for line in open(self.path, "r"):
            t += line
            num_line += 1
            if len(t) > self.text_size:
                yield t[: self.text_size]
                t = t[self.text_size :]
                num_yield += 1
        while len(t) > 0:
            yield t[: self.text_size]
            t = t[self.text_size :]
            num_yield += 1
        print(num_yield, num_line)


class EmptyTextDataset(IterableDataset):
    """
    yields empty text for use in prediction step, to generate text from nothing
    """

    def __init__(self, num_examples):
        self.num_examples = num_examples

    def __iter__(self):
        for _ in range(self.num_examples):
            yield ""


class IterableDatasetToDataset(Dataset):
    def __init__(self, dataset):
        self._dataset = [batch for batch in dataset]

    def __getitem__(self, i):
        return self._dataset[i]


def character_tokenizer():
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
    return tokenizer.train_new_from_iterator([], vocab_size=0, initial_alphabet=[])


IGNORE_INDEX = -100


class RenameCollateFn:
    def __init__(self, collate_fn, rename_map: Dict):
        self.collate_fn, self.rename_map = collate_fn, rename_map

    def __call__(self, examples):
        batch = self.collate_fn(examples)
        return {
            (self.rename_map[key] if key in self.rename_map else key): val
            for (key, val) in batch.items()
        }
