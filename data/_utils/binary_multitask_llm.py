from typing import List
from torch.utils.data import Dataset, IterableDataset
from torch import Tensor
import torch
from torch.utils.data import default_collate


"""
helpers for data relevant for binary multitask scenario
"""


class MultitaskDataset(Dataset):
    """
    returns vector (corresponding to task labels) for each example
    """

    pass


class MultiTaskDatasetCollateFn(Dataset):
    """
    returns batches with per-example labels in the key 'example_labels'
    """

    def __call__(self, examples: List[Tensor]):
        return {
            "example_labels": default_collate(
                [example.long() for example in examples]
            ).long()
        }


class TokenMultitaskDataset(Dataset):
    """
    returns vector (corresponding to task labels) for each token in each example
    """

    pass


class ReadTokenMultitaskDataset(IterableDataset, TokenMultitaskDataset):
    """
    reads the output of `LLMBinaryMultitaskWritePredictions` to given a dataset
    """
    def __init__(self, path):
        self.path = path

        num = 0
        for _ in self:
            num += 1
        self.len = num

    def __len__(self):
        return self.len

    def __iter__(self):
        f = open(self.path, "r")
        print(next(f))
        prev_i = -1
        labels = None
        for line in f:
            entries = line.strip().split(",")
            i = int(entries[0])
            t = int(entries[1])
            _labels = Tensor(list(map(float, entries[2:])))
            num_tasks = len(_labels)
            if i != prev_i:
                # start of new example
                # yield empty labels if any
                for _ in range(prev_i + 1, i):
                    yield torch.zeros(0, num_tasks)
                if labels is not None:
                    # yield if there is something to yield
                    yield torch.sigmoid(torch.stack(labels, dim=0))
                    # reset prev_i and labels
                prev_i = i
                labels = []

            labels.append(_labels)

        yield torch.sigmoid(torch.stack(labels, dim=0))


class TokenMultitaskDatasetCollateFn:
    """
    returns batch with the key 'labels' (per-token binary multi-task labels) and
    corresponding 'attention_mask'.

    if used by cb-llm, need to rename 'labels' to 'concept_labels'
    """

    def __init__(self, max_len):
        self.max_len = max_len

    def __call__(self, examples: List[Tensor]):
        num_tasks = next(iter(examples)).shape[1]
        max_len = max(map(len, examples))
        batch_size = len(examples)
        labels = torch.zeros(batch_size, max_len, num_tasks)
        attention_mask = torch.zeros(batch_size, max_len)
        for _example, _labels, _attention_mask in zip(examples, labels, attention_mask):
            _labels[: len(_example)] = _example
            _attention_mask[: len(_example)] = 1
        return {
            "labels": labels[:, : self.max_len],#.long(),
            "attention_mask": attention_mask[:, : self.max_len],
        }
