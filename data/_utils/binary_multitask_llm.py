from typing import List
from torch.utils.data import Dataset, IterableDataset
from torch import Tensor
import torch
from torch.utils.data import default_collate


"""
data relevant for binary multitask scenario, which includes CB scenario
"""


class MultitaskDataset(Dataset):
    """
    returns vector (corresponding to task labels) for each example
    """

    pass


class MultiTaskDatasetCollateFn(Dataset):
    """
    just applies default collate function, then put into dictionary with one item whose
    key is 'example_labels'
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
    this collate function is generic to the token multitask case, but below describes
    its application to concept loss.

    takes in list of examples from a `FnsTokenConceptDataset` giving
    `concept_labels` and `attention_mask`.  this is used by `ConceptLoss`, and also
    to evaluate token-level concept predictions, and train a token-level concept
    classifier.
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
