from typing import List
from torch.utils.data import Dataset
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
            'example_labels': default_collate(examples)
        }


class TokenMultitaskDataset(Dataset):
    """
    returns vector (corresponding to task labels) for each token in each example
    """
    pass


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
        for _example, _labels, _attention_mask in zip(
            examples, labels, attention_mask
        ):
            _labels[: len(_example)] = _example
            _attention_mask[: len(_example)] = 1
        return {
            "labels": labels[:, : self.max_len],
            "attention_mask": attention_mask[:, : self.max_len],
        }