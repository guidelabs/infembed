from typing import Callable, List
from data._utils.binary_multitask_llm import MultitaskDataset, TokenMultitaskDataset
from torch.utils.data import Dataset
from torch import Tensor
import torch


class FnsConceptDataset(MultitaskDataset):
    """
    applies sequence of functions to example to give concepts.  each example in dataset
    is raw text
    """

    def __init__(self, dataset: Dataset, fns: List[Callable]):
        self.fns = fns
        self.dataset = dataset

    def _get_concepts(self, example):
        return Tensor([fn(example) for fn in self.fns])

    def __getitem__(self, i: int):
        return self._get_concepts(self.dataset[i])

    def __iter__(self):
        for x in self.dataset:
            yield self._get_concepts(x)


class FnsTokenConceptDataset(TokenMultitaskDataset):
    """
    returns a vector of concept labels for each position in tokenized text
    in general, label at position t corresponds to text occurying *strictly* before t
    """
    def __init__(self, dataset: Dataset, tokenizer, fns: List[Callable]):
        self.fns, self.tokenizer = fns, tokenizer
        self.dataset = dataset

    def _get_concepts(self, example):
        # first convert to tokens
        input_ids = self.tokenizer(example)["input_ids"]
        if len(input_ids) == 0:
            return torch.zeros(0, len(self.fns)).bool()
        token_strs = [self.tokenizer.decode(input_id) for input_id in input_ids]
        return Tensor(
            [
                [fn("".join(token_strs[:t])) for fn in self.fns]
                for t in range(len(token_strs))
            ]
        )

    def __getitem__(self, i: int):
        return self._get_concepts(self.dataset[i])

    def __iter__(self):
        for x in self.dataset:
            yield self._get_concepts(x)
