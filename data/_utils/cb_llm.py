from typing import Callable, List
from torch.utils.data import Dataset
from torch import Tensor
import torch


class ConceptDataset(Dataset):
    """
    returns vector of concept labels for each text
    """

    pass


class FnsConceptDataset(ConceptDataset):
    """
    applies sequence of functions to example to give concepts
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


class TokenConceptDataset(Dataset):
    """
    returns a vector of concept labels for each position in text
    in general, label at position t corresponds to text occurying *strictly* before t
    """

    pass


class FnsTokenConceptDataset(TokenConceptDataset):
    def __init__(self, dataset: Dataset, tokenizer, fns: List[Callable]):
        self.fns, self.tokenizer = fns, tokenizer
        self.dataset = dataset

    def _get_concepts(self, example):
        # first convert to tokens
        input_ids = self.tokenizer(example)["input_ids"]
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


class TokenConceptDatasetCollateFn:
    """
    takes in list of examples from a `FnsTokenConceptDataset` giving
    `concept_labels` and `attention_mask`.  this is used by `ConceptLoss`, and also
    to evaluate token-level concept predictions, and train a token-level concept
    classifier.
    """

    def __init__(self, max_len):
        self.max_len = max_len

    def __call__(self, examples: List[Tensor]):
        num_concepts = next(iter(examples)).shape[1]
        max_len = max(map(len, examples))
        batch_size = len(examples)
        concept_labels = torch.zeros(batch_size, max_len, num_concepts)
        attention_mask = torch.zeros(batch_size, max_len)  # , num_concepts)
        for _example, _concept_labels, _attention_mask in zip(
            examples, concept_labels, attention_mask
        ):
            _concept_labels[: len(_example)] = _example
            _attention_mask[: len(_example)] = 1
        return {
            "concept_labels": concept_labels[:, : self.max_len],
            "attention_mask": attention_mask[:, : self.max_len],
        }
