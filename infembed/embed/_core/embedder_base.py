from abc import ABC, abstractmethod
from typing import Any, Optional, List
from torch.utils.data import DataLoader
from torch.nn import Module


class EmbedderBase(ABC):
    """
    An abstract class to define the interface for embedding computation.
    TODO: figure out which sklearn mixin this should be an implementation of
    """
    def __init__(self, **kwargs: Any):
        pass

    @abstractmethod
    def fit(self, dataloader: DataLoader):
        r"""
        Does the computation to tailor the embeddings to a specific model and data.
        Some implementations of `EmbedderBase` may not actually do any tailoring,
        i.e. use embeddings from external model, i.e. CLIP.

        Args:
            dataloader (DataLoader): The dataloader containing data needed to learn how
                    to compute the embeddings.  Some implementations of `EmbedderBase`
                    may not actually use this argument.
        """

    @abstractmethod
    def predict(self, dataloader: DataLoader):
        r"""
        Computes the embeddings for a dataloader.

        Args:
            dataloader (`DataLoader`): dataloader whose examples to compute influence
                    embeddings for.
        """

    def get_name(self):
        return type(self).__name__