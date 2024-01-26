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
            dataloader (DataLoader): dataloader whose examples to compute embeddings
                    for.
        """

    def get_name(self):
        return type(self).__name__
    
    def save(self, path: str):
        """
        For some implementations, the `fit` method does computation whose results we
        may want to save.  This method saves those results to a file.

        Args:
            path (str): path of file to save results in.
        """
        pass

    def load(self, path: str):
        """
        Loads the results saved by the `save` method.  Instead of calling `fit`, one
        can instead call `load`.

        Args:
            path (str): path of file to load results from.
        """
        pass