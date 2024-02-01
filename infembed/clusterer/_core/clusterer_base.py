from abc import ABC, abstractmethod
from typing import Any
import pandas as pd
from infembed.utils.common import Data


class ClustererBase(ABC):
    """
    An abstract class to define the interface for clustering.
    TODO: figure out which sklearn mixin this should be an implementation of
    """
    @abstractmethod
    def fit(self, data: Data):
        r"""
        Does the prepratory computation needed to assign new examples to clusters.  For
        example with K-Means, this might determine the locations of the cluster
        centers.

        Args:
            data (Data): `Data` representing the examples used for doing the
                    prepratory computation.
        """
        pass

    @abstractmethod
    def fit_predict(self, data: Data):
        r"""
        Assigns the examples in `embeddings` to clusters, after doing the prepratory
        computation.

        Args:
            data (Data): `Data` representing the examples used for doing the
                    prepratory computation.
        """
        pass

    @abstractmethod
    def predict(self, data: Data):
        r"""
        Assigns the examples in `embeddings` to clusters.

        Args:
            data (Data): `Data` representing the examples used for doing the
                    prepratory computation.
        """
        pass