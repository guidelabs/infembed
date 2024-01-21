from typing import List
from infembed.clusterer._core.clusterer_base import ClustererBase
import torch
import numpy as np
from collections import defaultdict
from infembed.utils.common import Data


def _cluster_assignments_to_indices(assignments: np.ndarray) -> List[List[int]]:
    d = defaultdict(list)
    for (i, k) in enumerate(assignments):
        d[int(k)].append(i)
    return [d[k] for k in sorted(d.keys())]


class SklearnClusterer(ClustererBase):
    """
    Implementation of `ClustererBase` that is a wrapper around an Sklearn clustering
    implementation, i.e. it just converts the input tensor to numpy array.
    """

    def __init__(self, sklearn_clusterer):
        self.sklearn_clusterer = sklearn_clusterer

    def fit(self, data: Data):
        r"""
        Does the prepratory computation needed to assign new examples to clusters.  For
        example with K-Means, this might determine the locations of the cluster
        centers.

        Args:
            data (Data): `Data` representing the examples used for doing the
                    prepratory computation.
        """
        if not hasattr(self.sklearn_clusterer, "fit"):
            raise NotImplementedError
        np.random.seed(42)
        self.sklearn_clusterer.fit(data.embeddings.detach().cpu())
        return self

    def fit_predict(self, data: Data) -> List[List[int]]:
        r"""
        Assigns the examples represented by `embeddings` to clusters, after doing the
        prepratory computation.

        Args:
            data (Data): `Data` representing the examples to assign to clusters.
        """
        if not hasattr(self.sklearn_clusterer, "fit_predict"):
            raise NotImplementedError
        np.random.seed(42)
        return _cluster_assignments_to_indices(
            torch.from_numpy(
                self.sklearn_clusterer.fit_predict(data.embeddings.detach().cpu())
            )
        )

    def predict(self, data: Data) -> List[List[int]]:
        r"""
        Assigns the examples in `embeddings` to clusters.

        Args:
            data (Data): `Data` representing the examples to assign to clusters.
        """
        if not hasattr(self.sklearn_clusterer, "predict"):
            raise NotImplementedError
        return _cluster_assignments_to_indices(
            self.sklearn_clusterer.predict(data.embeddings.detach().cpu())
        )
