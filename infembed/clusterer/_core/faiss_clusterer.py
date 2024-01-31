from typing import List
from infembed.clusterer._core.clusterer_base import ClustererBase
from infembed.clusterer._utils.common import _cluster_assignments_to_indices
import torch
import numpy as np
from collections import defaultdict
from infembed.utils.common import Data
import faiss


class FAISSClusterer(ClustererBase):
    """
    Implementation of `ClustererBase` that is a wrapper around FAISS's Kmeans
    implementation.
    """

    def __init__(self, **faiss_kmeans_kwargs):
        """
        Args:
            faiss_kmeans_kwargs: keyword arguments to be directly passed to the
                    `faiss.Kmeans` constructor.  Relevant arguments include
                    `k` - the number of clusters, `spherical` - whether to
                    normalize the centroids after each iteration, `niter` - number of
                    k-means iterations.  See https://github.com/facebookresearch/faiss/wiki/Faiss-building-blocks:-clustering,-PCA,-quantization
                    for details.
        """
        self.faiss_kmeans_kwargs = faiss_kmeans_kwargs
        self.kmeans = None

    def fit(self, data: Data):
        r"""
        Does the prepratory computation needed to assign new examples to clusters.  For
        example with K-Means, this might determine the locations of the cluster
        centers.

        Args:
            data (Data): `Data` representing the examples used for doing the
                    prepratory computation.
        """
        assert isinstance(data.embeddings, torch.Tensor)
        d = data.embeddings.shape[1]
        self.kmeans = faiss.Kmeans(d=d, seed=42, **self.faiss_kmeans_kwargs)
        self.kmeans.train(data.embeddings)
        return self

    def fit_predict(self, data: Data) -> List[List[int]]:
        r"""
        Assigns the examples represented by `embeddings` to clusters, after doing the
        prepratory computation.

        Args:
            data (Data): `Data` representing the examples to assign to clusters.
        """
        self.fit(data)
        return self.predict(data)

    def predict(self, data: Data) -> List[List[int]]:
        r"""
        Assigns the examples in `embeddings` to clusters.

        Args:
            data (Data): `Data` representing the examples to assign to clusters.
        """
        assert isinstance(data.embeddings, torch.Tensor)
        _, I = self.kmeans.index.search(data.embeddings, 1)
        return _cluster_assignments_to_indices(I.squeeze())