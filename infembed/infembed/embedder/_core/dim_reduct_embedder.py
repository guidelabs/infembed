import logging
from typing import Dict, Optional
from infembed.embedder._core.embedder_base import EmbedderBase
from infembed.embedder._utils.common import (
    NotFitException,
    _format_inputs_dataset,
    _progress_bar_constructor,
)
from torch.utils.data import DataLoader
from sklearn.decomposition import IncrementalPCA
import torch
from torch import Tensor
import dill as pickle


class PCAEmbedder(EmbedderBase):
    """
    This embedder takes in a base embedder, and reduces the dimension of its embeddings
    using Sklearn's IncrementalPCA implementation.  The incremental aspect allows to do
    PCA on a large number of embeddings which may not otherwise fit in memory.
    """

    def __init__(
        self,
        base_embedder: EmbedderBase,
        incremental_pca_kwargs: Dict,
        show_progress: bool = True,
    ):
        r"""
        Args:
            base_embedder (EmbedderBase): the embedder whose embeddings will have its
                    dimension reduced.
            incremental_pca_kwargs (dict): kwargs to pass to the `IncrementalPCA`
                    constructor.  See https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.IncrementalPCA.html
                    for details.  For example, `n_components` specifies the dimension
                    of the projection.
        """
        self.base_embedder, self.incremental_pca_kwargs = (
            base_embedder,
            incremental_pca_kwargs,
        )

        self.show_progress = show_progress

        self.incremental_pca: Optional[IncrementalPCA] = None

    def fit(
        self,
        dataloader: DataLoader,
    ):
        r"""
        Does the computation needed for computing embeddings, which is
        finding the top eigenvectors / eigenvalues of the Hessian, computed
        using `dataloader`.  Note that the batch size of `dataloader` needs to be
        greater than the `n_components` specified in the `incremental_pca_kwargs`
        constructor argument.

        Args:
            dataloader (DataLoader): The dataloader containing data needed to learn how
                    to compute the embeddings
        """
        self.incremental_pca = IncrementalPCA(**self.incremental_pca_kwargs)
        self.base_embedder.fit(dataloader)

        # compute embeddings for one batch of dataloader at once, and feed to
        # `IncrementalPCA`

        # set `show_progress` for `base_embedder` to False to avoid too much logging
        # TODO: achieve this by changing debug level of a embedder-specific logger
        self.base_embedder.show_progress = False

        if self.show_progress:
            dataloader = _progress_bar_constructor(
                self, dataloader, "pca", "training data"
            )

        for batch in dataloader:
            batch_embeddings = self.base_embedder.predict(_format_inputs_dataset(batch))
            # import pdb
            # pdb.set_trace()
            if self.incremental_pca.n_components <= len(batch_embeddings):
                # because can only call `partial_fit` if batch size is large enough
                self.incremental_pca.partial_fit(batch_embeddings)

        return self

    def predict(self, dataloader: DataLoader) -> Tensor:
        """
        Returns the embeddings for `dataloader`.

        Args:
            dataloader (`DataLoader`): dataloader whose examples to compute embeddings
                    for.
        """
        if self.incremental_pca is None:
            raise NotFitException(
                "The results needed to compute embeddings not available.  Please either call the `fit` or `load` methods."
            )
        
        if self.show_progress:
            dataloader = _progress_bar_constructor(
                self, dataloader, "embeddings", "test data"
            )

        # apply pca to a batch's embeddings at a time, to save memory
        # define a helper function that returns the embeddings for a batch
        def get_batch_embeddings(batch):
            self.base_embedder.show_progress = False
            batch_base_embeddings = self.base_embedder.predict(
                _format_inputs_dataset(batch)
            )
            return torch.from_numpy(self.incremental_pca.transform(batch_base_embeddings))

        logging.info("compute embeddings")
        return torch.cat([get_batch_embeddings(batch) for batch in dataloader], dim=0)

    def save(self, path: str):
        """
        This method saves the results of `fit` to a file.

        Args:
            path (str): path of file to save results in.
        """
        with open(path, "wb") as f:
            pickle.dump(self.incremental_pca, f)

    def load(self, path: str):
        """
        Loads the results saved by the `save` method.  Instead of calling `fit`, one
        can instead call `load`.

        Args:
            path (str): path of file to load results from.
        """
        with open(path, "rb") as f:
            self.incremental_pca = pickle.load(f)

    def reset(self):
        """
        Removes the effect of calling `fit` or `load`
        """
        self.incremental_pca = None
