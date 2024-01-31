from typing import Callable, List, Optional, Union
from infembed.embedder._core.dim_reduct_embedder import PCAEmbedder
from infembed.embedder._core.embedder_base import EmbedderBase
from infembed.embedder._utils.common import (
    _check_loss_fn,
    _compute_jacobian_sample_wise_grads_per_batch,
    _flatten_sample_wise_grads,
    _progress_bar_constructor,
    _set_active_parameters,
)
from torch.nn import Module
from torch.utils.data import DataLoader
from torch import Tensor
import torch
import logging


class GradientEmbedder(EmbedderBase):
    r"""
    Computes per-example loss gradients as the embeddings.
    """

    def __init__(
        self,
        model: Module,
        layers: Optional[List[str]] = None,
        loss_fn: Optional[Union[Module, Callable]] = None,
        sample_wise_grads_per_batch: bool = False,
        show_progress: bool = False,
    ):
        """
        Args:
            model (Module): The model used to compute the embeddings.
            layers (list of str, optional): names of modules in which to consider
                    gradients.  If `None` or not provided, all modules will be used.
                    Default: None
            loss_fn (Module or Callable, optional): The loss function used to compute the
                    Hessian.  It should behave like a "reduction" loss function, where
                    reduction is either 'sum', 'mean', or 'none', and have a
                    `reduction` attribute.  For example, `BCELoss(reduction='sum')`
                    could be a valid loss function.  See the caveat under the
                    description for the `sample_wise_grads_per_batch` argument.  If None,
                    the loss is the output of `model`, which is assumed to be a single
                    scalar for a batch.
                    Default: None
            sample_wise_grads_per_batch (bool, optional): Whether to use an efficiency
                    trick to compute the per-example gradients.  If True, `loss_fn` must
                    behave like a `reduction='sum'` or `reduction='sum'` loss function,
                    i.e. `BCELoss(reduction='sum')` or `BCELoss(reduction='mean')`.  If
                    False, `loss_fn` must behave like a `reduction='none'` loss
                    function, i.e. `BCELoss(reduction='none')`.
                    Default: True
            show_progress (bool, optional): Whether to show the progress of
                    computations in both the `fit` and `predict` methods.
                    Default: False
        """
        self.model = model

        self.loss_fn = loss_fn
        self.sample_wise_grads_per_batch = sample_wise_grads_per_batch

        # check `loss_fn`
        self.reduction_type = _check_loss_fn(
            loss_fn, "loss_fn", sample_wise_grads_per_batch
        )

        self.layer_modules = None
        if not (layers is None):
            self.layer_modules = _set_active_parameters(model, layers)
        else:
            self.layer_modules = list(model.modules())

        self.show_progress = show_progress

    def fit(self, dataloader: DataLoader):
        r"""
        Does the computation needed for computing embeddings, which is
        finding the top eigenvectors / eigenvalues of the Hessian, computed
        using `dataloader`.  For this implementation, no such computation is needed.

        Args:
            dataloader (DataLoader): The dataloader containing data needed to learn how
                    to compute the embeddings
        """
        return self

    def predict(self, dataloader: DataLoader) -> Tensor:
        """
        Returns the embeddings for `dataloader`.

        Args:
            dataloader (`DataLoader`): dataloader whose examples to compute embeddings
                    for.
        """
        if self.show_progress:
            dataloader = _progress_bar_constructor(
                self, dataloader, "embeddings", "test data"
            )

        return_device = torch.device("cpu")

        # define a helper function that returns the embeddings for a batch
        def get_batch_embeddings(batch):
            features, labels = tuple(batch[0:-1]), batch[-1]

            # get jacobians (and corresponding name of parameters?)
            jacobians = _compute_jacobian_sample_wise_grads_per_batch(
                self, features, labels, self.loss_fn, self.reduction_type
            )
            with torch.no_grad():
                return _flatten_sample_wise_grads(jacobians).to(device=return_device)
            
        if self.show_progress:
            logging.info("compute embeddings") 
        return torch.cat([get_batch_embeddings(batch) for batch in dataloader], dim=0)
    
    def save(self, path: str):
        """
        This method saves the results of `fit` to a file.  Note that this
        implementation does not compute any results in `fit`, so this method does not
        save anything.

        Args:
            path (str): path of file to save results in.
        """
        pass

    def load(self, path: str):
        """
        Loads the results saved by the `save` method.  Instead of calling `fit`, one
        can instead call `load`.  Note that this implementation does not save any
        results in `save`, so this method does not load anything.

        Args:
            path (str): path of file to load results from.
        """
        pass

    def reset(self):
        """
        Removes the effect of calling `fit` or `load`
        """
        pass


class PCAGradientEmbedder(PCAEmbedder):
    """
    This embedder computes gradients, but unlike `GradientEmbedder`, additionally
    reduces their dimension using PCA.  This is a wrapper around `PCAEmbedder` which
    uses `GradientEmbedder` as the 'base_embedder'.
    """
    def __init__(
        self,
        model: Module,
        layers: Optional[List[str]] = None,
        loss_fn: Optional[Union[Module, Callable]] = None,
        sample_wise_grads_per_batch: bool = False,
        projection_dim: int = 10,
        show_progress: bool = False,
    ):
        """
        Args:
            model (Module): The model used to compute the embeddings.
            layers (list of str, optional): names of modules in which to consider
                    gradients.  If `None` or not provided, all modules will be used.
                    Default: None
            loss_fn (Module or Callable, optional): The loss function used to compute the
                    Hessian.  It should behave like a "reduction" loss function, where
                    reduction is either 'sum', 'mean', or 'none', and have a
                    `reduction` attribute.  For example, `BCELoss(reduction='sum')`
                    could be a valid loss function.  See the caveat under the
                    description for the `sample_wise_grads_per_batch` argument.  If None,
                    the loss is the output of `model`, which is assumed to be a single
                    scalar for a batch.
                    Default: None
            sample_wise_grads_per_batch (bool, optional): Whether to use an efficiency
                    trick to compute the per-example gradients.  If True, `loss_fn` must
                    behave like a `reduction='sum'` or `reduction='sum'` loss function,
                    i.e. `BCELoss(reduction='sum')` or `BCELoss(reduction='mean')`.  If
                    False, `loss_fn` must behave like a `reduction='none'` loss
                    function, i.e. `BCELoss(reduction='none')`.
                    Default: True
            projection_dim (int, optional):  The dimension of the embeddings that are 
                    computed.
                    Default: 10
            show_progress (bool, optional): Whether to show the progress of
                    computations in both the `fit` and `predict` methods.
                    Default: False
        """
        PCAEmbedder.__init__(
            self,
            base_embedder=GradientEmbedder(
                model=model,
                layers=layers,
                loss_fn=loss_fn,
                sample_wise_grads_per_batch=sample_wise_grads_per_batch,
                show_progress=False,
            ),
            projection_dim=projection_dim,
            show_progress=show_progress,
        )