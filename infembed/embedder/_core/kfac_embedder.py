from infembed.embedder._core.embedder_base import EmbedderBase
from typing import Any, Optional, Tuple, Union, List, Callable
from infembed.embedder._utils.common import (
    _check_loss_fn,
    _compute_jacobian_sample_wise_grads_per_batch,
    _progress_bar_constructor,
    _set_active_parameters,
    _top_eigen,
)
from infembed.embedder._utils.kfac import (
    _LayerHessianFlattenedAcumulator,
    _LayerHessianFlattenedIndependentAcumulator,
    Reshaper,
    _accumulate_with_layer_inputs_and_output_gradients,
    _extract_layer_parameters,
)
import torch
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader


class KFACEmbedder(EmbedderBase):
    """
    Computes influence embeddings using the KFAC approximation to the Hessian, which
    makes the following approximations: 1) it computes the Gauss-Newton Hessian
    (GNH), which is always guaranteed to be PSD, instead of the Hessian.  2) The GNH
    is assumed to be block-diagonal, where the blocks correspond to parameters from
    different layers.

    KFAC can potentially use an additional approximation: when computing the GNH for
    a given layer, assume the input activations and pseudo-gradients are independent.
    This implementations allows to specify whether to use this independence assumption
    regarding those factors.

    This implementation is intended mostly for testing purposes, because it directly
    instantiates an estimate of the Hessian, which is memory-intensive.  For
    practical use cases, one should use the fast implementation `FastKFACEmbedder`.

    TODO's:
    - when `sample_wise_grads_per_batch=True`, handle the case where the same module
        is called more than once in a forward pass
    - introduce a `projection_dim_per_layer` options and allow it to be false, so that
        eigenvalues across all per-layer GNH's are considered to choose the projection
    - figure out what happens if one module in `layers` is a submodule of another
        module
    - figure out how to turn into sklearn pipeline
    """
    def __init__(
        self,
        model: Module,
        layers: Optional[List[str]] = None,
        loss_fn: Optional[Union[Module, Callable]] = None,
        test_loss_fn: Optional[Union[Module, Callable]] = None,
        sample_wise_grads_per_batch: bool = False,
        projection_dim: Union[int, List[int]] = 50,
        seed: int = 0,
        hessian_reg: float = 1e-6,
        hessian_inverse_tol: float = 1e-5,
        projection_on_cpu: bool = True,
        show_progress: bool = False,
        independent_factors: bool = True,
    ):
        """
        Args:
            model (Module): The model used to compute the embeddings.
            layers (list of str, optional): names of modules in which to consider
                    gradients.  If `None` or not provided, all modules will be used.
                    Default: `None`
            loss_fn (Module or Callable, optional): The loss function used to compute the
                    Hessian.  TODO: specify what are valid losses
            test_loss_fn: (Module or callable, optional): The loss function used to compute
                    the 'influence explanations'
            sample_wise_grads_per_batch (bool, optional): Whether to use an efficiency
                    trick to compute the per-example gradients.  Only works if layers we
                    consider gradients in are `Linear` or `Conv2d` layers.
            projection_dim (int or list of int): This argument specifies the number of
                    dimensions in the embedding that come from each layer.  This can either
                    be a list of integers with the same number of elements as `layers`, or
                    a single integer, in which case every layer contributes the same
                    number of dimensions.
        """
        self.model = model

        self.loss_fn = loss_fn
        # If test_loss_fn not provided, it's assumed to be same as loss_fn
        self.test_loss_fn = loss_fn if test_loss_fn is None else test_loss_fn
        self.sample_wise_grads_per_batch = sample_wise_grads_per_batch

        # we save the reduction type for both `loss_fn` and `test_loss_fn` because
        # 1) if `sample_wise_grads_per_batch` is true, the reduction type is needed
        # to compute per-example gradients, and 2) regardless, reduction type for
        # `loss_fn` is needed to compute the Hessian.

        # check `loss_fn`
        self.reduction_type = _check_loss_fn(
            loss_fn, "loss_fn", sample_wise_grads_per_batch
        )
        # check `test_loss_fn` if it was provided
        if not (test_loss_fn is None):
            self.test_reduction_type = _check_loss_fn(
                test_loss_fn, "test_loss_fn", sample_wise_grads_per_batch
            )
        else:
            self.test_reduction_type = self.reduction_type

        self.layer_modules = None
        if not (layers is None):
            self.layer_modules = _set_active_parameters(model, layers)
        else:
            self.layer_modules = list(model.modules())

        # below initializations are specific to `KFACEmbedder`

        # expand `projection_dim` to list if not one
        if not isinstance(projection_dim, list):
            projection_dim = [projection_dim for _ in self.layer_modules]
        self.projection_dims = projection_dim

        torch.manual_seed(seed)  # for reproducibility

        self.hessian_reg = hessian_reg
        self.hessian_inverse_tol = hessian_inverse_tol

        # infer the device the model is on.  all parameters are assumed to be on the
        # same device
        self.model_device = next(model.parameters()).device

        self.projection_on_cpu = projection_on_cpu
        self.show_progress = show_progress
        self.independent_factors = independent_factors

        self.layer_Rs = None

    def fit(
        self,
        dataloader: DataLoader,
    ):
        r"""
        Does the computation needed for computing influence embeddings, which is
        finding the top eigenvectors / eigenvalues of the Hessian, computed
        using `dataloader`.

        Args:
            dataloader (DataLoader): The dataloader containing data needed to learn how
                    to compute the embeddings
        """
        self.layer_Rs = self._retrieve_projections_kfac_influence_function(
            dataloader,
            self.projection_on_cpu,
            self.show_progress,
            self.independent_factors,
        )

    def _retrieve_projections_kfac_influence_function(
        self,
        dataloader: DataLoader,
        projection_on_cpu: bool,
        show_progress: bool,
        independent_factors: bool,
    ):
        """
        For each layer, returns the basis of truncated SVD.  Explicitly form the
        Hessian in flattened (2D) form, so could directly apply SVD to it.
        """
        if independent_factors:
            layer_accumulators = [
                _LayerHessianFlattenedIndependentAcumulator(layer)
                for layer in self.layer_modules
            ]
            _accumulate_with_layer_inputs_and_output_gradients(
                layer_accumulators,
                self.model,
                dataloader,
                self.layer_modules,
                self.reduction_type,
                self.loss_fn,
                show_progress,
            )
            layer_hessians_flattened = [
                # assuming each layer accumulator only has 1 element, i.e. not doing any
                # additional block-wise approximation of the Hessian in a layer, hence the
                # `[0]`
                torch.kron(
                    layer_accumulator.layer_A_accumulators[0].results(),
                    layer_accumulator.layer_S_accumulators[0].results(),
                )
                for layer_accumulator in layer_accumulators
            ]

        else:
            layer_accumulators = [
                _LayerHessianFlattenedAcumulator(layer) for layer in self.layer_modules
            ]
            _accumulate_with_layer_inputs_and_output_gradients(
                layer_accumulators,
                self.model,
                dataloader,
                self.layer_modules,
                self.reduction_type,
                self.loss_fn,
                show_progress,
            )
            layer_hessians_flattened = [
                layer_accumulator.accumulator.results()
                for layer_accumulator in layer_accumulators
            ]

        layer_Rs = []

        projection_device = (
            torch.device("cpu") if projection_on_cpu else self.model_device
        )

        for projection_dim, layer_hessian_flattened in zip(
            self.projection_dims, layer_hessians_flattened
        ):
            ls, vs = _top_eigen(
                layer_hessian_flattened,
                projection_dim,
                self.hessian_reg,
                self.hessian_inverse_tol,
            )
            ls = (1.0 / ls) ** 0.5
            layer_Rs.append((ls.unsqueeze(0) * vs).to(device=projection_device))

        return layer_Rs

    def predict(
        self,
        dataloader: DataLoader
    ) -> Tensor:
        """
        Returns the influence embeddings for `dataloader`.

        Args:
            dataloader (`DataLoader`): dataloader whose examples to compute influence
                    embeddings for.
        """
        if self.show_progress:
            dataloader = _progress_bar_constructor(
                self, dataloader, "influence embeddings", "training data"
            )
            
        # always return embeddings on cpu
        return_device = torch.device("cpu")

        # choose the correct loss function and reduction type based on `test`
        # actually, `test` is always true
        test = True
        loss_fn = self.test_loss_fn if test else self.loss_fn
        reduction_type = self.test_reduction_type if test else self.reduction_type

        # define a helper function that returns the embeddings for a batch
        def get_batch_embeddings(batch):
            features, labels = tuple(batch[0:-1]), batch[-1]

            # get jacobians (and corresponding name of parameters?)
            jacobians = _compute_jacobian_sample_wise_grads_per_batch(
                self, features, labels, loss_fn, reduction_type
            )
            
            with torch.no_grad():

                def get_batch_layer_embeddings(R, layer, layer_params):
                    layer_params_flattened = torch.stack(
                        [
                            Reshaper.tot_to_one_d(layer, layer_param)  # [0]
                            for layer_param in zip(*layer_params)
                        ],
                        dim=0,
                    )
                    return torch.stack(
                        [
                            torch.matmul(
                                layer_params_flattened,
                                R_col.to(layer_params_flattened.device),
                            )
                            for R_col in R.T.to(self.model_device)
                        ],
                        dim=1,
                    ).to(return_device)

                return torch.cat(
                    [
                        get_batch_layer_embeddings(R, layer, layer_params)
                        for (R, layer, (_, layer_params)) in zip(
                            self.layer_Rs,
                            self.layer_modules,
                            _extract_layer_parameters(
                                self.model, self.layer_modules, jacobians
                            ),
                        )
                    ],
                    dim=1,
                )

        return torch.cat(
            [get_batch_embeddings(batch) for batch in dataloader], dim=0
        )