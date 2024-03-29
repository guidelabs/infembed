import logging
from infembed.embedder._core.embedder_base import EmbedderBase
from typing import Optional, Union, List, Callable
from infembed.embedder._utils.common import (
    NotFitException,
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
from dataclasses import dataclass
import dill as pickle
import torch.nn as nn


@dataclass
class KFACEmbedderFitResults:
    layer_Rs: List[Tensor]


SUPPORTED_LAYERS = [nn.Linear, nn.Conv2d]


class KFACEmbedder(EmbedderBase):
    """
    Computes embeddings which are "influence embeddings" - vectors such that the
    dot-product of two examples' embeddings is the "influence" of one example on the
    other, where the general notion of influence is as defined in Koh and Liang
    (https://arxiv.org/abs/1703.04730).  See the paper by Wang and Adebayo et al
    (https://arxiv.org/abs/2312.04712) for more background on influence embeddings.

    Influence embeddings are dependent on the exact definition and implementation of
    influence that is used.  This implementation is based on an implementation of
    influence (see Grosse and Bae et al, https://arxiv.org/abs/2308.03296) that uses
    a KFAC approximation to the Hessian, which makes the following approximations:
    1) it computes the Gauss-Newton Hessian (GNH), which is always guaranteed to be
    PSD, instead of the Hessian.  2) The GNH is assumed to be block-diagonal, where the
    blocks correspond to parameters from different layers.

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
        layer_projection_dim: Optional[int] = 50,
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
                    There is a caveat: `KFACEmbedder` can only consider gradients in
                    layers which are `Linear` or `Conv2d`.  If `layers` is provided,
                    they should satisfy these constraints.  If `layers is not provided,
                    the implementation automatically selects layers which satisfies
                    these constraints.
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
            test_loss_fn: (Module or callable, optional): The loss function used to compute
                    the 'influence explanations'.  This argument should not matter for
                    most use cases.  If None, is assumed to be the same as `loss_fn`.
            sample_wise_grads_per_batch (bool, optional): Whether to use an efficiency
                    trick to compute the per-example gradients.  If True, `loss_fn` must
                    behave like a `reduction='sum'` or `reduction='sum'` loss function,
                    i.e. `BCELoss(reduction='sum')` or `BCELoss(reduction='mean')`.  If
                    False, `loss_fn` must behave like a `reduction='none'` loss
                    function, i.e. `BCELoss(reduction='none')`.
                    Default: True
            layer_projection_dim (int, optional): This argument specifies the number of
                    dimensions in the embedding that come from each layer. Each layer
                    is assumed to contribute the same number of dimensions.
                    Default: 50
            seed (int, optional): Random seed for reproducibility.
                    Default: 0
            hessian_reg (float, optional): This implementation computes the eigenvalues /
                    eigenvectors of Hessians.  We add an entry to the Hessian's
                    diagonal entries before computing them.  This is that entry.
                    Default: 1e-6
            hessian_inverse_tol (float): This implementation computes the
                    pseudo-inverse of the (square root of) Hessians.  This is the
                    tolerance to use in that computation.
                    Default: 1e-6
            projection_on_cpu (bool, optional): Whether to move the projection,
                    i.e. low-rank approximation of the inverse Hessian, to cpu, to save
                    gpu memory.
                    Default: True
            show_progress (bool, optional): Whether to show the progress of
                    computations in both the `fit` and `predict` methods.
                    Default: False
            independent_factors (bool, optional): Whether to make the approximating
                    assumption that the input and output gradient for a layer are
                    independent when estimating the Hessian.
                    Default: True
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

        self.layer_modules = _set_active_parameters(
            model,
            layers,
            supported_layers=SUPPORTED_LAYERS
        )

        # below initializations are specific to `KFACEmbedder`

        self.layer_projection_dim = layer_projection_dim

        torch.manual_seed(seed)  # for reproducibility

        self.hessian_reg = hessian_reg
        self.hessian_inverse_tol = hessian_inverse_tol

        # infer the device the model is on.  all parameters are assumed to be on the
        # same device
        self.model_device = next(model.parameters()).device

        self.projection_on_cpu = projection_on_cpu
        self.show_progress = show_progress
        self.independent_factors = independent_factors

        self.fit_results: Optional[KFACEmbedderFitResults] = None

    def fit(
        self,
        dataloader: DataLoader,
    ):
        r"""
        Does the computation needed for computing embeddings, which is
        finding the top eigenvectors / eigenvalues of the Hessian, computed
        using `dataloader`.

        Args:
            dataloader (DataLoader): The dataloader containing data needed to learn how
                    to compute the embeddings
        """
        self.fit_results = self._retrieve_projections_kfac_embedder(
            dataloader,
            self.projection_on_cpu,
            self.show_progress,
            self.independent_factors,
        )
        return self

    def _retrieve_projections_kfac_embedder(
        self,
        dataloader: DataLoader,
        projection_on_cpu: bool,
        show_progress: bool,
        independent_factors: bool,
    ) -> KFACEmbedderFitResults:
        """
        For each layer, returns the basis of truncated SVD.  Explicitly form the
        Hessian in flattened (2D) form, so could directly apply SVD to it.
        """
        logging.info("compute training data statistics")
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
                accumulate_device=None,
            )
            try:
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
            except:
                import pdb
                pdb.set_trace()

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

        logging.info("compute factors")
        for layer_hessian_flattened, layer in zip(
            layer_hessians_flattened, self.layer_modules
        ):
            logging.info(f"compute factors for layer {layer}")
            ls, vs = _top_eigen(
                layer_hessian_flattened,
                self.layer_projection_dim,
                self.hessian_reg,
                self.hessian_inverse_tol,
            )
            ls = (1.0 / ls) ** 0.5
            layer_Rs.append((ls.unsqueeze(0) * vs).to(device=projection_device))

        return KFACEmbedderFitResults(layer_Rs)

    def predict(self, dataloader: DataLoader) -> Tensor:
        """
        Returns the embeddings for `dataloader`.

        Args:
            dataloader (`DataLoader`): dataloader whose examples to compute embeddings
                    for.
        """
        if self.fit_results is None:
            raise NotFitException(
                "The results needed to compute embeddings not available.  Please either call the `fit` or `load` methods."
            )

        if self.show_progress:
            dataloader = _progress_bar_constructor(
                self, dataloader, "embeddings", "test data"
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
                            self.fit_results.layer_Rs,
                            self.layer_modules,
                            _extract_layer_parameters(
                                self.model, self.layer_modules, jacobians
                            ),
                        )
                    ],
                    dim=1,
                )

        with torch.no_grad():
            logging.info("compute embeddings")
            return torch.cat(
                [get_batch_embeddings(batch) for batch in dataloader], dim=0
            )

    def save(self, path: str):
        """
        This method saves the results of `fit` to a file.

        Args:
            path (str): path of file to save results in.
        """
        with open(path, "wb") as f:
            pickle.dump(self.fit_results, f)

    def load(self, path: str, projection_on_cpu: bool = True):
        """
        Loads the results saved by the `save` method.  Instead of calling `fit`, one
        can instead call `load`.

        Args:
            path (str): path of file to load results from.
            projection_on_cpu (bool, optional): whether to load the results onto cpu.
                    results will not be moved onto gpu if model is not on gpu.
                    Default: True
        """
        with open(path, "rb") as f:
            self.fit_results = pickle.load(f)
        projection_device = (
            torch.device("cpu") if projection_on_cpu else self.model_device
        )
        self.fit_results.layer_Rs = [layer_R.to(device=projection_device) for layer_R in self.fit_results.layer_Rs]

    def reset(self):
        """
        Removes the effect of calling `fit` or `load`
        """
        self.fit_results = None
