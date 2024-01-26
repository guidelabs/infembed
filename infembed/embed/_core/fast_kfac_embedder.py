from infembed.embedder._core.embedder_base import EmbedderBase
from typing import Any, Optional, Union, List, Callable
from infembed.embedder._utils.common import (
    _check_loss_fn,
    _format_inputs_dataset,
    _progress_bar_constructor,
    _set_active_parameters,
    _top_eigen,
    NotFitException,
)
from infembed.embedder._utils.kfac import (
    _LayerCaptureAccumulator,
    _LayerHessianFlattenedIndependentAcumulator,
    BlockSplitTwoD,
    DummySplitTwoD,
    Reshaper,
    _accumulate_with_layer_inputs_and_output_gradients,
)
import torch
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader
import logging
from dataclasses import dataclass
import dill as pickle
import torch.nn as nn


@dataclass
class FastKFACEmbedderFitResults:
    """
    stores the results of calling `FastKFACEmbedder.fit`
    """

    layer_R_A_factors: List[List[Tensor]]
    layer_R_S_factors: List[List[Tensor]]
    layer_R_scales: List[List[Tensor]]


SUPPORTED_LAYERS = [nn.Linear, nn.Conv2d]


class FastKFACEmbedder(EmbedderBase):
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
    PSD, instead of the Hessian.  2) The GNH is assumed to be block-diagonal, where
    the blocks correspond to parameters from different layers.

    This implementation is "fast" in that it never instantiates the Hessian, and
    furthermore, optionally constructions a block diagonal approximation of the Hessian
    *within* a single layer.
    """

    def __init__(
        self,
        model: Module,
        layers: Optional[List[str]] = None,
        loss_fn: Optional[Union[Module, Callable]] = None,
        test_loss_fn: Optional[Union[Module, Callable]] = None,
        sample_wise_grads_per_batch: bool = False,
        projection_dim: Optional[int] = 50,
        layer_block_projection_dim: Optional[int] = None,
        seed: int = 0,
        hessian_reg: float = 1e-6,
        hessian_inverse_tol: float = 1e-5,
        projection_on_cpu: bool = True,
        show_progress: bool = False,
        per_layer_blocks: int = 1,
    ):
        """
        Args:
            model (Module): The model used to compute the embeddings.
            layers (list of str, optional): names of modules in which to consider
                    gradients.  If `None` or not provided, all modules will be used.
                    There is a caveat: `FastKFACEmbedder` can only consider gradients
                    in layers which are `Linear` or `Conv2d`.  Thus regardless of
                    the modules specified by `layers`, only layers in them which are
                    of those types will be used for calculating gradients.
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
            projection_dim (int, optional): Directly determines the embedding's
                    dimension, if not None.  If None and `layer_block_projection_dim`
                    is not None, the latter specifies the embedding dimension.  If both
                    arguments are None, all parameters in layers specified by `layers`
                    contribute to the embedding, i.e. there is no dimension reduction
                    done, so that the embedding dimension is the number of parameters
                    whose gradients considered.
                    Default: 50
            layer_block_projection_dim (int, optional): This argument is only used if
                    `projection_dim` is None, and specifies the number of
                    dimensions in the embedding that come from each block in each
                    layer. Each block is assumed to contribute the same number of dimensions.
                    Default: None
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
            per_layer_blocks (int, optional): The number of blocks in the block diagonal
                    approximation of the Hessian within a single layer.
                    Default: 1

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
            # TODO: should let `self.layer_modules` only contain supported layers
            self.layer_modules = _set_active_parameters(
                model, layers, supported_layers=SUPPORTED_LAYERS
            )
        else:
            # only use supported layers.  TODO: add warning that some layers are not supported
            self.layer_modules = [
                layer for layer in model.modules() if type(layer) in SUPPORTED_LAYERS
            ]

        # below initializations are specific to `KFACEmbedder`
        self.layer_block_projection_dim = layer_block_projection_dim
        self.projection_dim = projection_dim

        torch.manual_seed(seed)  # for reproducibility

        self.hessian_reg = hessian_reg
        self.hessian_inverse_tol = hessian_inverse_tol

        # infer the device the model is on.  all parameters are assumed to be on the
        # same device
        self.model_device = next(model.parameters()).device

        self.projection_on_cpu = projection_on_cpu
        self.show_progress = show_progress

        # determine how to split each layer's parameters
        if per_layer_blocks == 1:
            self.layer_split_two_ds = [DummySplitTwoD() for _ in self.layer_modules]
        else:
            self.layer_split_two_ds = [
                BlockSplitTwoD(num_blocks=per_layer_blocks) for _ in self.layer_modules
            ]

        self.fit_results: Optional[FastKFACEmbedderFitResults] = None

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
        self.fit_results = self._retrieve_projections_fast_kfac_embedder(
            dataloader,
            self.projection_on_cpu,
            self.show_progress,
        )
        return self

    def _retrieve_projections_fast_kfac_embedder_helper(
        self,
        dataloader: DataLoader,
        projection_on_cpu: bool,
        show_progress: bool,
        layer_block_projection_dim: Optional[int] = None,
        eigenvalue_threshold: Optional[int] = None,
        return_eigenvalue_only: bool = False,
    ):
        """
        Three use cases: 1) extract a specified number of eigenvalues per layer based
        on `layer_block_projection_dim`, and only return the eigenvalues, 2) extract a
        specified number of eigenvectors / eigenvalues per layer based on
        `layer_block_projection_dim`, 3) extract eigenvectors / eigenvalues for which
        the eigenvalue is above `eigenvalue_threshold`.
        """
        # get A and S for each layer
        logging.info("compute training data statistics")
        layer_accumulators = [
            _LayerHessianFlattenedIndependentAcumulator(layer, split_two_d)
            for (layer, split_two_d) in zip(self.layer_modules, self.layer_split_two_ds)
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
        # store A and S over all layers and over all blocks in layers
        layer_As = [
            [
                accumulator.results()
                for accumulator in layer_accumulator.layer_A_accumulators
            ]
            for layer_accumulator in layer_accumulators
        ]
        layer_Ss = [
            [
                accumulator.results()
                for accumulator in layer_accumulator.layer_S_accumulators
            ]
            for layer_accumulator in layer_accumulators
        ]

        # for each layer, and for each block in a layer, extract factors of decomposition
        # TODO: consider adjusting `projection_dim` based on the number of blocks
        layer_R_A_factors = []
        layer_R_S_factors = []
        layer_R_ls = []
        # layer_R_scales = []

        projection_device = (
            torch.device("cpu") if projection_on_cpu else self.model_device
        )

        # logging.info("compute factors")
        for k, (layer_A, layer_S, layer) in enumerate(
            zip(layer_As, layer_Ss, self.layer_modules)
        ):
            logging.info(f"compute factors for layer {layer}")

            R_A_factors = []
            R_S_factors = []
            R_ls = []
            # R_scales = []

            for A, S in zip(layer_A, layer_S):
                # do SVD of A and S.  don't truncate based `hessian_inverse_tol`
                A_ls, A_vs = _top_eigen(
                    A,
                    None,  # TODO: to save memory, use heuristic to limit returns
                    self.hessian_reg,
                    0,
                )

                S_ls, S_vs = _top_eigen(
                    S,
                    None,
                    self.hessian_reg,
                    0,
                )
                # figure out top eigenvalues of `kron(A, S)` and the which factors' outer
                # product generates the corresponding eigenvectors
                ls = torch.outer(A_ls, S_ls)
                num_S_factors = len(S_ls)

                # figure out indices of eigenvalues / eigenvectors to keep
                if eigenvalue_threshold is None:
                    # if keeping based on `layer_block_projection_dim`
                    flattened_indices_to_keep = torch.argsort(
                        ls.view(-1), descending=True
                    )[:layer_block_projection_dim]
                    # assert layer_block_projection_dim is None
                else:
                    # if keeping based on `eigenvalue_threshold`
                    # TODO: can sort so that largest eigenvalue is first
                    eps = 1e-4
                    flattened_indices_to_keep = [
                        idx
                        for (idx, l) in enumerate(ls.view(-1))
                        if l > (eigenvalue_threshold + eps)
                    ]

                top_ijs = [
                    (int(flattened_idx / num_S_factors), flattened_idx % num_S_factors)
                    for flattened_idx in flattened_indices_to_keep
                ]

                # top_ijs = [
                #     (int(flattened_pos / num_S_factors), flattened_pos % num_S_factors)
                #     for flattened_pos in torch.argsort(ls.view(-1), descending=True)[
                #         :projection_dim
                #     ]
                # ]

                top_ls = torch.Tensor([ls[i, j] for (i, j) in top_ijs])

                # add factors for both if not `return_eigenvalue_only`.  otherwise,
                # return dummy's for the factors
                if not return_eigenvalue_only:
                    # appending the factors for a block
                    if len(top_ijs) == 0:
                        # if not using factors from block, append dummy value of `None`
                        R_A_factors.append(None)
                        R_S_factors.append(None)
                    else:
                        R_A_factors.append(
                            torch.stack([A_vs[:, i] for (i, _) in top_ijs], dim=0).to(
                                device=projection_device
                            )
                        )
                        R_S_factors.append(
                            torch.stack([S_vs[:, j] for (_, j) in top_ijs], dim=0).to(
                                device=projection_device
                            )
                        )

                R_ls.append(top_ls.to(device=projection_device))
                # R_scales.append((top_ls**-0.5).to(device=projection_device))

            layer_R_A_factors.append(R_A_factors)
            layer_R_S_factors.append(R_S_factors)
            layer_R_ls.append(R_ls)
            # layer_R_scales.append(R_scales)

        return layer_R_A_factors, layer_R_S_factors, layer_R_ls
        # return FastKFACEmbedderFitResults(
        #     layer_R_A_factors, layer_R_S_factors, layer_R_scales
        # )

    def _retrieve_projections_fast_kfac_embedder(
        self,
        dataloader: DataLoader,
        projection_on_cpu: bool,
        show_progress: bool,
    ):
        """
        For each layer, compute expected outer product of layer inputs and output
        gradients (A and S).  This should be possible given total memory is roughly
        same as model size.  Then, for each layer, extract the two kinds of vectors.

        A helper function is used to return embeddings of the desired dimension.
        """
        if (self.layer_block_projection_dim is not None) or (
            self.projection_dim is None and self.layer_block_projection_dim is None
        ):
            # if specified `layer_block_projection_dim`, or using all parameters,
            # directly compute eigenvectors / eigenvectors
            (
                layer_R_A_factors,
                layer_R_S_factors,
                layer_R_ls,
            ) = self._retrieve_projections_fast_kfac_embedder_helper(
                dataloader,
                projection_on_cpu,
                show_progress,
                layer_block_projection_dim=self.layer_block_projection_dim,
                eigenvalue_threshold=None,
                return_eigenvalue_only=False,
            )
        else:
            assert self.projection_dim is not None
            # first get all eigenvalues to get eigenvalue threshold
            logging.info("compute factors, first pass to get eigenvalue threshold")
            _, _, layer_R_ls = self._retrieve_projections_fast_kfac_embedder_helper(
                dataloader,
                projection_on_cpu,
                show_progress,
                layer_block_projection_dim=None,  # TODO: for now return all eigenvalues
                eigenvalue_threshold=None,
                return_eigenvalue_only=True,
            )
            # figure out eigenvalue threshold
            all_ls = torch.Tensor([l for R_ls in layer_R_ls for ls in R_ls for l in ls])
            sorted_ls = torch.sort(all_ls, descending=True).values
            eigenvalue_threshold = (
                sorted_ls[self.projection_dim]
                if len(sorted_ls) >= self.projection_dim + 1
                else None
            )

            logging.info("compute factors, second pass to get eigenvalue threshold")
            (
                layer_R_A_factors,
                layer_R_S_factors,
                layer_R_ls,
            ) = self._retrieve_projections_fast_kfac_embedder_helper(
                dataloader,
                projection_on_cpu,
                show_progress,
                layer_block_projection_dim=None,  # is ignored because `eigenvalue_threshold` is specified
                eigenvalue_threshold=eigenvalue_threshold,
                return_eigenvalue_only=False,
            )

        # scale is inverse of eigenvalue
        layer_R_scales = [[ls**-0.5 for ls in R_ls] for R_ls in layer_R_ls]
        return FastKFACEmbedderFitResults(
            layer_R_A_factors, layer_R_S_factors, layer_R_scales
        )

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
                self, dataloader, "embeddings", "training data"
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
            # for each layer, get the input and output gradient
            layer_capture_accumulators = (
                _accumulate_with_layer_inputs_and_output_gradients(
                    [_LayerCaptureAccumulator() for _ in self.layer_modules],
                    self.model,
                    _format_inputs_dataset(batch),
                    self.layer_modules,
                    reduction_type,
                    loss_fn,
                    show_progress=False,
                )
            )
            layer_inputs = [
                accumulator.layer_inputs[0]  # because only capture a single batch
                for accumulator in layer_capture_accumulators
            ]
            layer_output_gradients = [
                accumulator.layer_output_gradients[0]
                for accumulator in layer_capture_accumulators
            ]

            with torch.no_grad():
                # iterate over batches, layers, and blocks within layers

                def get_batch_layer_embeddings(
                    R_A_factors,
                    R_S_factors,
                    R_scales,
                    layer,
                    layer_input,
                    layer_output_gradient,
                    layer_split_two_d,
                ):
                    # input is layer, layer input, layer output gradient
                    # those are sufficient to create 2D W for the layer
                    # then, simulate dot-product with `outer(S, A)`
                    layer_input = Reshaper.layer_input_to_two_d(layer, layer_input)
                    layer_output_gradient = Reshaper.layer_output_gradient_to_two_d(
                        layer, layer_output_gradient
                    )

                    def get_batch_layer_block_embeddings(
                        block_A_factors,
                        block_S_factors,
                        block_scales,
                        block_layer_input,
                        block_layer_output_gradient,
                    ):
                        # if no factors for the block, as indicated by dummy value
                        # of `None` for them, return embedding with width of 0
                        if block_A_factors is None:
                            batch_size = block_layer_input.shape[0]
                            return torch.zeros((batch_size, 0))

                        # for each factor, simulate dot-product with `outer(S, A)`
                        def get_batch_layer_block_coordinates(
                            block_A_factor,
                            block_S_factor,
                            block_scale,
                            block_layer_input,
                            block_layer_output_gradient,
                        ):
                            return torch.sum(
                                (block_layer_output_gradient @ block_S_factor)
                                * (block_layer_input @ block_A_factor)
                                * block_scale,
                                dim=tuple(
                                    range(1, len(block_layer_output_gradient.shape) - 1)
                                ),
                            )

                        return torch.stack(
                            [
                                get_batch_layer_block_coordinates(
                                    block_A_factor,
                                    block_S_factor,
                                    block_scale,
                                    block_layer_input,
                                    block_layer_output_gradient,
                                )
                                for (
                                    block_A_factor,
                                    block_S_factor,
                                    block_scale,
                                ) in zip(
                                    block_A_factors.to(self.model_device),
                                    block_S_factors.to(self.model_device),
                                    block_scales.to(self.model_device),
                                )
                            ],
                            dim=1,
                        )

                    return torch.cat(
                        [
                            get_batch_layer_block_embeddings(
                                block_A_factors,
                                block_S_factors,
                                block_scales,
                                block_layer_input,
                                block_layer_output_gradient,
                            )
                            for (
                                block_A_factors,
                                block_S_factors,
                                block_scales,
                                block_layer_input,
                                block_layer_output_gradient,
                            ) in zip(
                                R_A_factors,
                                R_S_factors,
                                R_scales,
                                layer_split_two_d(layer_input),
                                layer_split_two_d(layer_output_gradient),
                            )
                        ],
                        dim=1,
                    )

                with torch.no_grad():
                    return torch.cat(
                        [
                            get_batch_layer_embeddings(
                                R_A_factors,
                                R_S_factors,
                                R_scales,
                                layer,
                                layer_input,
                                layer_output_gradient,
                                layer_split_two_d,
                            )
                            for (
                                R_A_factors,
                                R_S_factors,
                                R_scales,
                                layer,
                                layer_input,
                                layer_output_gradient,
                                layer_split_two_d,
                            ) in zip(
                                self.fit_results.layer_R_A_factors,
                                self.fit_results.layer_R_S_factors,
                                self.fit_results.layer_R_scales,
                                self.layer_modules,
                                layer_inputs,
                                layer_output_gradients,
                                self.layer_split_two_ds,
                            )
                        ],
                        dim=1,
                    ).to(return_device)

        logging.info("compute embeddings")
        return torch.cat([get_batch_embeddings(batch) for batch in dataloader], dim=0)

    def save(self, path: str):
        """
        This method saves the results of `fit` to a file.

        Args:
            path (str): path of file to save results in.
        """
        with open(path, "wb") as f:
            pickle.dump(self.fit_results, f)

    def load(self, path: str):
        """
        Loads the results saved by the `save` method.  Instead of calling `fit`, one
        can instead call `load`.

        Args:
            path (str): path of file to load results from.
        """
        with open(path, "rb") as f:
            self.fit_results = pickle.load(f)

    def reset(self):
        """
        Removes the effect of calling `fit` or `load`
        """
        self.fit_results = None
