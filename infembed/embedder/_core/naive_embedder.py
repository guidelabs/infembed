from dataclasses import dataclass
import logging
from typing import Any, Callable, List, Optional, Tuple, Union
import warnings
from infembed.embedder._core.embedder_base import EmbedderBase
from infembed.embedder._utils.common import (
    NotFitException,
    _check_loss_fn,
    _compute_batch_loss_influence_function_base,
    _compute_jacobian_sample_wise_grads_per_batch,
    _flatten_params,
    _functional_call,
    _parameter_add,
    _parameter_dot,
    _parameter_linear_combination,
    _parameter_multiply,
    _parameter_to,
    _params_to_names,
    _progress_bar_constructor,
    _set_active_parameters,
    _top_eigen,
    _unflatten_params_factory,
)
from infembed.embedder._utils.gradient import (
    SAMPLEWISE_GRADS_PER_BATCH_SUPPORTED_LAYERS,
    _extract_parameters_from_layers,
)
from infembed.embedder._utils.hvp import AutogradHVP, _dataset_fn
from torch.nn import Module
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm
import dill as pickle
import logging
from infembed.utils.common import profile
from operator import add
import torch.nn as nn


@dataclass
class NaiveEmbedderFitResults:
    """
    stores the results of calling `NaiveEmbedder.fit`
    """

    R: Tensor


def _flatten_forward_factory(
    model: nn.Module,
    loss_fn: Optional[Union[Module, Callable]],
    reduction_type: str,
    unflatten_fn: Callable,
    param_names: List[str],
):
    """
    Given a model, loss function, reduction type of the loss, function that unflattens
    1D tensor input into a tuple of tensors, the name of each tensor in that tuple,
    each of which represents a parameter of `model`, and returns a factory. The factory
    accepts a batch, and returns a function whose input is the parameters represented
    by `param_names`, and output is the total loss of the model with those parameters,
    calculated on the batch. The parameter input to the returned function is assumed to
    be *flattened* via the inverse of `unflatten_fn`, which takes a tuple of tensors to
    a 1D tensor. This returned function, accepting a single flattened 1D parameter, is
    useful for computing the parameter gradient involving the batch as a 1D tensor, and
    the Hessian involving the batch as a 2D tensor. Both quantities are needed to
    calculate the kind of influence scores returned by implementations of
    `InfluenceFunctionBase`.
    """
    # this is the factory that accepts a batch
    def flatten_forward_factory_given_batch(batch):

        # this is the function that factory returns, which is a function of flattened
        # parameters
        def flattened_forward(flattened_params):
            # as everywhere else, the all but the last elements of a batch are
            # assumed to correspond to the features, i.e. input to forward function
            features, labels = tuple(batch[0:-1]), batch[-1]

            _output = _functional_call(
                model, dict(zip(param_names, unflatten_fn(flattened_params))), features
            )

            # compute the total loss for the batch, adjusting the output of
            # `loss_fn` based on `reduction_type`
            return _compute_batch_loss_influence_function_base(
                loss_fn, _output, labels, reduction_type
            )

        return flattened_forward

    return flatten_forward_factory_given_batch


def _compute_dataset_func(
    inputs_dataset: Union[Tuple[Tensor, ...], DataLoader],
    model: Module,
    loss_fn: Optional[Union[Module, Callable]],
    reduction_type: str,
    layer_modules: Optional[List[Module]],
    f: Callable,
    show_progress: bool,
    **f_kwargs,
):
    """
    This function is used to compute higher-order functions of a given model's loss
    over a given dataset, using the model's current parameters. For example, that
    higher-order function `f` could be the Hessian, or a Hessian-vector product.
    This function uses the factory returned by `_flatten_forward_factory`, which given
    a batch, returns the loss for the batch as a function of flattened parameters.
    In particular, for each batch in `inputs_dataset`, this function uses the factory
    to obtain `flattened_forward`, which returns the loss for `model`, using the batch.
    `flattened_forward`, as well as the flattened parameters for `model`, are used by
    argument `f`, a higher-order function, to compute a batch-specific quantity.
    For example, `f` could compute the Hessian via `torch.autograd.functional.hessian`,
    or compute a Hessian-vector product via `torch.autograd.functional.hvp`. Additional
    arguments besides `flattened_forward` and the flattened parameters, i.e. the vector
    in Hessian-vector products, can be passed via named arguments.
    """
    # extract the parameters in a tuple
    params = tuple(
        model.parameters()
        if layer_modules is None
        else _extract_parameters_from_layers(layer_modules)
    )

    # construct functions that can flatten / unflatten tensors, and get
    # names of each param in `params`.
    # Both are needed for calling `_flatten_forward_factory`
    _unflatten_params = _unflatten_params_factory(
        tuple([param.shape for param in params])
    )
    param_names = _params_to_names(params, model)

    # prepare factory
    factory_given_batch = _flatten_forward_factory(
        model,
        loss_fn,
        reduction_type,
        _unflatten_params,
        param_names,
    )

    # the function returned by the factor is evaluated at a *flattened* version of
    # params, so need to create that
    flattened_params = _flatten_params(params)

    # define function of a single batch
    def batch_f(batch):
        flattened_forward = factory_given_batch(batch)  # accepts flattened params
        return f(flattened_forward, flattened_params, **f_kwargs)

    # sum up results of `batch_f`
    if show_progress:
        inputs_dataset = tqdm(inputs_dataset, desc="processing `hessian_dataset` batch")

    return _dataset_fn(inputs_dataset, batch_f, add)


class NaiveEmbedder(EmbedderBase):
    """
    Computes embeddings which are "influence embeddings" - vectors such that the
    dot-product of two examples' embeddings is the "influence" of one example on the
    other, where the general notion of influence is as defined in Koh and Liang
    (https://arxiv.org/abs/1703.04730).  See the paper by Wang and Adebayo et al
    (https://arxiv.org/abs/2312.04712) for more background on influence embeddings.

    Influence embeddings are dependent on the exact definition and implementation of
    influence that is used.  This implementation is based on an implementation of
    influence that explicitly constructs the Hessian.  This implementation thus is not
    scalable, and should be used for testing purposes only.
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
    ):
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
            supported_layers=SAMPLEWISE_GRADS_PER_BATCH_SUPPORTED_LAYERS
            if sample_wise_grads_per_batch
            else None,
        )

        self.projection_dim = projection_dim

        torch.manual_seed(seed)  # for reproducibility

        self.hessian_reg = hessian_reg
        self.hessian_inverse_tol = hessian_inverse_tol

        # infer the device the model is on.  all parameters are assumed to be on the
        # same device
        self.model_device = next(model.parameters()).device

        self.projection_on_cpu = projection_on_cpu
        self.show_progress = show_progress

        self.fit_results = None

    def _retrieve_projections_naive_embedder(
        self,
        dataloader: DataLoader,
        projection_on_cpu: bool,
        show_progress: bool,
    ) -> Tensor:
        r"""
        Returns the matrix `R` described in the documentation for
        `NaiveEmbedder`. In short, `R` has the property that
        :math`H^{-1} \approx RR'`, where `H` is the Hessian. Since this is a "naive"
        implementation, it does so by explicitly forming the Hessian, converting
        it to a 2D tensor, and computing its eigenvectors / eigenvalues, before
        filtering out some eigenvalues and then inverting them. The returned matrix
        `R` represents a set of parameters in parameter space. Since the Hessian
        is obtained by first flattening the parameters, each column of `R` corresponds
        to a *flattened* parameter in parameter space.

        Args:
            dataloader (DataLoader): The returned matrix `R` gives a low-rank
                    approximation of the Hessian `H`. This dataloader defines the
                    dataset used to compute the Hessian that is being approximated.
            projection_on_cpu (bool, optional): Whether to move the projection,
                    i.e. low-rank approximation of the inverse Hessian, to cpu, to save
                    gpu memory.
            show_progress (bool): Computing the Hessian that is being approximated
                    requires summing up the Hessians computed using different batches,
                    which may take a long time. If `show_progress` is true, the number
                    of batches that have been processed will be displayed. It will try
                    to use tqdm if available for advanced features (e.g. time
                    estimation). Otherwise, it will fallback to a simple output of
                    progress.

        Returns a `dataclass` with the following attributes:
            R (Tensor): Tall and skinny tensor with width `projection_dim`
                    (initialization argument). Each column corresponds to a flattened
                    parameter in parameter-space. `R` has the property that
                    :math`H^{-1} \approx RR'`.
        """
        # compute the hessian using the dataloader. hessian is always computed using
        # the training loss function. H is 2D, with each column / row corresponding to
        # a different parameter. we cannot directly use
        # `torch.autograd.functional.hessian`, because it does not return a 2D tensor.
        # instead, to compute H, we first create a function that accepts *flattened*
        # model parameters (i.e. a 1D tensor), and outputs the loss of `self.model`,
        # using those parameters, aggregated over `dataloader`. this function is then
        # passed to `torch.autograd.functional.hessian`. because its input is 1D, the
        # resulting hessian is 2D, as desired. all this functionality is handled by
        # `_compute_dataset_func`.
        H = _compute_dataset_func(
            dataloader,
            self.model,
            self.loss_fn,
            self.reduction_type,
            self.layer_modules,
            torch.autograd.functional.hessian,
            show_progress,
        )

        # H is approximately `vs @ torch.diag(ls) @ vs.T``, using eigendecomposition
        ls, vs = _top_eigen(
            H, self.projection_dim, self.hessian_reg, self.hessian_inverse_tol
        )

        # if no positive eigenvalues exist, we cannot compute a low-rank
        # approximation of the square root of the hessian H, so raise exception
        if len(ls) == 0:
            raise Exception(
                "Hessian has no positive "
                "eigenvalues, so cannot take its square root."
            )

        # `R` is `vs @ torch.diag(ls ** -0.5)`, since H^{-1} is approximately
        #  `vs @ torch.diag(ls ** -1) @ vs.T`
        # see https://en.wikipedia.org/wiki/Eigendecomposition_of_a_matrix#Matrix_inverse_via_eigendecomposition # noqa: E501
        # for details, which mentions that discarding small eigenvalues (as done in
        # `_top_eigen`) reduces noisiness of the inverse.
        ls = (1.0 / ls) ** 0.5
        R = (ls.unsqueeze(0) * vs).to(
            device=torch.device("cpu") if projection_on_cpu else self.model_device
        )
        return NaiveEmbedderFitResults(R)
    
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
        logging.info("start computing Hessian's eigenvectors / eigenvalues")
        self.fit_results = self._retrieve_projections_naive_embedder(
            dataloader, self.projection_on_cpu, self.show_progress
        )
        return self
    
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

        R = self.fit_results.R

        # define a helper function that returns the embeddings for a batch
        def get_batch_embeddings(batch):
            # if `R` is on cpu, and `self.model_device` was not cpu, this implies
            # `R` was too large to fit in gpu memory, and we should do the matrix
            # multiplication of the batch jacobians with `R` separately for each
            # column of `R`, to avoid moving the entire `R` to gpu all at
            # once and running out of gpu memory
            batch_jacobians = _basic_computation_naive_embedder(
                self, batch[0:-1], batch[-1], loss_fn, reduction_type
            )
            if R.device == torch.device(
                "cpu"
            ) and self.model_device != torch.device("cpu"):
                return torch.stack(
                    [
                        torch.matmul(batch_jacobians, R_col.to(batch_jacobians.device))
                        for R_col in R.T
                    ],
                    dim=1,
                ).to(return_device)
            else:
                return torch.matmul(batch_jacobians, R).to(device=return_device)
            
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
            

def _basic_computation_naive_embedder(
    embedder_inst: NaiveEmbedder,
    inputs: Tuple[Any, ...],
    targets: Optional[Tensor] = None,
    loss_fn: Optional[Union[Module, Callable]] = None,
    reduction_type: Optional[str] = None,
) -> Tensor:
    """
    This computes the per-example parameter gradients for a batch, flattened into a
    2D tensor where the first dimension is batch dimension. This is used by
    `NaiveInfluenceFunction` which computes embedding vectors for each example by
    projecting their parameter gradients.
    """
    # `jacobians` contains one tensor for each parameter we compute jacobians for.
    # the first dimension of each tensor is the batch dimension, and the remaining
    # dimensions correspond to the parameter, so that for the tensor corresponding
    # to parameter `p`, its shape is `(batch_size, *p.shape)`
    jacobians = _compute_jacobian_sample_wise_grads_per_batch(
        embedder_inst, inputs, targets, loss_fn, reduction_type
    )

    return torch.stack(
        [
            _flatten_params(tuple(jacobian[i] for jacobian in jacobians))
            for i in range(len(next(iter(jacobians))))
        ],
        dim=0,
    )