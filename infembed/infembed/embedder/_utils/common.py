from functools import reduce
from typing import Any, Iterable, List, Optional, Union, Callable, Tuple, overload
from infembed.embedder._utils.gradient import (
    _compute_jacobian_wrt_params,
    _compute_jacobian_wrt_params_with_sample_wise_trick,
)
import torch
from torch.nn import Module
from torch import Tensor
import warnings
from torch.utils.data import Dataset, DataLoader
from infembed.embedder._core.embedder_base import EmbedderBase
from infembed.embedder._utils.progress import progress
from torch import Tensor
import torch.nn as nn


def _check_loss_fn(
    loss_fn: Optional[Union[Module, Callable]],
    loss_fn_name: str,
    sample_wise_grads_per_batch: Optional[bool] = None,
) -> str:
    """
    This checks whether `loss_fn` satisfies the requirements assumed of all
    implementations of `TracInCPBase`. It works regardless of whether the
    implementation has the `sample_wise_grads_per_batch` attribute.
    It returns the reduction type of the loss_fn. If `sample_wise_grads_per_batch`
    if not provided, we assume the implementation does not have that attribute.
    """
    # if `loss_fn` is `None`, there is nothing to check. then, the reduction type is
    # only used by `_compute_jacobian_wrt_params_with_sample_wise_trick`, where
    # reduction type should be "sum" if `loss_fn` is `None`.
    if loss_fn is None:
        return "sum"

    # perhaps since `Module` is an implementation of `Callable`, this has redundancy
    assert isinstance(loss_fn, Module) or callable(loss_fn)

    reduction_type = "none"

    # If we are able to access the reduction used by `loss_fn`, we check whether
    # the reduction is compatible with `sample_wise_grads_per_batch`, if it has the
    # attribute.
    if hasattr(loss_fn, "reduction"):
        reduction = loss_fn.reduction  # type: ignore
        if sample_wise_grads_per_batch is None:
            assert reduction in [
                "sum",
                "mean",
            ], 'reduction for `loss_fn` must be "sum" or "mean"'
            reduction_type = str(reduction)
        elif sample_wise_grads_per_batch:
            assert reduction in ["sum", "mean"], (
                'reduction for `loss_fn` must be "sum" or "mean" when '
                "`sample_wise_grads_per_batch` is True"
            )
            reduction_type = str(reduction)
        else:
            assert reduction == "none", (
                'reduction for `loss_fn` must be "none" when '
                "`sample_wise_grads_per_batch` is False"
            )
    else:
        # if we are unable to access the reduction used by `loss_fn`, we warn
        # the user about the assumptions we are making regarding the reduction
        # used by `loss_fn`
        if sample_wise_grads_per_batch is None:
            warnings.warn(
                f'Since `{loss_fn_name}` has no "reduction" attribute, the '
                f'implementation  assumes that `{loss_fn_name}` is a "reduction" loss '
                "function that reduces the per-example losses by taking their *sum*. "
                f"If `{loss_fn_name}` instead reduces the per-example losses by "
                f"taking their mean, please set the reduction attribute of "
                f'`{loss_fn_name}` to "mean", i.e. '
                f'`{loss_fn_name}.reduction = "mean"`.'
            )
            reduction_type = "sum"
        elif sample_wise_grads_per_batch:
            warnings.warn(
                f"Since `{loss_fn_name}`` has no 'reduction' attribute, and "
                "`sample_wise_grads_per_batch` is True, the implementation assumes "
                f"that `{loss_fn_name}` is a 'reduction' loss function that reduces "
                f"the per-example losses by taking their *sum*. If `{loss_fn_name}` "
                "instead reduces the per-example losses by taking their mean, "
                f'please set the reduction attribute of `{loss_fn_name}` to "mean", '
                f'i.e. `{loss_fn_name}.reduction = "mean"`. Note that if '
                "`sample_wise_grads_per_batch` is True, the implementation "
                "assumes the reduction is either a sum or mean reduction."
            )
            reduction_type = "sum"
        else:
            warnings.warn(
                f'Since `{loss_fn_name}` has no "reduction" attribute, and '
                "`sample_wise_grads_per_batch` is False, the implementation "
                f'assumes that `{loss_fn_name}` is a "per-example" loss function (see '
                f"documentation for `{loss_fn_name}` for details).  Please ensure "
                "that this is the case."
            )

    return reduction_type


def _set_active_parameters(model: Module, layers: List[str]) -> List[Module]:
    """
    sets relevant parameters, as indicated by `layers`, to have `requires_grad=True`,
    and returns relevant modules.
    """
    assert isinstance(layers, List), "`layers` should be a list!"
    assert len(layers) > 0, "`layers` cannot be empty!"
    assert isinstance(layers[0], str), "`layers` should contain str layer names."
    layer_modules = [_get_module_from_name(model, layer) for layer in layers]
    for layer, layer_module in zip(layers, layer_modules):
        for name, param in layer_module.named_parameters():
            if not param.requires_grad:
                warnings.warn(
                    "Setting required grads for layer: {}, name: {}".format(
                        ".".join(layer), name
                    )
                )
                param.requires_grad = True
    return layer_modules


def _get_module_from_name(model: Module, layer_name: str) -> Any:
    r"""
    Returns the module (layer) object, given its (string) name
    in the model.

    Args:
            name (str): Module or nested modules name string in self.model

    Returns:
            The module (layer) in self.model.
    """

    return reduce(getattr, layer_name.split("."), model)


def _eig_helper(H: Tensor):
    """
    wrapper around `torch.linalg.eig` that sorts eigenvalues / eigenvectors by
    ascending eigenvalues, like `torch.linalg.eigh`, and returns the real component
    (since `H` is never complex, there should never be a complex component. however,
    `torch.linalg.eig` always returns a complex tensor, which in this case would
    actually have no complex component)
    """
    ls, vs = torch.linalg.eig(H)
    ls, vs = ls.real, vs.real
    ls_argsort = torch.argsort(ls)
    vs = vs[:, ls_argsort]
    ls = ls[ls_argsort]
    return ls, vs


def _top_eigen(
    H: Tensor, k: Optional[int], hessian_reg: float, hessian_inverse_tol: float
) -> Tuple[Tensor, Tensor]:
    """
    This is a wrapper around `torch.linalg.eig` that performs some pre /
    post-processing to make it suitable for computing the low-rank
    "square root" of a matrix, i.e. given square matrix H, find tall and
    skinny L such that LL' approximates H. This function returns eigenvectors (as the
    columns of a matrix Q) and corresponding eigenvectors (as diagonal entries in
    a matrix V), and we can then let L=QV^{1/2}Q'.  However, doing so requires the
    eigenvalues in V to be positive.  Thus, this function does pre-processing (adds
    an entry to the diagonal of H) and post-processing (returns only the top-k
    eigenvectors / eigenvalues where the eigenvalues are above a positive tolerance)
    to encourage and guarantee, respectively, that the returned eigenvalues be
    positive.  The pre-processing shifts the eigenvalues up by a constant, and the
    post-processing effectively replaces H with the most similar matrix (in terms of
    Frobenius norm) whose eigenvalues are above the tolerance, see
    https://nhigham.com/2021/01/26/what-is-the-nearest-positive-semidefinite-matrix/.

    Args:
        H (Tensor): a 2D square Tensor for which the top eigenvectors / eigenvalues
                will be computed.
        k (int): how many eigenvectors / eigenvalues to return (before dropping pairs
                whose eigenvalue is below the tolerance).
        hessian_reg (float): We add an entry to the diagonal of `H` to encourage it to
                be positive definite. This is that entry.
        hessian_inverse_tol (float): To compute the "square root" of `H` using the top
                eigenvectors / eigenvalues, the eigenvalues should be positive, and
                furthermore if above a tolerance, the inversion will be more
                numerically stable. Therefore, we only return eigenvectors /
                eigenvalues where the eigenvalue is above a tolerance. This argument
                specifies that tolerance.

    Returns:
        (eigenvalues, eigenvectors) (tuple of tensors): Mimicking the output of
                `torch.linalg.eigh`, `eigenvalues` is a 1D tensor of the top-k
                eigenvalues of the regularized `H` that are additionally above
                `hessian_inverse_tol`, and `eigenvectors` is a 2D tensor whose columns
                contain the corresponding eigenvectors. The eigenvalues are in
                ascending order.
    """
    # add regularization to hopefully make H positive definite
    H = H + (torch.eye(len(H)).to(device=H.device) * hessian_reg)

    # find eigvectors / eigvals of H
    # ls are eigenvalues, in ascending order
    # columns of vs are corresponding eigenvectors
    ls, vs = _eig_helper(H)

    # despite adding regularization to the hessian, it may still not be positive
    # definite. we can get rid of negative eigenvalues, but for numerical stability
    # can get rid of eigenvalues below a tolerance
    keep = ls > hessian_inverse_tol

    ls = ls[keep]
    vs = vs[:, keep]

    # only keep the top `k` eigvals / eigvectors
    if not (k is None):
        ls = ls[-k:]
        vs = vs[:, -k:]

    # `torch.linalg.eig` is not deterministic in that you can multiply an eigenvector
    # by -1, and it is still an eigenvector. to make eigenvectors deterministic,
    # we multiply an eigenvector according to some rule that flips if you multiply
    # the eigenvector by -1. in this case, that rule is whether the sum of the
    # entries of the eigenvector are > 0
    rule = torch.sum(vs, dim=0) > 0  # entries are 0/1
    rule_multiplier = (2 * rule) - 1  # entries are -1/1
    vs = vs * rule_multiplier.unsqueeze(0)

    return ls, vs


class _DatasetFromList(Dataset):
    def __init__(self, _l: List[Any]) -> None:
        self._l = _l

    def __getitem__(self, i: int) -> Any:
        return self._l[i]

    def __len__(self) -> int:
        return len(self._l)


def _format_inputs_dataset(inputs_dataset: Union[Tuple[Any, ...], DataLoader]):
    # if `inputs_dataset` is not a `DataLoader`, turn it into one.
    # `_DatasetFromList` turns a list into a `Dataset` where `__getitem__`
    # returns an element in the list, and using it to construct a `DataLoader`
    # with `batch_size=None` gives a `DataLoader` that yields a single batch.
    if not isinstance(inputs_dataset, DataLoader):
        inputs_dataset = DataLoader(
            _DatasetFromList([inputs_dataset]), shuffle=False, batch_size=None
        )
    return inputs_dataset


def _progress_bar_constructor(
    inst: EmbedderBase,
    inputs_dataset: DataLoader,
    quantities_name: str,
    dataset_name: str = "inputs_dataset",
):
    # Try to determine length of progress bar if possible, with a default
    # of `None`.
    inputs_dataset_len = None
    try:
        inputs_dataset_len = len(inputs_dataset)
    except TypeError:
        warnings.warn(
            f"Unable to determine the number of batches in "
            f"`{dataset_name}`. Therefore, if showing the progress "
            f"of the computation of {quantities_name}, "
            "only the number of batches processed can be "
            "displayed, and not the percentage completion of the computation, "
            "nor any time estimates."
        )

    return progress(
        inputs_dataset,
        desc=(
            f"Using {inst.get_name()} to compute {quantities_name}. " "Processing batch"
        ),
        total=inputs_dataset_len,
    )


def _compute_jacobian_sample_wise_grads_per_batch(
    inst: EmbedderBase,
    inputs: Tuple[Any, ...],
    targets: Optional[Tensor] = None,
    loss_fn: Optional[Union[Module, Callable]] = None,
    reduction_type: Optional[str] = "none",
) -> Tuple[Tensor, ...]:
    if inst.sample_wise_grads_per_batch:
        return _compute_jacobian_wrt_params_with_sample_wise_trick(
            inst.model,
            inputs,
            targets,
            loss_fn,
            reduction_type,
            inst.layer_modules,
        )
    return _compute_jacobian_wrt_params(
        inst.model,
        inputs,
        targets,
        loss_fn,
        inst.layer_modules,
    )


def _params_to_names(params: Iterable[nn.Parameter], model: nn.Module) -> List[str]:
    """
    Given an iterable of parameters, `params` of a model, `model`, returns the names of
    the parameters from the perspective of `model`. This is useful if, given
    parameters for which we do not know the name, want to pass them as a dict
    to a function of those parameters, i.e. `torch.nn.utils._stateless`.
    """
    param_id_to_name = {
        id(param): param_name for (param_name, param) in model.named_parameters()
    }
    return [param_id_to_name[id(param)] for param in params]


def _compute_batch_loss_influence_function_base(
    loss_fn: Optional[Union[Module, Callable]],
    input: Any,
    target: Any,
    reduction_type: str,
):
    """
    In implementations of `EmbedderBase`, we need to compute the total loss
    for a batch given `loss_fn`, whose reduction can either be 'none', 'sum', or
    'mean', and whose output requires different scaling based on the reduction. This
    helper houses that common logic, and returns the total loss for a batch given the
    predictions (`inputs`) and labels (`targets`) for it. We compute the total loss
    in order to compute the Hessian.
    """
    if loss_fn is not None:
        _loss = loss_fn(input, target)
    else:
        # following convention of `_compute_jacobian_wrt_params`, is no loss function is
        # provided, the quantity backpropped is the output of the forward function.
        assert reduction_type == "none"
        _loss = input

    if reduction_type == "none":
        # if loss_fn is a "reduction='none'" loss function, need to sum
        # up the per-example losses.
        return torch.sum(_loss)
    elif reduction_type == "mean":
        # in this case, we want the total loss for the batch, and should
        # multiply the mean loss for the batch by the batch size. however,
        # we can only infer the batch size if `_output` is a Tensor, and
        # we assume the 0-th dimension to be the batch dimension.
        if isinstance(input, Tensor):
            multiplier = input.shape[0]
        else:
            multiplier = 1
            msg = (
                "`loss_fn` was inferred to behave as a `reduction='mean'` "
                "loss function. however, the batch size of batches could not "
                "be inferred. therefore, the total loss of a batch, which is "
                "needed to compute the Hessian, is approximated as the output "
                "of `loss_fn` for the batch. if this approximation is not "
                "accurate, please change `loss_fn` to behave as a "
                "`reduction='sum'` loss function, or a `reduction='none'` "
                "and set `sample_grads_per_batch` to false."
            )
            warnings.warn(msg)
        return _loss * multiplier
    elif reduction_type == "sum":
        return _loss
    else:
        # currently, only support `reduction_type` to be
        # 'none', 'sum', or 'mean' for
        # `InfluenceFunctionBase` implementations
        raise Exception


class NotFitException(Exception):
    pass


def _parameter_to(params: Tuple[Tensor, ...], **to_kwargs) -> Tuple[Tensor, ...]:
    """
    applies the `to` method to all tensors in a tuple of tensors
    """
    return tuple(param.to(**to_kwargs) for param in params)


def _parameter_multiply(params: Tuple[Tensor, ...], c: Tensor) -> Tuple[Tensor, ...]:
    """
    multiplies all tensors in a tuple of tensors by a given scalar
    """
    return tuple(param * c for param in params)


def _parameter_dot(
    params_1: Tuple[Tensor, ...], params_2: Tuple[Tensor, ...]
) -> Tensor:
    """
    returns the dot-product of 2 tensors, represented as tuple of tensors.
    """
    return torch.Tensor(
        sum(
            torch.sum(param_1 * param_2)
            for (param_1, param_2) in zip(params_1, params_2)
        )
    )


def _parameter_add(
    params_1: Tuple[Tensor, ...], params_2: Tuple[Tensor, ...]
) -> Tuple[Tensor, ...]:
    """
    returns the sum of 2 tensors, represented as tuple of tensors.
    """
    return tuple(param_1 + param_2 for (param_1, param_2) in zip(params_1, params_2))


def _parameter_linear_combination(
    paramss: List[Tuple[Tensor, ...]], cs: Tensor
) -> Tuple[Tensor, ...]:
    """
    scales each parameter (tensor of tuples) in a list by the corresponding scalar in a
    1D tensor of the same length, and sums up the scaled parameters
    """
    assert len(cs.shape) == 1
    result = _parameter_multiply(paramss[0], cs[0])
    for params, c in zip(paramss[1:], cs[1:]):
        result = _parameter_add(result, _parameter_multiply(params, c))
    return result


def _set_attr(obj, names, val):
    if len(names) == 1:
        setattr(obj, names[0], val)
    else:
        _set_attr(getattr(obj, names[0]), names[1:], val)


def _del_attr(obj, names):
    if len(names) == 1:
        delattr(obj, names[0])
    else:
        _del_attr(getattr(obj, names[0]), names[1:])


def _model_make_functional(model, param_names, params):
    params = tuple([param.detach().requires_grad_() for param in params])

    for param_name in param_names:
        _del_attr(model, param_name.split("."))

    return params


def _model_reinsert_params(model, param_names, params, register=False):
    for param_name, param in zip(param_names, params):
        _set_attr(
            model,
            param_name.split("."),
            torch.nn.Parameter(param) if register else param,
        )


def _custom_functional_call(model, d, *features):
    param_names, params = zip(*list(d.items()))
    _params = _model_make_functional(model, param_names, params)
    _model_reinsert_params(model, param_names, params)
    out = model(*features)
    _model_reinsert_params(model, param_names, _params, register=True)
    return out


def _functional_call(model, d, features):
    """
    Makes a call to `model.forward`, which is treated as a function of the parameters
    in `d`, a dict from parameter name to parameter, instead of as a function of
    `features`, the argument that is unpacked to `model.forward` (i.e.
    `model.forward(*features)`).  Depending on what version of PyTorch is available,
    we either use our own implementation, or directly use `torch.nn.utils.stateless`
    or `torch.func.functional_call`.  Put another way, this function mimics the latter
    two implementations, using our own when the PyTorch version is too old.
    """
    import torch

    version = torch.__version__
    if version < "1.12":
        return _custom_functional_call(model, d, features)
    elif version >= "1.12" and version < "2.0":
        import torch.nn.utils.stateless

        return torch.nn.utils.stateless.functional_call(model, d, features)
    else:
        import torch.func

        return torch.func.functional_call(model, d, features)
