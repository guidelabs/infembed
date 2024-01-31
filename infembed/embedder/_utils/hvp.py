from abc import ABC, abstractmethod
from functools import reduce
from typing import Any, Callable, Optional, Tuple, Union
from infembed.embedder._utils.common import (
    _compute_batch_loss_influence_function_base,
    _functional_call,
    _parameter_add,
    _params_to_names,
)
from infembed.embedder._utils.gradient import _extract_parameters_from_layers
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import Tensor
import torch


class HVP(ABC):
    """
    An abstract class to define hessian-vector computation.
    """

    @abstractmethod
    def setup(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        layer_modules: Any,
        reduction_type: str,
        loss_fn: Optional[Union[nn.Module, Callable]] = None,
    ):
        """
        Any pre-processing to do before being able to do HVP computation.

        Args:
            module (nn.Module): model and parameters used to do HVP
            dataloader (DataLoader): HVP depends on the data.
            layer_modules: (list of nn.Module): the order of parameters in the vector
                    and Hessian are the order returned by
                    `_extract_parameters_from_layers(layer_modules)`
        """

    @abstractmethod
    def __call__(self, v: Tuple[Tensor, ...]):
        """
        Does the actual HVP computation

        Args:
            v (tuple of tensor): vector used in HVP computation.
        """


class AutogradHVP(HVP):
    """
    Implementation of `HVP` which uses `torch.autograd.functional.hvp`
    """

    def __init__(self, show_progress: bool, hvp_mode: str):
        """
        Args:
            show_progress (bool): whether to show how many batches have been processed
                    in the HVP computation
        """
        self.show_progress = show_progress
        self.hvp_mode = hvp_mode
        self.HVP: Optional[Callable] = None

    def setup(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        layer_modules: Any,
        reduction_type: str,
        loss_fn: Optional[Union[nn.Module, Callable]] = None,
    ):
        # first figure out names of params that require gradients. this is need to
        # create that function, as it replaces params based on their names
        params = tuple(
            model.parameters()
            if layer_modules is None
            else _extract_parameters_from_layers(layer_modules)
        )
        # the same position in `params` and `param_names` correspond to each other
        param_names = _params_to_names(params, model)

        # get factory that given a batch, returns a function that given params as
        # tuple of tensors, returns loss over the batch
        def tensor_tuple_loss_given_batch(batch):
            def tensor_tuple_loss(*params):
                # `params` is a tuple of tensors, and assumed to be order specified by
                # `param_names`
                features, labels = tuple(batch[0:-1]), batch[-1]

                _output = _functional_call(
                    model, dict(zip(param_names, params)), features
                )

                # compute the total loss for the batch, adjusting the output of
                # `self.loss_fn` based on `self.reduction_type`
                return _compute_batch_loss_influence_function_base(
                    loss_fn, _output, labels, reduction_type
                )

            return tensor_tuple_loss

        # define function that given batch and vector, returns HVP of loss using the
        # batch and vector
        def batch_HVP(batch, v):
            tensor_tuple_loss = tensor_tuple_loss_given_batch(batch)
            if self.hvp_mode == "vhp":
                return torch.autograd.functional.vhp(tensor_tuple_loss, params, v=v)[1]
            elif self.hvp_mode == "hvp":
                return torch.autograd.functional.hvp(tensor_tuple_loss, params, v=v)[1]
            elif self.hvp_mode == "manual_hvp":
                from torch.func import jvp, grad
                return jvp(
                    grad(tensor_tuple_loss, argnums=tuple(range(len(params)))),
                    params,
                    v,
                )[1]
            elif self.hvp_mode == "manual_revrev":
                from torch.func import vjp, grad
                _, vjp_fn = vjp(grad(tensor_tuple_loss, argnums=tuple(range(len(params)))), *params)
                return vjp_fn(v)
            else:
                raise Exception('`hvp_mode` not recognized')

        # define function that returns HVP of loss over `dataloader`, given a
        # specified vector
        def _HVP(v):
            _dataloader = dataloader
            if False and self.show_progress:
                _dataloader = tqdm(
                    dataloader, desc="processing `hessian_dataset` batch"
                )

            # the HVP of loss using the entire `dataloader` is the sum of the
            # per-batch HVP's
            return _dataset_fn(_dataloader, batch_HVP, _parameter_add, v)

        self.HVP = _HVP

    def __call__(self, v: Tuple[Tensor, ...]):
        if self.HVP is None:
            raise Exception("`setup` has not been called yet")
        return self.HVP(v)


def _dataset_fn(dataloader, batch_fn, reduce_fn, *batch_fn_args, **batch_fn_kwargs):
    """
    Applies `batch_fn` to each batch in `dataloader`, reducing the results using
    `reduce_fn`.  This is useful for computing Hessians and Hessian-vector
    products over an entire dataloader, and is used by both `NaiveEmbedder`
    and `ArnoldiEmbedder`.
    """
    _dataloader = iter(dataloader)

    def _reduce_fn(_result, _batch):
        return reduce_fn(_result, batch_fn(_batch, *batch_fn_args, **batch_fn_kwargs))

    result = batch_fn(next(_dataloader), *batch_fn_args, **batch_fn_kwargs)
    return reduce(_reduce_fn, _dataloader, result)
