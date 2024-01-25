from typing import Callable, List, Optional, Sequence, Tuple, Union, Any, cast
import warnings
from infembed.embedder._utils.sample_gradient import SampleGradientWrapper
from torch import Tensor
from torch.nn import Module
import torch


def apply_gradient_requirements(
    inputs: Tuple[Tensor, ...], warn: bool = True
) -> List[bool]:
    """
    Iterates through tuple on input tensors and sets requires_grad to be true on
    each Tensor, and ensures all grads are set to zero. To ensure that the input
    is returned to its initial state, a list of flags representing whether or not
     a tensor originally required grad is returned.
    """
    assert isinstance(
        inputs, tuple
    ), "Inputs should be wrapped in a tuple prior to preparing for gradients"
    grad_required = []
    for index, input in enumerate(inputs):
        assert isinstance(input, torch.Tensor), "Given input is not a torch.Tensor"
        grad_required.append(input.requires_grad)
        inputs_dtype = input.dtype
        # Note: torch 1.2 doesn't support is_complex for dtype that's why we check
        # on the existance of is_complex method.
        if not inputs_dtype.is_floating_point and not (
            hasattr(inputs_dtype, "is_complex") and inputs_dtype.is_complex
        ):
            if warn:
                warnings.warn(
                    """Input Tensor %d has a dtype of %s.
                    Gradients cannot be activated
                    for these data types."""
                    % (index, str(inputs_dtype))
                )
        elif not input.requires_grad:
            if warn:
                # TODO: figure out why get this warning
                warnings.warn(
                    "Input Tensor %d did not already require gradients, "
                    "required_grads has been set automatically." % index
                )
            input.requires_grad_()
    return grad_required


def _compute_jacobian_wrt_params_with_sample_wise_trick(
    model: Module,
    inputs: Tuple[Any, ...],
    labels: Optional[Tensor] = None,
    loss_fn: Optional[Union[Module, Callable]] = None,
    reduction_type: Optional[str] = "sum",
    layer_modules: List[Module] = None,
) -> Tuple[Any, ...]:
    r"""
    Computes the Jacobian of a batch of test examples given a model, and optional
    loss function and target labels. This method uses sample-wise gradients per
    batch trick to fully vectorize the Jacobian calculation. Currently, only
    linear and conv2d layers are supported.

    User must `add_hooks(model)` before calling this function.

    Args:
        model (torch.nn.Module): The trainable model providing the forward pass
        inputs (tuple[Any, ...]): The minibatch for which the forward pass is computed.
                It is unpacked before passing to `model`, so it must be a tuple.  The
                individual elements of `inputs` can be anything.
        labels (Tensor, optional): Labels for input if computing a loss function.
        loss_fn (torch.nn.Module or Callable, optional): The loss function. If a library
                defined loss function is provided, it would be expected to be a
                torch.nn.Module. If a custom loss is provided, it can be either type,
                but must behave as a library loss function would if `reduction='sum'` or
                `reduction='mean'`.
        reduction_type (str, optional): The type of reduction applied. If a loss_fn is
                passed, this should match `loss_fn.reduction`. Else if gradients are
                being computed on direct model outputs (scores), then 'sum' should be
                used.
                Defaults to 'sum'.
        layer_modules (torch.nn.Module, optional): A list of PyTorch modules w.r.t.
                 which jacobian gradients are computed.

    Returns:
        grads (tuple[Tensor, ...]): Returns the Jacobian for the minibatch as a
                tuple of gradients corresponding to the tuple of trainable parameters
                returned by `model.parameters()`. Each object grads[i] references to the
                gradients for the parameters in the i-th trainable layer of the model.
                Each grads[i] object is a tensor with the gradients for the `inputs`
                batch. For example, grads[i][j] would reference the gradients for the
                parameters of the i-th layer, for the j-th member of the minibatch.
    """
    with torch.autograd.set_grad_enabled(True):
        inputs = tuple(inp.clone() for inp in inputs)
        apply_gradient_requirements(inputs, warn=False)
        sample_grad_wrapper = SampleGradientWrapper(model, layer_modules)
        try:
            sample_grad_wrapper.add_hooks()

            out = model(*inputs)
            assert (
                out.dim() != 0
            ), "Please ensure model output has at least one dimension."

            if labels is not None and loss_fn is not None:
                loss = loss_fn(out, labels)
                # TODO: allow loss_fn to be Callable
                if (isinstance(loss_fn, Module) or callable(loss_fn)) and hasattr(
                    loss_fn, "reduction"
                ):
                    reduction = loss_fn.reduction  # type: ignore
                    msg0 = (
                        "Please ensure that loss_fn.reduction is set to `sum` or `mean`"
                    )

                    assert reduction != "none", msg0
                    msg1 = (
                        f"loss_fn.reduction ({reduction}) does not match"
                        f"reduction type ({reduction_type}). Please ensure they are"
                        " matching."
                    )
                    assert reduction == reduction_type, msg1
                msg2 = (
                    "Please ensure custom loss function is applying either a "
                    "sum or mean reduction."
                )
                assert out.shape != loss.shape, msg2

                if reduction_type != "sum" and reduction_type != "mean":
                    raise ValueError(
                        f"{reduction_type} is not a valid value for reduction_type. "
                        "Must be either 'sum' or 'mean'."
                    )
                out = loss

            sample_grad_wrapper.compute_param_sample_gradients(
                out, loss_mode=reduction_type
            )
            if layer_modules is not None:
                layer_parameters = _extract_parameters_from_layers(layer_modules)
            grads = tuple(
                param.sample_grad  # type: ignore
                for param in (
                    model.parameters() if layer_modules is None else layer_parameters
                )
                if hasattr(param, "sample_grad")
            )
        finally:
            sample_grad_wrapper.remove_hooks()

        return grads
    

def _extract_parameters_from_layers(layer_modules):
    layer_parameters = []
    if layer_modules is not None:
        layer_parameters = [
            parameter
            for layer_module in layer_modules
            for parameter in layer_module.parameters()
        ]
        assert (
            len(layer_parameters) > 0
        ), "No parameters are available for modules for provided input `layers`"
    return layer_parameters


def _compute_jacobian_wrt_params(
    model: Module,
    inputs: Tuple[Any, ...],
    labels: Optional[Tensor] = None,
    loss_fn: Optional[Union[Module, Callable]] = None,
    layer_modules: List[Module] = None,
) -> Tuple[Tensor, ...]:
    r"""
    Computes the Jacobian of a batch of test examples given a model, and optional
    loss function and target labels. This method is equivalent to calculating the
    gradient for every individual example in the minibatch.

    Args:
        model (torch.nn.Module): The trainable model providing the forward pass
        inputs (tuple[Any, ...]): The minibatch for which the forward pass is computed.
                It is unpacked before passing to `model`, so it must be a tuple.  The
                individual elements of `inputs` can be anything.
        labels (Tensor, optional): Labels for input if computing a loss function.
        loss_fn (torch.nn.Module or Callable, optional): The loss function. If a library
                defined loss function is provided, it would be expected to be a
                torch.nn.Module. If a custom loss is provided, it can be either type,
                but must behave as a library loss function would if `reduction='none'`.
        layer_modules (List[torch.nn.Module], optional): A list of PyTorch modules
                 w.r.t. which jacobian gradients are computed.
    Returns:
        grads (tuple[Tensor, ...]): Returns the Jacobian for the minibatch as a
                tuple of gradients corresponding to the tuple of trainable parameters
                returned by `model.parameters()`. Each object grads[i] references to the
                gradients for the parameters in the i-th trainable layer of the model.
                Each grads[i] object is a tensor with the gradients for the `inputs`
                batch. For example, grads[i][j] would reference the gradients for the
                parameters of the i-th layer, for the j-th member of the minibatch.
    """
    with torch.autograd.set_grad_enabled(True):
        out = model(*inputs)
        assert out.dim() != 0, "Please ensure model output has at least one dimension."

        if labels is not None and loss_fn is not None:
            loss = loss_fn(out, labels)
            if hasattr(loss_fn, "reduction"):
                msg0 = "Please ensure loss_fn.reduction is set to `none`"
                assert loss_fn.reduction == "none", msg0  # type: ignore
            else:
                msg1 = (
                    "Loss function is applying a reduction. Please ensure "
                    f"Output shape: {out.shape} and Loss shape: {loss.shape} "
                    "are matching."
                )
                assert loss.dim() != 0, msg1
                assert out.shape[0] == loss.shape[0], msg1
            out = loss

        if layer_modules is not None:
            layer_parameters = _extract_parameters_from_layers(layer_modules)
        grads_list = [
            torch.autograd.grad(
                outputs=out[i],
                inputs=cast(
                    Union[Tensor, Sequence[Tensor]],
                    model.parameters() if layer_modules is None else layer_parameters,
                ),
                grad_outputs=torch.ones_like(out[i]),
                retain_graph=True,
            )
            for i in range(out.shape[0])
        ]
        grads = tuple([torch.stack(x) for x in zip(*grads_list)])

        return tuple(grads)