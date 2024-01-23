from typing import Any, List, Optional, Union, Callable, Tuple, overload
import torch
from torch.nn import Module
from torch import Tensor
import warnings
from torch import device, Tensor


def _register_backward_hook(
    module: Module, hook: Callable, attr_obj: Any
) -> List[torch.utils.hooks.RemovableHandle]:
    grad_out: Dict[device, Tensor] = {}

    def forward_hook(
        module: Module,
        inp: Union[Tensor, Tuple[Tensor, ...]],
        out: Union[Tensor, Tuple[Tensor, ...]],
    ) -> None:
        nonlocal grad_out
        grad_out = {}

        def output_tensor_hook(output_grad: Tensor) -> None:
            grad_out[output_grad.device] = output_grad

        if isinstance(out, tuple):
            assert (
                len(out) == 1
            ), "Backward hooks not supported for module with >1 output"
            out[0].register_hook(output_tensor_hook)
        else:
            out.register_hook(output_tensor_hook)

    def pre_hook(module, inp):
        def input_tensor_hook(input_grad: Tensor):
            if len(grad_out) == 0:
                return
            hook_out = hook(module, input_grad, grad_out[input_grad.device])

            if hook_out is not None:
                return hook_out[0] if isinstance(hook_out, tuple) else hook_out

        if isinstance(inp, tuple):
            assert (
                len(inp) == 1
            ), "Backward hooks not supported for module with >1 input"
            inp[0].register_hook(input_tensor_hook)
            return inp[0].clone()
        else:
            inp.register_hook(input_tensor_hook)
            return inp.clone()

    return [
        module.register_forward_pre_hook(pre_hook),
        module.register_forward_hook(forward_hook),
    ]


@overload
def _format_tensor_into_tuples(inputs: None) -> None:
    ...


@overload
def _format_tensor_into_tuples(
    inputs: Union[Tensor, Tuple[Tensor, ...]]
) -> Tuple[Tensor, ...]:
    ...


def _format_tensor_into_tuples(
    inputs: Union[None, Tensor, Tuple[Tensor, ...]]
) -> Union[None, Tuple[Tensor, ...]]:
    if inputs is None:
        return None
    if not isinstance(inputs, tuple):
        assert isinstance(inputs, torch.Tensor), (
            "`inputs` must be a torch.Tensor or a tuple[torch.Tensor] "
            f"but found: {type(inputs)}"
        )
        inputs = (inputs,)
    return inputs