from typing import Optional, Tuple, List, Iterable
from infembed.embedder._utils.gradient import _extract_parameters_from_layers
from torch import Tensor
import torch
import torch.nn as nn
from tqdm import tqdm
from infembed.embedder._utils.common import (
    _compute_batch_loss_influence_function_base,
    _params_to_names,
)
from abc import abstractmethod
import torch.nn.functional as F


class _LayerCaptureInput:
    """
    simple hook to store the input to a layer.  if layer is called multiple times, only
    stores the input from the latest call.
    """

    def __init__(self, device="cpu") -> None:
        self.input: Optional[Tuple[Tensor, ...]] = None
        self.device = device

    def __call__(
        self, module: nn.Module, input: Tuple[Tensor, ...], output: Tuple[Tensor, ...]
    ):
        if self.device is not None:
            self.input = tuple(_input.detach().to(device=self.device) for _input in input)
        else:
            self.input = tuple(
                _input.detach() for _input in input
            )


class _LayerCaptureOutputGradient:
    """
    simple hook to store the output gradient to a layer.  if layer is called multiple
    times, only stores the output gradient from the latest call.
    """

    def __init__(self, device="cpu") -> None:
        self.output_gradient: Optional[Tuple[Tensor, ...]] = None
        self.device = device

    def __call__(
        self,
        module: nn.Module,
        input_gradient: Tuple[Tensor, ...],
        output_gradient: Tuple[Tensor, ...],
    ):
        if self.device is not None:
            self.output_gradient = tuple(
                _output_gradient.detach().to(device=self.device)
                for _output_gradient in output_gradient
            )
        else:
            self.output_gradient = tuple(
                _output_gradient.detach() for _output_gradient in output_gradient
            )


class _RunningAverage:
    """
    for computing running average over batches
    """

    def setup(self):
        self.num_samples = 0
        self.val = None
        return self

    def update(self, num_samples: int, val: Tensor):
        """
        `val` is an average and `num_samples` is the number of samples `val` represents
        """
        if self.val is None:
            self.val = val
            self.num_samples = num_samples
        else:
            if False:
                if True:
                    a = self.val.clone()
                    a += val * (num_samples / self.num_samples)
                    a /= (num_samples + self.num_samples) / self.num_samples

                if True:
                    b = self.val.clone()
                    b = ((val * num_samples) + (self.val * self.num_samples)) / (
                        num_samples + self.num_samples
                    )
                assert (a - b).abs().sum() < 1e-2
            self.val += val * (num_samples / self.num_samples)
            self.val /= (num_samples + self.num_samples) / self.num_samples
            self.num_samples = num_samples + self.num_samples

    def results(self):
        return self.val


class _LayerInputOutputGradientAccumulator:
    """
    for use in `_accumulate_with_layer_inputs_and_output_gradients`
    """

    @abstractmethod
    def setup(self):
        pass

    @abstractmethod
    def update(self, layer_input: Tensor, layer_output_gradient: Tensor, batch: Tuple):
        pass

    @abstractmethod
    def results(self):
        pass


class _LayerHessianFlattenedAcumulator(_LayerInputOutputGradientAccumulator):
    """
    estimates and accumulates Hessian for a layer, without assuming the layer input
    and output gradient are independent.
    """

    def __init__(self, layer: nn.Module):
        self.layer = layer

    def setup(self):
        self.accumulator = _RunningAverage().setup()

    def update(
        self, layer_input: Tensor, layer_output_gradient: Tensor, batch: Tuple
    ) -> None:
        batch_size = len(layer_input)

        # put into consistent form
        layer_input = Reshaper.layer_input_to_two_d(self.layer, layer_input)
        layer_output_gradient = Reshaper.layer_output_gradient_to_two_d(
            self.layer, layer_output_gradient
        )

        layer_batch_hessian_flattened_average = None

        for a, s in zip(layer_input, layer_output_gradient):
            # sum over all non-batch dimensions
            layer_sample_hessian_flattened = 0
            for _a, _s in zip(a, s):
                layer_sample_hessian_flattened += torch.kron(
                    torch.outer(_a, _a), torch.outer(_s, _s)
                ).detach()

            if layer_batch_hessian_flattened_average is None:
                layer_batch_hessian_flattened_average = layer_sample_hessian_flattened
            else:
                layer_batch_hessian_flattened_average += layer_sample_hessian_flattened
        layer_batch_hessian_flattened_average /= batch_size
        self.accumulator.update(batch_size, layer_batch_hessian_flattened_average)


def _extract_patches(x, kernel_size, stride, padding):
    """
    copied from https://github.com/alecwangcq/KFAC-Pytorch/blob/master/utils/kfac_utils.py#L13
    :param x: The input feature maps.  (batch_size, in_c, h, w)
    :param kernel_size: the kernel size of the conv filter (tuple of two elements)
    :param stride: the stride of conv operation  (tuple of two elements)
    :param padding: number of paddings. be a tuple of two elements
    :return: (batch_size, out_h, out_w, in_c*kh*kw)
    """
    if padding[0] + padding[1] > 0:
        x = F.pad(
            x, (padding[1], padding[1], padding[0], padding[0])
        ).data  # Actually check dims
    x = x.unfold(2, kernel_size[0], stride[0])
    x = x.unfold(3, kernel_size[1], stride[1])
    x = x.transpose_(1, 2).transpose_(2, 3).contiguous()
    x = x.view(x.size(0), x.size(1), x.size(2), x.size(3) * x.size(4) * x.size(5))
    return x


class SplitTwoD:
    """
    takes 2D representation of layer input or output gradient, i.e. what is returned by
    `Reshaper.layer_output_gradient_to_two_d`, and splits it along the last (i.e.
    input feature or output feature) dimension.
    """

    @abstractmethod
    def __call__(self, two_d: Tensor) -> Iterable[Tensor]:
        raise NotImplementedError

    @abstractmethod
    def num_splits(self, two_d):
        raise NotImplementedError


class DummySplitTwoD(SplitTwoD):
    """
    does not actually do any splitting.  use this if not doing any approximation of a
    single layer's hessian
    """

    def __call__(self, two_d) -> Iterable[Tensor]:
        yield two_d

    def num_splits(self, two_d):
        return 1


class BlockSplitTwoD(SplitTwoD):
    """
    splits into contiguous chunks
    """

    def __init__(self, num_blocks: int):
        self.num_blocks = num_blocks

    def _get_start_ends(self, two_d):
        assert len(two_d.shape) == 3
        size = two_d.shape[2]
        for i in range(self.num_blocks):
            start = int(i * size / self.num_blocks)
            end = int((i + 1) * size / self.num_blocks)
            if end > start:
                yield (start, end)

    def __call__(self, two_d) -> Iterable[Tensor]:
        for start, end in self._get_start_ends(two_d):
            yield two_d[:, :, start:end]

    def num_splits(self, two_d):
        return len([_ for _ in self._get_start_ends(two_d)])


def _get_A(layer: nn.Module, layer_input: Tensor):
    """
    method to get A for a batch, as 2D tensor, for both linear and conv2d layers
    needs to be consistent with `Reshaper`.  add bias features here; doing it during
    `_accumulate_with_layer_inputs_and_output_gradients` is too inflexible.
    """
    # layer_input = Reshaper.layer_input_to_two_d(layer, layer_input)
    batch_size = layer_input.shape[0]
    # outer product, summing over all dimensions except 0-th and last
    # this code can handle dim > 3, but it will only be used with dim = 3
    outer = torch.einsum("k...i,k...j->ij", layer_input, layer_input)
    # expectation is a per-example expectation
    return outer / batch_size


def _get_S(layer: nn.Module, layer_output_gradient: Tensor):
    """
    method to get S for a batch, as 2D tensor, for both linear and conv2d layers
    needs to be consistent with `Reshaper`
    """
    # layer_output_gradient = Reshaper.layer_output_gradient_to_two_d(
    #     layer, layer_output_gradient
    # )
    batch_size = len(layer_output_gradient)
    # this code can handle dim > 3, but it will only be used with dim = 3
    outer = torch.einsum(
        "k...i,k...j->ij", layer_output_gradient, layer_output_gradient
    )
    return outer / batch_size


class _LayerHessianFlattenedIndependentAcumulator(_LayerInputOutputGradientAccumulator):
    """
    estimates and accumulates Hessian for a layer, assuming the layer input
    and output gradient are independent.
    """

    def __init__(self, layer: nn.Module, split_two_d: Optional[SplitTwoD] = None):
        self.layer = layer
        if split_two_d is None:
            split_two_d = DummySplitTwoD()
        self.split_two_d = split_two_d

    def setup(self):
        self.layer_A_accumulators = None
        self.layer_S_accumulators = None
        # self.layer_A_accumulators = [_RunningAverage().setup() for _ in range(self.split_two_d.num_splits()))]
        # self.layer_S_accumulators = [_RunningAverage().setup() for _ in range(self.split_two_d.num_splits()))]

    def update(
        self, layer_input: Tensor, layer_output_gradient: Tensor, batch: Tuple
    ) -> None:
        batch_size = len(layer_input)
        layer_input = Reshaper.layer_input_to_two_d(self.layer, layer_input)
        layer_output_gradient = Reshaper.layer_output_gradient_to_two_d(
            self.layer, layer_output_gradient
        )

        if self.layer_A_accumulators is None:
            self.layer_A_accumulators = [
                _RunningAverage().setup()
                for _ in range(self.split_two_d.num_splits(layer_input))
            ]
            self.layer_S_accumulators = [
                _RunningAverage().setup()
                for _ in range(self.split_two_d.num_splits(layer_output_gradient))
            ]
            assert len(self.layer_A_accumulators) == len(self.layer_S_accumulators)
        for _accumulator, _layer_input in zip(
            self.layer_A_accumulators, self.split_two_d(layer_input)
        ):
            _accumulator.update(batch_size, _get_A(self.layer, _layer_input))
        for _accumulator, _layer_output_gradient in zip(
            self.layer_S_accumulators, self.split_two_d(layer_output_gradient)
        ):
            _accumulator.update(batch_size, _get_S(self.layer, _layer_output_gradient))


class _LayerCaptureAccumulator(_LayerInputOutputGradientAccumulator):
    """
    stores both quantities as a lists of tensors
    """

    def setup(self):
        self.layer_inputs = []
        self.layer_output_gradients = []

    def update(
        self, layer_input: Tensor, layer_output_gradient: Tensor, batch: Tuple
    ) -> None:
        self.layer_inputs.append(layer_input)
        self.layer_output_gradients.append(layer_output_gradient)


def _accumulate_with_layer_inputs_and_output_gradients(
    layer_accumulators: List[_LayerInputOutputGradientAccumulator],
    model,
    dataloader,
    layer_modules,
    reduction_type,
    loss_fn,
    show_progress,
    accumulate_device="cpu",
):
    """
    for each batch in `dataloader`, runs forward and backward pass to get the input and
    output gradient for each layer.  those are then given to each layer's accumulator
    to do some custom calculations, i.e. estimate the Hessian for the batch.
    """
    # add hooks to capture the input and output gradient of layers
    layer_input_hooks = [
        _LayerCaptureInput(device=accumulate_device) for _ in layer_modules
    ]
    layer_output_gradient_hooks = [
        _LayerCaptureOutputGradient(device=accumulate_device) for _ in layer_modules
    ]
    layer_input_hook_handles = [
        layer.register_forward_hook(hook)
        for (hook, layer) in zip(layer_input_hooks, layer_modules)
    ]
    layer_output_gradient_hook_handles = [
        layer.register_backward_hook(hook)
        for (hook, layer) in zip(layer_output_gradient_hooks, layer_modules)
    ]

    # accumulators require `setup`, i.e. clearing running averages
    for layer_accumulator in layer_accumulators:
        layer_accumulator.setup()

    # do forward and backward passes to get layer input and output gradient for each
    # batch
    _dataloader = dataloader
    if show_progress:
        _dataloader = tqdm(dataloader, desc="processing `hessian_dataset` batch")
    for batch in _dataloader:
        # forward pass
        features, labels = tuple(batch[0:-1]), batch[-1]
        _output = model(*features)

        # get inputs for each layer
        _layer_inputs = [
            layer_input_hook.input[0] for layer_input_hook in layer_input_hooks
        ]

        # backward pass
        loss = _compute_batch_loss_influence_function_base(
            loss_fn, _output, labels, reduction_type
        )
        loss.backward()

        # get output gradients for each layer
        _layer_output_gradients = [
            layer_output_gradient_hook.output_gradient[0]
            for layer_output_gradient_hook in layer_output_gradient_hooks
        ]

        # update accumulator based on results for the batch
        for _layer_input, _layer_output_gradient, layer_accumulator in zip(
            _layer_inputs, _layer_output_gradients, layer_accumulators
        ):
            layer_accumulator.update(_layer_input, _layer_output_gradient, batch)

    # remove hooks
    for hook_handle in layer_input_hook_handles:
        hook_handle.remove()
    for hook_handle in layer_output_gradient_hook_handles:
        hook_handle.remove()

    return layer_accumulators


class Reshaper:
    @staticmethod
    def layer_output_gradient_to_two_d(layer: nn.Module, layer_output_gradient: Tensor):
        """
        Does 2 things to the layer output gradient:

        1) reshaping: to manually compute gradients for parameters in a layer, need to
        reshape the layer input so that last dim is features, and previous dims
        correspond to examples and extra "locations" if applicable (i.e. locations in
        image or text).  in theory, these locations are the same between the layer
        input and output gradient, so this function makes the shapes of both quantities
        consistent with each other.

        3) to 2D: furthermore, processing those quantities often requires
        aggregating over the locations, and is agnostic to what each location
        represents, we also reshape all the locations into a single dimension.

        useful for `_get_A`, `_get_S`, `get_batch_embeddings`.
        """

        # 1) reshaping
        if isinstance(layer, nn.Linear) and len(layer_output_gradient.shape) == 2:
            # vanilla linear case - nothing to do
            pass

        elif isinstance(layer, nn.Conv2d):
            # conv case
            # `layer_output_gradient` has shape (batch size, number output, spatial height, spatial width)
            # first get presentation consistent with output of `_extract_patches`.  this
            # would have shape (batch size, spatial height, spatial width, number output).
            # thus, need to move dimension 1 to dimension 3
            layer_output_gradient = layer_output_gradient.transpose(1, 2).transpose(
                2, 3
            )

        elif isinstance(layer, nn.Linear) and len(layer_output_gradient.shape) == 3:
            # LLM case.
            # nothing to do, since linear layer gives consistency
            pass
        else:
            raise Exception(f"layer {layer} not supported")

        # 2) to 3D
        layer_output_gradient = layer_output_gradient.reshape(
            layer_output_gradient.shape[0], -1, layer_output_gradient.shape[-1]
        )

        return layer_output_gradient

    @staticmethod
    def layer_input_to_two_d(layer: nn.Module, layer_input: Tensor):
        """
        Does 3 things to the layer input:

        1) reshaping: to manually compute gradients for parameters in a layer, need to
        reshape the layer input so that last dim is features, and previous dims
        correspond to examples and extra "locations" if applicable (i.e. locations in
        image or text).  in theory, these locations are the same between the layer
        input and output gradient, so this function makes the shapes of both quantities
        consistent with each other.

        3) add bias: finally, the bias "feature" is also added.

        3) to 3D: furthermore, processing those quantities often requires
        aggregating over the locations, and is agnostic to what each location
        represents, we also reshape all the locations into a single dimension.

        useful for `_get_A`, `_get_S`, `get_batch_embeddings`.
        """

        # 1) reshaping
        if isinstance(layer, nn.Linear) and len(layer_input.shape) == 2:
            # vanilla linear case - nothing to do
            pass

        elif isinstance(layer, nn.Conv2d):
            # conv case
            # `layer_input` has shape (batch size, number channels, image height, image width)
            # the "features" are channels and pixels in the kernel, so that number of
            # features is number channels x kernel height x kernel width.  will first compute
            # these features for all examples in batch and all spatial locations in each
            # example, so that output has shape
            # (batch size, spatial height, spatial width, number of features)
            layer_input = _extract_patches(
                layer_input,
                layer.kernel_size,
                layer.stride,
                layer.padding,
            )

        elif isinstance(layer, nn.Linear) and len(layer_input.shape) == 3:
            # LLM case.  nothing to do, since linear layer gives consistency
            pass

        else:
            raise Exception(f"layer {layer} not supported")
        # import pdb
        # pdb.set_trace()
        # 2) add bias
        # do so for every "example" (meaning actual examples as well as image or text location)
        if layer.bias is not None:
            layer_input = torch.cat(
                [
                    layer_input,
                    # torch.ones((*layer_input.shape[:-1], 1)).to(layer_input.device),
                    torch.ones((*layer_input.shape[:-1], 1), device=layer_input.device),
                ],
                dim=-1,
            )

        # 3) to 3D
        layer_input = layer_input.reshape(
            layer_input.shape[0], -1, layer_input.shape[-1]
        )

        return layer_input

    @staticmethod
    def tot_to_one_d(layer: nn.Module, tot_params: Tuple[Tensor, ...]) -> Tensor:
        """
        reshape tuple of tensors to 1D, so can be multiplied with a 2D hessian.  Used
        in `KFACInfluenceFunction`, `KFACHVP`. 1D needs to be compatible with
        Kronecker Hessian approximation.
        """
        assert len(tot_params) in [1, 2]
        weight_index = 0
        weight = tot_params[weight_index]

        bias_index = None
        if len(tot_params) != 1:
            bias_index = 1
            bias = tot_params[bias_index]

        # first get weight parameter to 2D

        if isinstance(layer, nn.Linear):
            # linear or llm case
            # nothing to do, will reshape later in function
            pass
        elif isinstance(layer, nn.Conv2d):
            # conv case
            # first reshape weight to be 2D.  it is currently of shape
            # (number output, number channels, kernel height, kernel width).
            # looking at `layer_input_to_two_d`, need last 3 dimensions to be
            # consistent with how "features" are ordered in `_extract_patches`.
            # for now, assume that flattening those dimensions accomplishes this
            # consistency.  TODO: check this assumption.
            weight = weight.reshape(weight.shape[0], -1)
        else:
            raise Exception(f"layer {layer} not supported")

        # from 2D, concatenate the columns, following page 14 of
        # https://www.cs.toronto.edu/~rgrosse/courses/csc2541_2022/readings/L04_second_order.pdf
        one_d = weight.T.reshape(-1)

        # bias term is an additional input dimension, i.e. column, in the 2D weight
        # since concatenating columns above, just concatenate the bias parameter to the end
        if bias_index is not None:
            one_d = torch.cat([one_d, bias])

        return one_d

    @staticmethod
    def one_d_to_tot(
        layer: nn.Module, tot_params: Tuple[Tensor, ...], one_d_params: Tensor
    ) -> Tuple[Tensor, ...]:
        """
        reshape 1D tensor to tuple of tensors.  Used in `KFACHVP`.  needs the original
        tuple of tensors param to know how to reshape.
        """
        if isinstance(layer, nn.Linear):
            if len(tot_params) == 1:
                weight_index, bias_index = 0, None
            else:
                # TODO: explicitly check this
                weight_index, bias_index = 0, 1
            weight_shape = tot_params[weight_index].shape
            weight_length = weight_shape[0] * weight_shape[1]

            layer_hvp_weight = one_d_params[:weight_length].reshape(weight_shape)

            if bias_index is not None:
                layer_hvp_bias = one_d_params[weight_length:]
                return (layer_hvp_weight, layer_hvp_bias)
            else:
                return (layer_hvp_weight,)
        else:
            raise Exception(f"layer {layer} not supported")


def _extract_layer_parameters(model, layer_modules, v: Tuple[Tensor, ...]):
    """
    returns parameters and their names, grouped by layer
    """
    param_names = _params_to_names(
        _extract_parameters_from_layers(layer_modules), model
    )
    layer_param_names = [
        _params_to_names(_extract_parameters_from_layers([layer]), model)
        for layer in layer_modules
    ]
    param_name_to_index = {param_name: i for (i, param_name) in enumerate(param_names)}
    return [
        (
            _layer_param_names,
            tuple(
                [
                    v[param_name_to_index[param_name]]
                    for param_name in _layer_param_names
                ]
            ),
        )
        for _layer_param_names in layer_param_names
    ]
