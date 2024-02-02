from typing import Callable, Optional
import torch
import torch.nn as nn


def default_checkpoints_load_func(model, path, key=None):
    state = torch.load(open(path, "rb"))
    state_dict = state if key is None else state[key]
    model.load_state_dict(state_dict)


def lightning_checkpoints_load_func(model, path):
    raise NotImplementedError
    model.load_from_checkpoint(path)


def load_model(
    model,
    checkpoints_load_func: Optional[Callable] = None,
    checkpoint: Optional[str] = None,
    device: str = "cpu",
    eval: bool = True,
):
    """
    loads model checkpoint if provided, moves to specified device
    """
    if checkpoints_load_func is not None:
        checkpoints_load_func(model=model, path=checkpoint)
    model.to(device=device)
    if eval:
        model.eval()
    else:
        model.train()
    return model


class HuggingfaceWrapperModel(nn.Module):
    """
    `EmbedderBase` implementations expects a batch to be a tuple, where the last
    element is the label, and the previous elements are the features to be given to
    forward.  this wraps huggingface models to accept batches of that format.  it just
    duplicates the dictionary representing the huggingface input.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, features):
        return self.model(**features)


class HuggingfaceLoss(nn.Module):
    def __call__(self, output, target):
        return output.loss


def init_linear(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None: nn.init.zeros_(m.bias)


def get_all_parameters(model):
    return model.parameters()


class GenericConfigureOptimizers:
    def __init__(self, parameters_getter, optimizer_constructor):
        self.parameters_getter, self.optimizer_constructor = parameters_getter, optimizer_constructor

    def __call__(self, model):
        return self.optimizer_constructor(self.parameters_getter(model=model))