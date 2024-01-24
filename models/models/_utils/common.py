from typing import Callable, Optional
import torch
import torch.nn as nn


def default_checkpoints_load_func(model, path):
    model.load_state_dict(torch.load(open(path, "rb")))


def load_model(
    model,
    checkpoints_load_func: Optional[Callable] = None,
    checkpoint: Optional[str] = None,
    device: str = "cpu",
):
    """
    loads model checkpoint if provided, moves to specified device
    """
    if checkpoints_load_func is not None:
        checkpoints_load_func(model, checkpoint)
    model.to(device=device)
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