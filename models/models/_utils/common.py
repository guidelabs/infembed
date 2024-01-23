from typing import Callable, Optional
import torch


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