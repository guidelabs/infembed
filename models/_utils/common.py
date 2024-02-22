import copy
from functools import reduce
from typing import Any, Callable, Dict, List, Optional
from data._utils.common import default_batch_to_target, default_batch_to_x
import torch
import torch.nn as nn
import lightning.pytorch as pl
import lightning as L
import torch.nn.functional as F


def default_checkpoints_load_func(
    model,
    path,
    device,
    key=None,
    remove_prefix=None,
    add_prefix=None,
    active_module_names: Optional[List[str]] = None,
):
    # get state dict
    state = torch.load(open(path, "rb"), map_location=device)
    state_dict = state if key is None else state[key]

    # modify keys if needed
    if remove_prefix is not None:
        state_dict = {
            key[len(remove_prefix) :]: val for (key, val) in state_dict.items()
        }
    if add_prefix is not None:
        state_dict = {(add_prefix + key): val for (key, val) in state_dict.items()}
    print(model.load_state_dict(state_dict, strict=False))

    # set active modules if needed
    if active_module_names is not None:
        for p in model.parameters():
            p.requires_grad = False
        for module_name in active_module_names:
            for p in _get_module_from_name(model, module_name).parameters():
                p.requires_grad = True


def lightning_checkpoints_load_func(model, path):
    # raise NotImplementedError
    model.load_from_checkpoint(path)


def lightning_load_model(load_from_checkpoint_fn, path, eval):
    model = load_from_checkpoint_fn(path)
    if eval:
        model.eval()
    else:
        model.train()
    return model


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
    if checkpoints_load_func is not None and checkpoint is not None:
        checkpoints_load_func(model=model, path=checkpoint, device=device)
    else:
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
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def get_all_parameters(model):
    return model.parameters()


class GenericConfigureOptimizers:
    def __init__(self, parameters_getter, optimizer_constructor):
        self.parameters_getter, self.optimizer_constructor = (
            parameters_getter,
            optimizer_constructor,
        )

    def __call__(self, model):
        return self.optimizer_constructor(self.parameters_getter(model=model))


class GenericLightningModule(L.LightningModule):
    def __init__(
        self,
        decoder,
        loss_fn=None,
        configure_optimizers=None,
        scheduler_constructor=None,
    ):
        super().__init__()
        self.decoder, self.loss_fn, self._configure_optimizers = (
            decoder,
            loss_fn,
            configure_optimizers,
        )
        self.scheduler_constructor = scheduler_constructor

    _STEP_DO_NOT_LOG_KEYS = []

    def configure_optimizers(self):
        optimizer = self._configure_optimizers(self)
        if self.scheduler_constructor is None:
            return optimizer
        else:
            scheduler = self.scheduler_constructor(optimizer=optimizer)
            return [optimizer], [scheduler]

    def _step(self, batch, batch_idx) -> Dict:
        """
        this should call forward, additionally use the batch to compute things
        that depend on labels, and return a dictionary.  a subset of its output will be
        logged
        """
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        d = self._step(batch, batch_idx)
        self.log_dict(
            {
                f"train_{key}": val
                for (key, val) in d.items()
                if key[0] != "_" and key not in self._STEP_DO_NOT_LOG_KEYS
            },
            on_step=True,
            on_epoch=True,
        )
        return d

    def validation_step(self, batch, batch_idx):
        d = self._step(batch, batch_idx)
        self.log_dict(
            {
                f"validation_{key}": val
                for (key, val) in d.items()
                if key[0] != "_" and key not in self._STEP_DO_NOT_LOG_KEYS
            },
            on_step=True,
            on_epoch=True,
        )
        return d

    def prediction_step(self, batch, batch_idx):
        raise NotImplementedError

    def forward(self, batch):
        """
        this should return whatever can be computed without labels, and return a
        dictionary
        """


class MLP(nn.Module):
    def __init__(self, dims, pre_nonlinearity=False, post_nonlinearity=False):
        super().__init__()
        self.linears = nn.ModuleList(
            [
                nn.Linear(dim_in, dim_out)
                for (dim_in, dim_out) in zip(dims[:-1], dims[1:])
            ]
        )
        self.pre_nonlinearity, self.post_nonlinearity = (
            pre_nonlinearity,
            post_nonlinearity,
        )

    def forward(self, x):
        linears = iter(self.linears)
        if self.pre_nonlinearity:
            x = F.relu(x)
        x = next(linears)(x)
        for linear in linears:
            x = F.relu(x)
            x = linear(x)
        if self.post_nonlinearity:
            x = F.relu(x)
        return x


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def _get_module_from_name(model: nn.Module, layer_name: str) -> Any:
    r"""
    Returns the module (layer) object, given its (string) name
    in the model.

    Args:
            name (str): Module or nested modules name string in self.model

    Returns:
            The module (layer) in self.model.
    """

    return reduce(getattr, layer_name.split("."), model)


### DEPRECATED ###


class _GenericLightningModel(pl.LightningModule):
    """
    the most basic pl wrapper whose purpose is just to train.  doesn't log anything
    besides loss
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: Callable,
        configure_optimizers=None,
        batch_to_x: Callable = default_batch_to_x,
        batch_to_target: Callable = default_batch_to_target,
    ):
        super().__init__()
        self.model, self.loss_fn = model, loss_fn
        self._configure_optimizers = configure_optimizers
        self.batch_to_x = batch_to_x
        self.batch_to_target = batch_to_target

    def configure_optimizers(self):
        if self._configure_optimizers is not None:
            return self._configure_optimizers(self)
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, x):
        return self.model(*x)

    def _step(self, batch, batch_idx):
        # run forward
        x = self.batch_to_x(batch)
        y = self.batch_to_target(batch)
        y_hat = self.forward(x)
        loss = self.loss_fn(y_hat, y)
        return {"loss": loss}

    def training_step(self, batch, batch_idx):
        d = self._step(batch, batch_idx)
        self.log_dict(
            {f"train_{key}": val for (key, val) in d.items() if key[0] != "_"}
        )
        return d

    def validation_step(self, batch, batch_idx):
        d = self._step(batch, batch_idx)
        self.log_dict(
            {f"validation_{key}": val for (key, val) in d.items() if key[0] != "_"}
        )
        return d

    def prediction_step(self, batch, batch_idx):
        d = self._step(batch, batch_idx)
        self.log_dict(
            {f"prediction_{key}": val for (key, val) in d.items() if key[0] != "_"}
        )
        return d
