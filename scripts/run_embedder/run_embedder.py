from typing import Callable, List, Optional
from omegaconf import DictConfig, OmegaConf
import hydra
import importlib, os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from hydra.utils import instantiate


def run_embedder(
    embedder_constructor: Callable,
    model: nn.Module,
    layers: List[str],
    train_dataloader: Optional[DataLoader],
    test_dataloader: DataLoader,
    loss_fn: Callable,
    embeddings_path: str = "embeddings.pt",
    fit_results_path: Optional[str] = None,
    load_fit_results: bool = False,
    wrapper_embedder_constructor: Optional[Callable] = None,
):
    """
    Args:
        embedder_constructor (Callable): Function that given `model`, returns the
                `EmbedderBase` instance.
        model (Module): Model to provide to `embedder_constructor`.  It should already
                be on the correct device.
        layers (list of str): layers to consider gradients in
        train_dataloader (DataLoader, optional): Dataloader provided to `fit`.  It
                should yield batches on the correct device.
        test_dataloader (DataLoader): Dataloader provided to `predict`.  It should
                yield batches on the correct device.
        loss_fn (Callable): loss used for computing influence.
        embeddings_path (str): Where to save the computed embeddings.  It can
                either be a relative (to the working directory) path, or absolute path.
        fit_results_path (str, optional): Where the `save` will save and `load` will
                load.
        load_fit_results (bool, optional): Whether to call `load` instead of `fit`.
                Default: False
        wrapper_embedder_constructor (Callable, optional): Constructor whose arguments
                are just the embedder constructed from `embedder_constructor`.
    """
    embedder = embedder_constructor(model=model, layers=layers, loss_fn=loss_fn)
    if wrapper_embedder_constructor is not None:
        embedder = wrapper_embedder_constructor(base_embedder=embedder)
    if not load_fit_results:
        embedder.fit(train_dataloader)
    else:
        embedder.load(fit_results_path)
    embedder.save(fit_results_path)
    embeddings = embedder.predict(test_dataloader)
    torch.save(embeddings, open(embeddings_path, "wb"))


@hydra.main(version_base=None, config_path="conf", config_name="config")
def run(cfg: DictConfig):
    """
    computes embeddings.  see `run_embedder` to see what `cfg` should define.
    """
    # 'embedder_constructor' can either directly contain it, or also contain a
    # wrapper constructor.  handle the two cases
    if "wrapper_embedder_constructor" in cfg.embedder_constructor:
        embedder_constructor = instantiate(
            cfg.embedder_constructor.embedder_constructor
        )
        wrapper_embedder_constructor = instantiate(
            cfg.embedder_constructor.wrapper_embedder_constructor
        )
    else:
        embedder_constructor = instantiate(
            cfg.embedder_constructor
        )
        wrapper_embedder_constructor = None

    run_embedder(
        embedder_constructor,
        instantiate(cfg.model.model),
        OmegaConf.to_container(cfg.model.layers, resolve=True),
        instantiate(cfg.train_dataloader) if cfg.train_dataloader is not None else None,
        instantiate(cfg.test_dataloader),
        instantiate(cfg.loss),
        cfg.io.embeddings_path,
        cfg.io.fit_results_path,
        cfg.io.load_fit_results,
        wrapper_embedder_constructor,
    )


if __name__ == "__main__":
    run()
