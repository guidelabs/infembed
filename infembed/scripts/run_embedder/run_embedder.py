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
    embeddings_path: str = "embeddings.pt",
    fit_results_path: Optional[str] = None,
    load_fit_results: bool = False
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
        embeddings_path (str): Where to save the computed embeddings.  It can
                either be a relative (to the working directory) path, or absolute path.
        fit_results_path (str, optional): Where the `save` will save and `load` will
                load.
        load_fit_results (bool, optional): Whether to call `load` instead of `fit`.
                Default: False
    """
    embedder = embedder_constructor(model=model, layers=layers)
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
    run_embedder(
        instantiate(cfg.embedder_constructor),
        instantiate(cfg.model.model),
        OmegaConf.to_container(cfg.model.layers, resolve=True),
        instantiate(cfg.train_dataloader) if cfg.train_dataloader is not None else None,
        instantiate(cfg.test_dataloader),
        cfg.io.embeddings_path,
        cfg.io.fit_results_path,
        cfg.io.load_fit_results,
    )


if __name__ == "__main__":
    run()
