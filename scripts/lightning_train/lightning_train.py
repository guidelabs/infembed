# import pytorch_lightning as pl
from omegaconf import DictConfig
import hydra
from hydra.utils import instantiate
import wandb
import lightning as L
import torch

"""
this script runs either `trainer.fit` or `trainer.validate` or `trainer.predict`.
arguments are passed in via hydra.  which of the 3 methods is run is determined by whether
the hydra argument `mode.mode` is 'fit', 'validate', or 'predict'.
"""

WANDB_CONFIG_NAME = "wandb"


@hydra.main(
    config_path="conf",
    config_name="config",
    version_base="1.3",
)
def run(cfg: DictConfig):
    """
    this is what is called from command line.  `cfg` is the config that hydra creates
    """
    use_wandb = WANDB_CONFIG_NAME in cfg

    if use_wandb:
        wandb.init(project=cfg["wandb"]["project"])
    _run(cfg)


def _run(cfg: DictConfig):
    """
    this is the function that actually runs either `trainer.fit` or `trainer.validate`
    or `trainer.predict`.

    it can be called via `run`, if one specifies the config via hydra, or via
    `lightning_train_wandb_sweep.py`, if one specifies the config via Wandb sweep.
    """

    trainer = L.Trainer(
        callbacks=instantiate(cfg.callbacks), **instantiate(cfg.trainer_kwargs)
    )

    mode = "train"
    if "mode" in cfg:
        mode = cfg["mode"]["mode"]
    method = {
        "fit": trainer.fit,
        "validate": trainer.validate,
        "predict": trainer.predict,
    }[mode]

    method(model=instantiate(cfg.model), datamodule=instantiate(cfg.datamodule))


if __name__ == "__main__":
    # torch.multiprocessing.set_start_method('spawn')
    run()
