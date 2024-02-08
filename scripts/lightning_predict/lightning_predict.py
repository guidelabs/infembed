import lightning as L
# import pytorch_lightning as pl
from omegaconf import DictConfig
import hydra
from hydra.utils import instantiate
import wandb
import torch


WANDB_CONFIG_NAME = 'wandb'


@hydra.main(version_base=None, config_path="conf", config_name="config")
def run(cfg: DictConfig):
    
    use_wandb = WANDB_CONFIG_NAME in cfg

    if use_wandb:
        wandb.init(project=cfg['wandb']['project'])

    trainer = L.Trainer(
        callbacks=instantiate(cfg.callbacks), **instantiate(cfg.trainer_kwargs)
    )

    trainer.predict(model=instantiate(cfg.model), dataloaders=instantiate(cfg.dataloader))


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    run()