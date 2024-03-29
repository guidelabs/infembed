# import pytorch_lightning as pl
from omegaconf import DictConfig
import hydra
from hydra.utils import instantiate
import wandb
# import lightning.pytorch as L
import lightning as L

WANDB_CONFIG_NAME = 'wandb'


@hydra.main(version_base=None, config_path="conf", config_name="config")
def run(cfg: DictConfig):
    
    use_wandb = WANDB_CONFIG_NAME in cfg

    if use_wandb:
        wandb.init(project=cfg['wandb']['project'])

    trainer = L.Trainer(
        callbacks=instantiate(cfg.callbacks), **instantiate(cfg.trainer_kwargs)
    )

    validate = False
    if 'validate' in cfg:
        validate = cfg['validate']['validate']

    if not validate:
        trainer.fit(model=instantiate(cfg.model), datamodule=instantiate(cfg.datamodule))
    else:
        trainer.validate(model=instantiate(cfg.model), datamodule=instantiate(cfg.datamodule))


if __name__ == "__main__":
    run()