# import pytorch_lightning as pl
from omegaconf import DictConfig
import hydra
from hydra.utils import instantiate
import wandb

# import lightning.pytorch as L
import lightning as L
import torch
import yaml
import pytorch_lightning
import os


def run():

    if False:
        datamodule_yaml = "/home/ubuntu/Documents/infembed/scripts/lightning_train/conf/datamodule/tinystories_manual.yaml"
        model_yaml = "/home/ubuntu/Documents/infembed/scripts/lightning_train/conf/model/tinystories_33M.yaml"
        callbacks_yaml = "/home/ubuntu/Documents/infembed/scripts/lightning_train/conf/callbacks/standard.yaml"
        trainer_kwargs = {
            'accumulate_grad_batches': 8,
            'logger': pytorch_lightning.loggers.WandbLogger(),
        }
    if True:
        datamodule_yaml = "/home/ubuntu/Documents/infembed/scripts/lightning_train/conf/datamodule/tinystories_concept_real.yaml"
        model_yaml = "/home/ubuntu/Documents/infembed/scripts/lightning_train/conf/model/tinystories_33M_concept.yaml"
        callbacks_yaml = "/home/ubuntu/Documents/infembed/scripts/lightning_train/conf/callbacks/standard.yaml"
        trainer_kwargs = {
            'accumulate_grad_batches': 8,
            'logger': pytorch_lightning.loggers.WandbLogger(),
            'devices': [0],#,1,2,3],
            'strategy': 'ddp',
        }
    

    # working_dir = './lightning_train/test_concept'
    # try:
    #     os.makedirs(working_dir)
    #     import pdb
    #     pdb.set_trace()
    # except:
    #     pass
    # os.chdir(working_dir)

    with open(datamodule_yaml, "r") as f:
        datamodule = instantiate(yaml.safe_load(f))

    with open(model_yaml, "r") as f:
        model = instantiate(yaml.safe_load(f))

    with open(callbacks_yaml, "r") as f:
        callbacks = instantiate(yaml.safe_load(f))

    trainer = L.Trainer(
        callbacks=callbacks,
        **trainer_kwargs,
    )

    trainer.fit(model=model, datamodule=datamodule)

    assert False

    yaml.safe_load(stream)

    use_wandb = WANDB_CONFIG_NAME in cfg

    if use_wandb:
        wandb.init(project=cfg["wandb"]["project"])

    # trainer = L.Trainer(
    #     callbacks=instantiate(cfg.callbacks), **instantiate(cfg.trainer_kwargs)
    # )
    trainer = L.Trainer()
    # trainer = L.Trainer(
    #     **instantiate(cfg.trainer_kwargs)
    # )

    validate = False
    if "validate" in cfg:
        validate = cfg["validate"]["validate"]

    if not validate:
        trainer.fit(
            model=instantiate(cfg.model), datamodule=instantiate(cfg.datamodule)
        )
    else:
        trainer.validate(
            model=instantiate(cfg.model), datamodule=instantiate(cfg.datamodule)
        )


if __name__ == "__main__":
    # torch.multiprocessing.set_start_method('spawn')
    run()
