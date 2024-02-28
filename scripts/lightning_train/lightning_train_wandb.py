import wandb
from hydra import compose, initialize

# from ..helpers import wandb_param_to_hydra_param
from lightning_train import _run


"""
this script is called by the wandb sweeper agent.  it accepts the name of a hydra
config, as well as optional overrides as positional arguments.

example call: `python lightning_train_wandb.py +trainer_kwargs.max_steps=100 --config-name='toy`

usually, this script would be called by the wandb sweeper agent. see
`sweep_configs/test.yaml` for an example sweep config.  note that because parameters in
wandb sweep config cannot have '.' in them, parameters to sweep over are specified via 
"dot notation", except '.' is replaced with '-'.  for example, specify 'trainer_kwargs-max_steps'
instead of 'trainer_kwargs.max_steps'.
"""


# TODO: put globals in helpers
WANDB_CONFIG_NAME = "wandb"


def wandb_param_to_hydra_param(s: str):
    return s.replace("-", ".")


def run():
    """
    this is the method which will be called by the wandb sweeper
    """
    # read in the wandb config, which will have been populated by sweeps
    # these specify modifications to make to default config, as in normal hydra usage
    # normally, these would have been handled by the decorator.  here, need to instead
    # pass modifications to that decorator and get out the modified config.
    # do this with https://hydra.cc/docs/advanced/compose_api/

    # wandb config has overrides as a dict.  parameters refered to with dot notation
    # use compose api to get uppdated config

    wandb.init()

    with initialize(version_base=None, config_path="conf"):
        overrides = [
            f"{wandb_param_to_hydra_param(k)}={v}" for (k, v) in wandb.config.items()
        ]
        # TODO:
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("overrides", type=str, nargs="?")
        parser.add_argument("--config-name", type=str, default="config")
        args = parser.parse_args()
        print(args.overrides)
        cfg = compose(config_name=args.config_name, overrides=overrides)

        # once have the config, pass it to the workhorse `_run` method
        _run(cfg)


if __name__ == "__main__":
    run()
