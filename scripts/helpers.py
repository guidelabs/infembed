def wandb_param_to_hydra_param(s: str):
    return s.replace('/', '.')