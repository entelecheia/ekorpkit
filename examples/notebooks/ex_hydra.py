# %%
import os
from hydra import initialize, initialize_config_module, initialize_config_dir, compose
from omegaconf import OmegaConf

with initialize_config_module(config_module="ekorpkit.conf"):
    cfg = compose(config_name="config")
    print(cfg)
# %%
print(OmegaConf.to_yaml(cfg, resolve=True))
# %%
