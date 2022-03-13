# %%
# import os
import pandas as pd
from omegaconf import OmegaConf
from ekorpkit.corpora.loader import Corpora

cfg_path = 'conf/corpus_chunk.yaml'
cfg = OmegaConf.load(cfg_path)
print(OmegaConf.to_yaml(cfg, resolve=True))

cps = Corpora(**cfg)
# %%
dfs = []
for name in cps.corpora:
    print(name)
    df = cps.corpora[name]._data
    df['corpus'] = name
    dfs.append(df)

df_all = pd.concat(dfs)
# %%
df_all.head()

# %%
