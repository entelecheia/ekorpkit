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

path = '/workspace/data/projects/eKonPLM/data/lighttag/label-samples-tokens-1119/esg_topics2-LDA.k30-train-samples.csv'

df_sample = pd.read_csv(path)
df_sample.head()
# %%
df_sample_new = df_sample.merge(df_all, on=['corpus', 'id', 'chunk_no'])
# %%
print(len(df_sample), len(df_sample_new))
# %%
df_sample_new.tail(10)
# %%
