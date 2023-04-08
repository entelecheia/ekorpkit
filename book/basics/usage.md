---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Usage

## Via Command Line Interface (CLI)

```{code-cell} ipython3
!ekorpkit
```

### CLI example to build a corpus

```bash
ekorpkit --config-dir /workspace/projects/ekorpkit-book/config  \
    project=esgml \
    dir.workspace=/workspace \
    verbose=false \
    print_config=false \
    num_workers=1 \
    cmd=fetch_builtin_corpus \
    +corpus/builtin=_dummy_fomc_minutes \
    corpus.builtin.io.force.summarize=true \
    corpus.builtin.io.force.preprocess=true \
    corpus.builtin.io.force.build=false \
    corpus.builtin.io.force.download=false
```

### CLI Help

To see the available configurations for CLI, run the command:

```{code-cell} ipython3
:tags: [output_scroll]
!ekorpkit --help
```

```{code-cell} ipython3
!ekorpkit --info defaults
```

## Via Python

### Compose an ekorpkit config

```{code-cell} ipython3
:tags: [output_scroll]
from ekorpkit import eKonf
cfg = eKonf.compose()
print('Config type:', type(cfg))
eKonf.pprint(cfg)
```

### Instantiating objects with an ekorpkit config

#### compose a config for the nltk class

```{code-cell} ipython3
from ekorpkit import eKonf
config_group='preprocessor/tokenizer=nltk'
cfg = eKonf.compose(config_group=config_group)
eKonf.pprint(cfg)
nltk = eKonf.instantiate(cfg)
```

```{code-cell} ipython3
text = "I shall reemphasize some of those thoughts today in the context of legislative proposals that are now before the current Congress."
nltk.tokenize(text)
```

```{code-cell} ipython3
 nltk.nouns(text)
```

#### compose a config for the mecab class

```{code-cell} ipython3
config_group='preprocessor/tokenizer=mecab'
cfg = eKonf.compose(config_group=config_group)
eKonf.pprint(cfg)
```

#### intantiate a mecab config and tokenize a text

```{code-cell} ipython3
mecab = eKonf.instantiate(cfg)
text = 'IMF가 推定한 우리나라의 GDP갭률은 今年에도 소폭의 마이너스(−)를 持續하고 있다.'
mecab.tokenize(text)
```

#### compose and instantiate a `formal_ko` config for the normalizer class

```{code-cell} ipython3
config_group='preprocessor/normalizer=formal_ko'
cfg_norm = eKonf.compose(config_group=config_group)
norm = eKonf.instantiate(cfg_norm)
norm(text)
```

#### instantiate a mecab config with the above normalizer config

```{code-cell} ipython3
config_group='preprocessor/tokenizer=mecab'
cfg = eKonf.compose(config_group=config_group)
cfg.normalize = cfg_norm
mecab = eKonf.instantiate(cfg)
mecab.tokenize(text)
```
