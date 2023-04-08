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

# Installation

Install the latest version of ekorpkit:

```bash
pip install -U ekorpkit
```

To install all extra dependencies,

```bash
pip install ekorpkit[all]
```

To install all extra dependencies, exhaustively, (not recommended)

```bash
pip install ekorpkit[exhaustive]
```

## Extra dependencies

### List of extra dependency sets

```{code-cell} ipython3
from ekorpkit import eKonf
eKonf.dependencies()
```

### List of libraries in each dependency set

```{code-cell} ipython3
from ekorpkit import eKonf
eKonf.dependencies("tokenize")
```

```{code-cell} ipython3
eKonf.dependencies("dataset")
```

```{code-cell} ipython3
eKonf.dependencies("model")
```

```{code-cell} ipython3
eKonf.dependencies("visualize")
```

```{code-cell} ipython3
eKonf.dependencies("all")
```

### self-upgrade of ekorpkit

```{code-cell} ipython3
eKonf.upgrade(
  prelease=True, 
  quiet=True, 
  verbose=True, 
  force_reinstall=False)
```
