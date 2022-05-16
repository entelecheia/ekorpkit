# ekorpkit[iːkɔːkɪt]: (e)nglish (K)orean C(orp)us Tool(kit)

[![PyPI version](https://badge.fury.io/py/ekorpkit.svg)](https://badge.fury.io/py/ekorpkit) [![Jupyter Book Badge](https://jupyterbook.org/en/stable/_images/badge.svg)](https://entelecheia.github.io/ekorpkit-config/) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6497226.svg)](https://doi.org/10.5281/zenodo.6497226) [![release](https://github.com/entelecheia/ekorpkit/actions/workflows/release.yaml/badge.svg)](https://github.com/entelecheia/ekorpkit/actions/workflows/release.yaml) [![CodeQL](https://github.com/entelecheia/ekorpkit/actions/workflows/codeql-analysis.yml/badge.svg)](https://github.com/entelecheia/ekorpkit/actions/workflows/codeql-analysis.yml) [![test](https://github.com/entelecheia/ekorpkit/actions/workflows/test.yaml/badge.svg)](https://github.com/entelecheia/ekorpkit/actions/workflows/test.yaml) [![CircleCI](https://circleci.com/gh/entelecheia/ekorpkit/tree/main.svg?style=shield)](https://circleci.com/gh/entelecheia/ekorpkit/tree/main) [![codecov](https://codecov.io/gh/entelecheia/ekorpkit/branch/main/graph/badge.svg?token=8I4ORHRREL)](https://codecov.io/gh/entelecheia/ekorpkit) [![markdown-autodocs](https://github.com/entelecheia/ekorpkit/actions/workflows/markdown-autodocs.yaml/badge.svg)](https://github.com/entelecheia/ekorpkit/actions/workflows/markdown-autodocs.yaml)

eKorpkit provides a flexible interface for corpus management and analysis pipelines such as extraction, transformation, tokenization, training, and visualization.

- Powerful config composition backed by [Hydra](https://hydra.cc/) - Easily swap out corpora, datasets, models, preprocessors, visualizers and many more configurations without touching the code.

## [Tutorials](https://entelecheia.github.io/ekorpkit-config)

Tutorials for [ekorpkit](https://github.com/entelecheia/ekorpkit) package can be found at https://entelecheia.github.io/ekorpkit-config/

## [Installation](https://entelecheia.github.io/ekorpkit-config/docs/about/install.html)

Install the latest version of ekorpit:

```bash
pip install ekorpkit
```

To install all extra dependencies,

```bash
pip install ekorpkit[all]
```

## [The eKorpkit Corpus](https://github.com/entelecheia/ekorpkit/blob/main/docs/corpus/README.md)

The eKorpkit Corpus is a large, diverse, bilingual (ko/en) language modelling dataset.

![ekorpkit corpus](https://github.com/entelecheia/ekorpkit/blob/main/docs/figs/ekorpkit_corpus.png?raw=true)

## Citation

```tex
@software{lee_2022_6497226,
  author       = {Young Joon Lee},
  title        = {eKorpkit: English Korean Corpus Toolkit},
  month        = apr,
  year         = 2022,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.6497226},
  url          = {https://doi.org/10.5281/zenodo.6497226}
}
```

```tex
@software{lee_2022_ekorpkit,
  author       = {Young Joon Lee},
  title        = {eKorpkit: English Korean Corpus Toolkit},
  month        = apr,
  year         = 2022,
  publisher    = {GitHub},
  url          = {https://github.com/entelecheia/ekorpkit}
}
```

## License

- eKorpkit is licensed under the Creative Commons License(CCL) 4.0 [CC-BY](https://creativecommons.org/licenses/by/4.0). This license covers the eKorpkit package and all of its components.
- Each corpus adheres to its own license policy. Please check the license of the corpus before using it!
