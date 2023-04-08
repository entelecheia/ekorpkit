# ekorpkit 【iːkɔːkɪt】 : **eKo**nomic **R**esearch **P**ython Tool**kit**

[![pypi-image]][pypi-url]
[![version-image]][release-url]
[![release-date-image]][release-url]
[![jupyter-book-image]][jupyter book]
[![codeql-image]][codeql-url]
[![test-image]][test-url]
[![circleci-image]][circleci-url]
[![codecov-image]][codecov-url]
[![license-image]][license-url]

<!-- Links: -->

[pypi-image]: https://badge.fury.io/py/ekorpkit.svg
[pypi-url]: https://pypi.org/project/ekorpkit
[license-image]: https://img.shields.io/github/license/entelecheia/ekorpkit
[license-url]: https://github.com/entelecheia/ekorpkit/blob/main/LICENSE
[version-image]: https://img.shields.io/github/v/release/entelecheia/ekorpkit?sort=semver
[release-date-image]: https://img.shields.io/github/release-date/entelecheia/ekorpkit
[release-url]: https://github.com/entelecheia/ekorpkit/releases
[jupyter-book-image]: https://jupyterbook.org/en/stable/_images/badge.svg
[jupyter book]: https://entelecheia.cc
[codeql-image]: https://github.com/entelecheia/ekorpkit/actions/workflows/codeql-analysis.yml/badge.svg
[codeql-url]: https://github.com/entelecheia/ekorpkit/actions/workflows/codeql-analysis.yml
[test-image]: https://github.com/entelecheia/ekorpkit/actions/workflows/test.yaml/badge.svg
[test-url]: https://github.com/entelecheia/ekorpkit/actions/workflows/test.yaml
[circleci-image]: https://circleci.com/gh/entelecheia/ekorpkit/tree/main.svg?style=shield
[circleci-url]: https://circleci.com/gh/entelecheia/ekorpkit/tree/main
[codecov-image]: https://codecov.io/gh/entelecheia/ekorpkit/branch/main/graph/badge.svg?token=8I4ORHRREL
[codecov-url]: https://codecov.io/gh/entelecheia/ekorpkit
[repo-url]: https://github.com/entelecheia/ekorpkit
[docs-url]: https://entelecheia.cc
[changelog]: https://github.com/entelecheia/ekorpkit/blob/main/CHANGELOG.md
[contributing guidelines]: https://github.com/entelecheia/ekorpkit/blob/main/CONTRIBUTING.md

<!-- Links: -->

eKorpkit provides a flexible interface for NLP and ML research pipelines such as extraction, transformation, tokenization, training, and visualization. Its powerful config composition is backed by [Hydra](https://hydra.cc/).

## Warning: This is a work in progress

This project is still under development. The API is subject to change. Until the first stable release, the version number will be 0.x.x. Please use it at your own risk. If you have any questions or suggestions, please feel free to contact me.

Especially, some core configuration interface parts of the package will be carbed out and moved to a separate package. The package will be renamed to [**hyfi**](https://github.com/entelecheia/hyfi) (Hydra Fast Interface). Image generation and visualization will be moved to a separate package. The package will be renamed to [**ekaros**](https://github.com/entelecheia/ekaros) (from Íkaros[Icarus] in Greek mythology).

## Key features

### Easy Configuration

- You can compose your configuration dynamically, enabling you to easily get the perfect configuration for each research.
- You can override everything from the command line, which makes experimentation fast, and removes the need to maintain multiple similar configuration files.
- With a help of the **eKonf** class, it is also easy to compose configurations in a jupyter notebook environment.

### No Boilerplate

- eKorpkit lets you focus on the problem at hand instead of spending time on boilerplate code like command line flags, loading configuration files, logging etc.

### Workflows

- A workflow is a configurable automated process that will run one or more jobs.
- You can divide your research into several unit jobs (tasks), then combine those jobs into one workflow.
- You can have multiple workflows, each of which can perform a different set of tasks.

### Sharable and Reproducible

- With eKorpkit, you can easily share your datasets and models.
- Sharing configs along with datasets and models makes every research reproducible.
- You can share each unit jobs or an entire workflow.

### Pluggable Architecture

- eKorpkit has a pluggable architecture, enabling it to combine with your own implementation.

## [Tutorials](https://entelecheia.github.io/ekorpkit-book)

Tutorials for [ekorpkit](https://github.com/entelecheia/ekorpkit) package can be found at https://entelecheia.github.io/ekorpkit-book/

## [Installation](https://entelecheia.github.io/ekorpkit-book/docs/basics/install.html)

Install the latest version of ekorpkit:

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
  title        = {eKorpkit: eKonomic Research Python Toolkit},
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
  title        = {eKorpkit: eKonomic Research Python Toolkit},
  month        = apr,
  year         = 2022,
  publisher    = {GitHub},
  url          = {https://github.com/entelecheia/ekorpkit}
}
```

## Changelog

See the [CHANGELOG] for more information.

## Contributing

Contributions are welcome! Please see the [contributing guidelines] for more information.

## License

- This project is released under the [MIT License][license-url].
- Each corpus adheres to its own license policy. Please check the license of the corpus before using it!
