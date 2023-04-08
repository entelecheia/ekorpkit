# Introduction

[![pypi-image]][pypi-url]
[![version-image]][release-url]
[![release-date-image]][release-url]
[![license-image]][license-url]
[![home-img]][home-url]
[![course-img]][course-url]
[![research-img]][research-url]
[![last-commit-img]][last-commit-url]

[pypi-image]: https://badge.fury.io/py/ekorpkit.svg
[pypi-url]: https://pypi.org/project/ekorpkit
[license-image]: https://img.shields.io/github/license/entelecheia/ekorpkit
[license-url]: https://github.com/entelecheia/ekorpkit/blob/main/LICENSE
[version-image]: https://img.shields.io/github/v/release/entelecheia/ekorpkit?sort=semver
[release-date-image]: https://img.shields.io/github/release-date/entelecheia/ekorpkit
[release-url]: https://github.com/entelecheia/ekorpkit/releases

[home-img]: https://img.shields.io/badge/home-entelecheia.me-blue
[home-url]: https://entelecheia.me
[course-img]: https://img.shields.io/badge/course-entelecheia.ai-blue
[course-url]: https://course.entelecheia.ai
[research-img]: https://img.shields.io/badge/research-entelecheia.ai-blue
[research-url]: https://research.entelecheia.ai
[linkedin-img]: https://img.shields.io/badge/LinkedIn-blue?logo=linkedin
[linkedin-url]: https://www.linkedin.com/in/entelecheia/
[last-commit-img]: https://img.shields.io/github/last-commit/entelecheia/lecture?label=last%20update
[last-commit-url]: https://github.com/entelecheia/lecture

## Introduction

eKorpkit is a powerful software tool that helps facilitate natural language processing (NLP) and machine learning (ML) research by providing a flexible interface for various stages of a research pipeline, such as data extraction, preprocessing, training, and visualization. The use of the [Hydra](https://hydra.cc/) and [pydantic](https://pydantic-docs.helpmanual.io) libraries for configuration management and validation makes it more flexible and efficient to use. It also designed to be user-friendly, extensible and shareable.

As Artificial Intelligence (AI) becomes more advanced, it is essential to consider not just its technical capabilities but also its ethical and societal implications. In this context, the ancient Greek concept of entelecheia, coined by Aristotle, is particularly relevant. The term refers to the state of having achieved one's full potential or the realization of one's purpose. In the context of AI, entelecheia can be understood as the ability of these technologies to fully realize their potential and perform their intended functions in an optimal way.

eKorpkit is designed with the principles of entelecheia in mind. It allows NLP and ML researchers to design, implement, and evaluate their models in a way that takes into account the underlying principles of human intelligence, such as the ability to understand context, handle ambiguity, and adapt to changing conditions.

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

## Contents

```{tableofcontents}

```
