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

# Workflows

- A workflow is a configurable automated process that will run one or more jobs.
- You can divide your research into several unit jobs (tasks), then combine those jobs into one workflow.
- You can have multiple workflows, each of which can perform a different set of tasks.


## Run a job with `ekorpkit-run`

```bash
ekorpkit-run +job=build_corpus name=bok_minutes num_workers=10
```
