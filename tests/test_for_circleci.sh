#!/bin/sh

echo "pytest for github-ci"
pytest -m "not local and not gpu"

