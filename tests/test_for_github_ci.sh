#!/bin/sh
set -e

echo "pytest for github-ci"
pytest -m "not local and not gpu"

