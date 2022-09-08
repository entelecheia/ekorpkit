#!/bin/bash
set -x
set -o allexport
source .env.test
set +o allexport

rm -rf /workspace/projects/ekorpkit-test
pip install -r requirements-test.txt
pip install -U -e .[all]
pytest
# ./codecov -t ${CODECOV_TOKEN}
