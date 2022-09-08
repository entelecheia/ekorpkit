#!/bin/bash

rm -rf /workspace/projects/ekorpkit-test
cd ekorpkit
pip install -r requirements-test.txt
pip install -U -e .[all]
pytest
cd ..
# ./codecov -t ${CODECOV_TOKEN}
