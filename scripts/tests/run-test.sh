#!/bin/bash

BRANCH=${1:-dev/0.1.39}

rm -rf /workspace/projects/ekorpkit-test
rm -rf ekorpkit

git clone https://github.com/entelecheia/ekorpkit.git --branch $BRANCH --single-branch
cd ekorpkit

pip install -r requirements-test.txt
pip install -U -e .[all]
pytest
# ./codecov -t ${CODECOV_TOKEN}

cd ..
