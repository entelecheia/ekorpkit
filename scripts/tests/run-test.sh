#!/bin/bash

TEST_BRANCH=${1:-dev/0.1.40}
CODECOV_TOKEN=${CODECOV_TOKEN:-""}

rm -rf /workspace/projects/ekorpkit-test
rm -rf ekorpkit

git clone https://github.com/entelecheia/ekorpkit.git --branch "$TEST_BRANCH" --single-branch
cd ekorpkit || exit

pip install -r requirements-test.txt
pip install -U -e ".[all]"
pytest
if [ -n "$CODECOV_TOKEN" ]; then
  codecov -t "${CODECOV_TOKEN}"
fi

cd ..
