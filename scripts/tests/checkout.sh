#!/bin/sh

TEST_BRANCH=${1:-dev/0.1.40}

git clone https://github.com/entelecheia/ekorpkit.git --branch "$TEST_BRANCH" --single-branch

cd ekorpkit || exit
