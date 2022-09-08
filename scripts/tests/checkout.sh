#!/bin/sh

# Set next version number
BRANCH=${1:-dev/0.1.39}

git clone https://github.com/entelecheia/ekorpkit.git --branch $BRANCH --single-branch
cd ekorpkit
