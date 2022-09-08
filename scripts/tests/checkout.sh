#!/bin/sh

# Set next version number
BRANCH=$1

git clone https://github.com/entelecheia/ekorpkit.git --branch $BRANCH --single-branch
cd ekorpkit

