#!/bin/sh

# Set next version number
RELEASE=$1

# Create tags
git commit --allow-empty -m "Release $RELEASE"
git tag -a "$RELEASE" -m "Version $RELEASE"

# Push
git push --tags