#!/bin/sh
set -o allexport
source .env
set +o allexport

./codecov -t ${CODECOV_TOKEN}
