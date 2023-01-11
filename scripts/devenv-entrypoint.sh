#!/bin/bash
set -x
set -o allexport
# shellcheck disable=SC1091
source .env.dev
# shellcheck disable=SC1091
source .docker/.env.docker.dev
set +o allexport

# print all commands to stdout
set -x

service ssh start

# start jupyter notebook in background and redirect output to logfile
# change working directory to workspace root
# set token to value of JUPYTER_TOKEN
# set port to value of JUPYTER_DOCKER_PORT
jupyter lab \
    --no-browser \
    --notebook-dir="$CONTAINER_WORKSPACE_ROOT" \
    --ServerApp.token="$JUPYTER_TOKEN" \
    --port="$CONTAINER_JUPYTER_PORT" \
    --ip=0.0.0.0 \
    --allow-root

set +x 
