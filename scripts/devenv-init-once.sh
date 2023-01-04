#!/bin/bash
set -x
set -o allexport
# shellcheck disable=SC1091
source .env.dev
# shellcheck disable=SC1091
source .env.docker
set +o allexport

# print all commands to stdout
set -x

chezmoi init --apply --verbose
chezmoi apply

service ssh start

mkdir -p "$JUPYTER_LOG_DIR"
DATESTAMP="$(date +'%y%m%d%H%M%S')"
LOGFILE="$JUPYTER_LOG_DIR/.jupyter-$DATESTAMP.log"
printf "Logs written to %s\n" "$LOGFILE"

# start jupyter notebook in background and redirect output to logfile
# change working directory to workspace root
# set token to value of JUPYTER_TOKEN
# set port to value of JUPYTER_DOCKER_PORT
nohup jupyter notebook \
    --no-browser \
    --notebook-dir="$DOCKER_WORKSPACE_ROOT" \
    -NotebookApp.token="$JUPYTER_TOKEN" \
    --port="$JUPYTER_DOCKER_PORT" \
    --ip=0.0.0.0 \
    --allow-root >"$LOGFILE" &

set +x 
