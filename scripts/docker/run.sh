#!/bin/bash
set -x
set -o allexport
source .env.test
set +o allexport

CMD=${1:-/bin/bash}

docker run -it --rm \
  --runtime=nvidia \
  --gpus $NVIDIA_VISIBLE_DEVICES \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --env-file .env.test \
  --volume ${PWD}:/ekorpkit \
  --volume ${PWD}/workspace/data/archive/disco-imagen:/workspace/data/archive/disco-imagen \
  --name $EKORPKIT_TEST_DOCKER_CONTAINER_NAME \
  $EKORPKIT_TEST_DOCKER_IMAGE_NAME:latest $CMD
