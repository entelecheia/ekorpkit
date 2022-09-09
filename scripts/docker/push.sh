#!/bin/bash
set -x
set -o allexport
source .env
set +o allexport

docker push $EKORPKIT_TEST_DOCKER_IMAGE_NAME:latest
docker push $EKORPKIT_TEST_DOCKER_IMAGE_NAME:$DATESTAMP
