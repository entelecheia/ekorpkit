#!/bin/bash
set -x
set -o allexport
source .env.test
set +o allexport

docker build \
    --network=host --rm \
    . -t $EKORPKIT_TEST_DOCKER_IMAGE_NAME:$DATESTAMP

docker tag ${EKORPKIT_TEST_DOCKER_IMAGE_NAME}:${DATESTAMP} ${EKORPKIT_TEST_DOCKER_IMAGE_NAME}:latest
