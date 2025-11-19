#!/bin/sh

IMAGE_NAME="snake-rl-container"

if [ "$(docker images -q ${IMAGE_NAME}:latest)" ]; then
  docker run -it --gpus all ${IMAGE_NAME}
else
  docker build --network host -t ${IMAGE_NAME} . && docker run -it --gpus all ${IMAGE_NAME}
fi