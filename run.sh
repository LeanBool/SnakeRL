#!/bin/sh

IMAGE_NAME="snake-rl-container"

docker build --network host -t ${IMAGE_NAME} . 
docker run -it --volume /tmp/.X11-unix:/tmp/.X11-unix --gpus all --env DISPLAY=$DISPLAY ${IMAGE_NAME}