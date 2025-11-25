#!/bin/sh

IMAGE_NAME="snake-rl-container"

# use sudo xhost +local:docker
docker build --network host -t ${IMAGE_NAME} . 
docker run --privileged -p 6006:6006 -it --volume /tmp/.X11-unix:/tmp/.X11-unix --gpus all --env DISPLAY=$DISPLAY ${IMAGE_NAME}