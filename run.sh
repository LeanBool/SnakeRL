#!/bin/sh

IMAGE_NAME="snake-rl-container"

sudo xhost +local:docker
docker build --network host -t ${IMAGE_NAME} . 
docker run --privileged -p 8080:6006 -it --volume /tmp/.X11-unix:/tmp/.X11-unix --gpus all --env DISPLAY=$DISPLAY ${IMAGE_NAME}

CONTAINER_ID="$(sudo docker ps -a | grep 'snake-rl-container' | awk '{print $1}')"
docker cp ${CONTAINER_ID}:/home/docker_user/model/ ./
chmod -R 777 ./model