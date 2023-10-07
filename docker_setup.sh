#!/bin/sh

# change these items:
# --name to your container's name
# --volume to the locations you have your data and repo

<< DockerTags :
DockerTags
# link to pytorch docker hub  #https://hub.docker.com/r/pytorch/pytorch/tags
# cuda 11.6 version
DOCKER_TAG=pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

<< DockerContainerBuild :
DockerContainerBuild
docker run -it --ipc=host \
      --gpus device=ALL \
      --name=ProtoASNet_AS_XAI  \
      --volume=path/to/your/workspace:/workspace \
      --volume=path/to/your/workspace:/data \
      $DOCKER_TAG
