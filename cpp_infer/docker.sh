# two ways to run docker
docker run -it  \
    -v /home/jing/Workspace/Data/zc/ARMData/:/data/ \
    --gpus all \
    --runtime=nvidia \
    --volume=/tmp/.X11-unix:/tmp/.X11-unix \
    -e DISPLAY=$DISPLAY \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e NVIDIA_DRIVER_CAPABILITIES=all \
nvidia/cuda:latest /bin/bash
#zc/cuda10.2-cudnn8-devel-ubuntu18.04-trt7134 /bin/bash
#docker run -it f9a80a55f492 /bin/bash


# docker attach ubuntu:18.04
