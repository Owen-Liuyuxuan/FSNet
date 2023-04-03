FROM nvidia/cuda:11.1-cudnn8-devel-ubuntu20.04

RUN apt-get update
RUN apt-get upgrade -y
RUN apt-get update
RUN apt-get install python3.8 python3-pip nano libsm6 libxext6 libxrender-dev libgl1-mesa-glx libglib2.0-0 -y
RUN pip3 install -U pip
RUN pip3 install future -U
RUN apt install git nano htop -y
RUN pip3 install tensorflow pandas matplotlib numpy pillow opencv-python scikit-image numba tqdm cython fire easydict cityscapesscripts pyquaternion xeus-python pytest


ARG CUDA_VER="110"
ARG TORCH_VER="1.7.1"
ARG VISION_VER="0.8.2"

RUN pip3 install torch==${TORCH_VER} torchvision==${VISION_VER} -f https://download.pytorch.org/whl/cu${CUDA_VER}/torch_stable.html