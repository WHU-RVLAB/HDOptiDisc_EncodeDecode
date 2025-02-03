FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

# Installs system dependencies.
RUN apt-get update \
        && apt-get install -y \
        python3 python3-pip \
        && apt-get clean

# Installs python dependencies. 
RUN pip3 install --upgrade pip \
        && pip3 install matplotlib scipy numpy torch torchvision xgboost

