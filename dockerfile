FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive

# Installs system dependencies.
RUN apt-get update \
        && apt-get clean

# Installs python dependencies. 
RUN pip install --upgrade pip \
        && pip install matplotlib scipy numpy \
        && pip cache purge

