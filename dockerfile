FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel as base

RUN sed -i s/archive.ubuntu.com/mirror.sjtu.edu.cn/g /etc/apt/sources.list \
&& sed -i s/security.ubuntu.com/mirror.sjtu.edu.cn/g /etc/apt/sources.list

ENV DEBIAN_FRONTEND=noninteractive

# Installs system dependencies.
RUN apt-get update \
        && apt-get clean

# Installs python dependencies. 
RUN pip install \
        matplotlib>\
        scipy\
        numpy

