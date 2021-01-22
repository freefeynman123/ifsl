# This image can run with or without GPUs
FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04

# Disable interactive functions
ENV DEBIAN_FRONTEND noninteractive

# Basic stuff
RUN apt-get update --fix-missing && \
    apt-get install -y \
        locales \
        wget \
        bzip2 \
        sudo \
        git \
        git-lfs \
        nano \
        vim \
        zip \
        unrar \
        curl \
        htop \
        iperf \
        iotop \
        iputils-ping \
        net-tools \
        tmux \
        smbclient \
        python3 \
        python3-pip \
        tzdata \
        libsm6 libxext6 libxrender-dev \
        libgl1-mesa-glx \
        && rm -rf /var/lib/apt/lists/*

# Fix UTF
# https://stackoverflow.com/a/38553499/1407427
RUN sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen && \
    dpkg-reconfigure --frontend=noninteractive locales && \
    update-locale LANG=en_US.UTF-8
ENV LANG en_US.UTF-8

# Set timezone
ENV TZ Europe/Warsaw
RUN sudo dpkg-reconfigure -f noninteractive tzdata

# Install basic python stack
RUN pip3 install virtualenv\
                 jupyter \
                 ipython \
                 scipy \
                 sklearn \
                 prettytable \
                 termcolor \
                 tqdm \
                 requests \
                 h5py \
                 pillow \
                 matplotlib \
                 seaborn \
                 --no-cache-dir

# Create user user "tclbot" with password "tcl". User will be have sudo permissions so will be able to install new stuff on the fly.
RUN useradd -ms /bin/bash tclbot; usermod -aG users,sudo,adm tclbot; echo "tclbot:tcl" | chpasswd
# Now, switch to the new user
USER tclbot

# Set workdir
WORKDIR /home/tclbot

COPY requirements.txt /home/tclbot

# Install basic python stack
RUN pip3 install --upgrade pip && \
    pip3 install -U wandb && \
    pip3 install -r requirements.txt\
     --no-cache-dir

