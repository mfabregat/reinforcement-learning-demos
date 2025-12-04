FROM pytorch/pytorch:2.9.1-cuda13.0-cudnn9-devel

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

ARG USER_UID=1000
ARG USER_GID=1000
ARG USERNAME=user

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    curl \
    wget \
    vim \
    nano \
    build-essential \
    sudo \
    x11-apps \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install specific packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    pip \
    swig \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

    
# Setup user configuration
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers 

USER $USERNAME

## Use bash instead of sh
SHELL ["/bin/bash", "-c"]

WORKDIR /reinforcement-learning-demos

