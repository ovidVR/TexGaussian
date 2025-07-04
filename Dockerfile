FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel

ENV TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6;8.9;9.0"
# Install system dependencies
ENV TZ=Europe/Paris
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt update -y \
    && apt upgrade -y \
    && apt install -y \
    sudo \
    curl \
    git \
    git-lfs \
    htop \
    nano \
    locales \
    libxxf86vm-dev \
    wget \
    && locale-gen en_US.UTF-8


RUN pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

WORKDIR /app