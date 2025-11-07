# syntax=docker/dockerfile:1
FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-devel

# ARG DEBIAN_FRONTEND=noninteractive

# System deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        ffmpeg \
        libgl1 \
        rsync \
        libglib2.0-0 \
        ca-certificates \
        g++ \ 
        pkg-config \
        && rm -rf /var/lib/apt/lists/*

# Caches (bind mount on HPC if desired)
ENV HF_HOME=/opt/cache/hf \
    TRANSFORMERS_CACHE=/opt/cache/hf \
    HUGGINGFACE_HUB_CACHE=/opt/cache/hf \
    PIP_NO_CACHE_DIR=1
RUN mkdir -p /opt/cache/hf

WORKDIR /app

# CRITICAL FOR CROSS-GPU BUILDS:
# 8.6 = RTX 3080 My PC
# 8.9 = L40 (Ada Lovelace) Slurm Cluster
# +PTX = Future-proofing for newer GPUs
ENV FORCE_CUDA=1
ENV TORCH_CUDA_ARCH_LIST="8.6 8.9+PTX"

# Install Python deps
COPY requirements.txt requirements+.txt requirements++.txt ./
RUN pip install -r requirements.txt -r requirements+.txt -r requirements++.txt
# Install PyTorch3D (requires some specific build wheels and dependencies)
RUN pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"


ENV PYTHONPATH=/app

# Default to bash; override with `apptainer exec ... python ...`
ENTRYPOINT ["/bin/bash"] 