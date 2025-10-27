# syntax=docker/dockerfile:1
FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime

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
        && rm -rf /var/lib/apt/lists/*

# Caches (bind mount on HPC if desired)
ENV HF_HOME=/opt/cache/hf \
    TRANSFORMERS_CACHE=/opt/cache/hf \
    HUGGINGFACE_HUB_CACHE=/opt/cache/hf \
    PIP_NO_CACHE_DIR=1
RUN mkdir -p /opt/cache/hf

WORKDIR /app

# Install Python deps
COPY requirements.txt requirements+.txt requirements++.txt ./
RUN pip install -r requirements.txt -r requirements+.txt -r requirements++.txt

# Project code
COPY . .

ENV PYTHONPATH=/app

# Default to bash; override with `apptainer exec ... python ...`
ENTRYPOINT ["/bin/bash"] 