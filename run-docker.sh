#!/bin/bash
# Run Marigold training with Docker
# Loads environment variables from .env file

# Load environment variables from .env file
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
else
    echo "Error: .env file not found. Please create it from the template."
    exit 1
fi

# Validate required paths
if [ ! -d "$DATASET_PATH" ]; then
    echo "Error: DATASET_PATH not found: $DATASET_PATH"
    echo "Please update DATASET_PATH in .env file"
    exit 1
fi

if [ ! -d "$CHECKPOINT_PATH" ]; then
    echo "Error: CHECKPOINT_PATH not found: $CHECKPOINT_PATH"
    echo "Please update CHECKPOINT_PATH in .env file"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_PATH"

# Run Docker container with GPU support
docker run --gpus all \
    --rm \
    -it \
    -e BASE_DATA_DIR=/data \
    -e BASE_CKPT_DIR=/checkpoints \
    -v "$DATASET_PATH:/data:ro" \
    -v "$CHECKPOINT_PATH:/checkpoints:ro" \
    -v "$OUTPUT_PATH:/app/output" \
    -v "$HF_CACHE_PATH:/opt/cache/hf" \
    marigold:latest \
    python script/normals/train.py "$@"

