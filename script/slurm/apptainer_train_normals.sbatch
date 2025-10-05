#!/bin/bash
#SBATCH --job-name=marigold-normals
#SBATCH --partition=gpu
#SBATCH --qos=gpu:1
#SBATCH --output=%x-%j.out

# Load environment variables from .env file if it exists
if [ -f "$PWD/.env" ]; then
    echo "Loading environment variables from .env"
    export $(grep -v '^#' $PWD/.env | xargs)
else
    echo "Warning: .env file not found. Using default values."
fi

# Paths you may need to adjust
IMAGE_DIR=${IMAGE_DIR:-$PWD/images}
IMAGE_NAME=${IMAGE_NAME:-marigold:latest}
SIF=$IMAGE_DIR/${IMAGE_NAME//[:\/]/_}.sif
DATA_DIR=${DATASET_PATH:-/path/to/base_data_dir}
CKPT_DIR=${CHECKPOINT_PATH:-/path/to/base_ckpt_dir}
OUT_DIR=${OUTPUT_PATH:-$PWD/output}
CACHE_DIR=${HF_CACHE_PATH:-$PWD/.cache/hf}
CONFIG=${CONFIG:-config/train_marigold_normals.yaml}

# Environment variables to pass to container
export WANDB_API_KEY=${WANDB_API_KEY}
export HUGGINGFACE_HUB_TOKEN=${HUGGINGFACE_HUB_TOKEN}

echo "==============================================="
echo "Job Configuration:"
echo "  Image: $SIF"
echo "  Data Dir: $DATA_DIR"
echo "  Checkpoint Dir: $CKPT_DIR"
echo "  Output Dir: $OUT_DIR"
echo "  Cache Dir: $CACHE_DIR"
echo "  Config: $CONFIG"
echo "==============================================="

mkdir -p "$IMAGE_DIR"

# Build the SIF if not present (builds from local Docker image)
if [ ! -f "$SIF" ]; then
    echo "Building Apptainer image from Docker..."
    apptainer build "$SIF" docker-daemon://$IMAGE_NAME
fi

# Create necessary directories
mkdir -p "$CACHE_DIR" "$OUT_DIR"

# Run wandb login before training
echo "Logging into Weights & Biases..."
apptainer exec \
    --nv \
    --env WANDB_API_KEY=$WANDB_API_KEY \
    "$SIF" \
    wandb login $WANDB_API_KEY

# Run training
echo "Starting training..."
apptainer exec \
    --nv \
    --env WANDB_API_KEY=$WANDB_API_KEY \
    --env HUGGINGFACE_HUB_TOKEN=$HUGGINGFACE_HUB_TOKEN \
    --env BASE_DATA_DIR=/data \
    --env BASE_CKPT_DIR=/ckpt \
    --env HF_HOME=/cache/hf \
    --env TRANSFORMERS_CACHE=/cache/hf \
    --env HUGGINGFACE_HUB_CACHE=/cache/hf \
    --bind "$DATA_DIR:/data" \
    --bind "$CKPT_DIR:/ckpt" \
    --bind "$OUT_DIR:/work/out" \
    --bind "$CACHE_DIR:/cache/hf" \
    "$SIF" \
    python script/normals/train.py \
        --config "$CONFIG" \
        --base_data_dir /data \
        --base_ckpt_dir /ckpt \
        --output_dir /work/out \
        --add_datetime_prefix 