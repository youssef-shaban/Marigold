#!/bin/bash
#SBATCH --job-name=mg-normals-train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out

# Paths you may need to adjust
IMAGE_DIR=$PWD/images
IMAGE_NAME=marigold-normals:cu124
SIF=$IMAGE_DIR/marigold-normals.sif
DATA_DIR=/path/to/base_data_dir
CKPT_DIR=/path/to/base_ckpt_dir
OUT_DIR=$PWD/output
CONFIG=config/train_marigold_normals.yaml

module purge
module load apptainer

mkdir -p "$IMAGE_DIR"

# Build the SIF if not present (needs Docker registry access on login/head node)
if [ ! -f "$SIF" ]; then
    apptainer build "$SIF" docker-daemon://$IMAGE_NAME
fi

# Bind mounts: data, checkpoints, cache, output
CACHE_DIR=$PWD/.cache/hf
mkdir -p "$CACHE_DIR" "$OUT_DIR"

export TRANSFORMERS_CACHE=/cache/hf
export HUGGINGFACE_HUB_CACHE=/cache/hf
export HF_HOME=/cache/hf

apptainer exec \
    --nv \
    --bind "$DATA_DIR:/data" \
    --bind "$CKPT_DIR:/ckpt" \
    --bind "$CACHE_DIR:/cache/hf" \
    --bind "$OUT_DIR:/work/out" \
    "$SIF" \
    python script/normals/train.py \
        --config "$CONFIG" \
        --base_data_dir /data \
        --base_ckpt_dir /ckpt \
        --output_dir /work/out \
        --add_datetime_prefix 