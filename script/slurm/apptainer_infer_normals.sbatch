#!/bin/bash
#SBATCH --job-name=mg-infer
#SBATCH --partition=gen-part
#SBATCH --qos=scavenger
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --output=%x-%j.out
#SBATCH --exclude=lse-hpcnode6
# -------------------- User configuration --------------------
# Update the paths below to match your cluster layout. You typically only
# need to change INPUT_DIR and OUTPUT_DIR between runs.
IMAGE_DIR=/projects/pointclouds/images
IMAGE_PATH=ykorny/marigold-normals:latest
CODE_DIR=/projects/pointclouds/Marigold
INPUT_DIR=/projects/pointclouds/input_images
OUTPUT_DIR=/projects/pointclouds/predictions
CHECKPOINTS_DIR=/projects/pointclouds/ckpt/

# Inference controls
DENOISE_STEPS=5
PROCESSING_RES=1024
ENSEMBLE_SIZE=1
HALF_PRECISION=false
USE_ASPECT_RATIO=true
MULTI_VIEW=true

# Additional HuggingFace auth (optional)
# export HF_HOME=/projects/pointclouds/.cache/huggingface

# -------------------- Derived paths --------------------
SCRIPT_PATH=$CODE_DIR/script/slurm
SIF=$IMAGE_DIR/marigold-normals.sif

mkdir -p "$IMAGE_DIR"
mkdir -p "$OUTPUT_DIR"

# -------------------- Build container image if missing --------------------
if [ ! -f "$SIF" ]; then
    srun --nodes=1 --ntasks=1 apptainer build "$SIF" docker://"$IMAGE_PATH"
fi

# -------------------- Compose inference command --------------------
CONTAINER_INPUT=/work/input
CONTAINER_OUTPUT=/work/output

CMD_ARGS=(
    --checkpoint "/ckpt/housediffusion"
    --input_rgb_dir "$CONTAINER_INPUT"
    --output_dir "$CONTAINER_OUTPUT"
    --denoise_steps "$DENOISE_STEPS"
    --processing_res "$PROCESSING_RES"
    --ensemble_size "$ENSEMBLE_SIZE"
)

if [ "$HALF_PRECISION" = "true" ]; then
    CMD_ARGS+=(--half_precision)
fi

if [ "$USE_ASPECT_RATIO" = "true" ]; then
    CMD_ARGS+=(--use_aspect_ratio)
fi

if [ "$MULTI_VIEW" = "true" ]; then
    CMD_ARGS+=(--multi_view)
fi

# -------------------- Launch inference --------------------
set -o allexport
if [ -f ~/.env ]; then
    source ~/.env
fi
set +o allexport

apptainer exec \
    --nv \
    --bind "$INPUT_DIR:$CONTAINER_INPUT" \
    --bind "$OUTPUT_DIR:$CONTAINER_OUTPUT" \
    --bind "$CODE_DIR:/app" \
    "$SIF" \
    python /app/script/normals/run.py "${CMD_ARGS[@]}"

