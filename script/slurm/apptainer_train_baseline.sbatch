#!/bin/bash
#SBATCH --job-name=mg-baseline
#SBATCH --partition=gen-part
#SBATCH --qos=gpu1-32h
#SBATCH --gpus=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --output=%x-%j.out

# Paths you may need to adjust
IMAGE_DIR=/projects/pointclouds/images
IMAGE_PATH=ykorny/marigold-normals:latest
SCRIPT_PATH=/projects/pointclouds/Marigold/script/slurm/
SIF=$IMAGE_DIR/marigold-normals.sif
DATA_DIR=/projects/pointclouds/house_dataset/
OUT_DIR=/projects/pointclouds/output
CODE_DIR=/projects/pointclouds/Marigold
mkdir -p "$IMAGE_DIR"

# Build the SIF if not present (needs Docker registry access on login/head node)
if [ ! -f "$SIF" ]; then
    apptainer build "$SIF" docker://"$IMAGE_PATH"
fi

set -o allexport
source ~/.env
set +o allexport

apptainer exec  \
        --nv \
       --bind "$DATA_DIR:/data" \
       --bind "$OUT_DIR:/work/out" \
       --bind "$CODE_DIR:/app" \
       --bind "$SCRIPT_PATH:/script" \
       "$SIF" \
       /app/script/slurm/start_train_baseline.sh

