# Docker Setup for Marigold Training

Simple Docker setup that works with both Docker and Apptainer/Slurm.

## Step 1: Configure Environment

Create `.env` file from template and edit with your paths:

```bash
cp env.example .env
```

Then edit `.env`:
```bash
# .env
DATASET_PATH=/path/to/your/dataset
CHECKPOINT_PATH=/home/youssef/houseDiffusion/Marigold/checkpoints
HF_CACHE_PATH=${HOME}/.cache/huggingface
OUTPUT_PATH=${PWD}/output
```

## Step 2: Build Docker Image

```bash
./build-docker.sh
```

Or manually:
```bash
docker build -t marigold:latest .
```

## Step 3: Run Training with Docker

```bash
./run-docker.sh --config config/train_marigold_normals.yaml
```

Or manually:
```bash
# Load environment variables
source .env

# Run training
docker run --gpus all --rm -it \
    -e BASE_DATA_DIR=/data \
    -e BASE_CKPT_DIR=/checkpoints \
    -v "$DATASET_PATH:/data:ro" \
    -v "$CHECKPOINT_PATH:/checkpoints:ro" \
    -v "$OUTPUT_PATH:/app/output" \
    -v "$HF_CACHE_PATH:/opt/cache/hf" \
    marigold:latest \
    python script/normals/train.py --config config/train_marigold_normals.yaml
```

## Common Commands

### Test GPU availability:
```bash
source .env
docker run --gpus all --rm \
    -e BASE_DATA_DIR=/data \
    -e BASE_CKPT_DIR=/checkpoints \
    marigold:latest \
    python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### Interactive shell:
```bash
source .env
docker run --gpus all --rm -it \
    -e BASE_DATA_DIR=/data \
    -e BASE_CKPT_DIR=/checkpoints \
    -v "$DATASET_PATH:/data:ro" \
    -v "$CHECKPOINT_PATH:/checkpoints:ro" \
    -v "$OUTPUT_PATH:/app/output" \
    marigold:latest \
    /bin/bash
```

### Resume training:
```bash
./run-docker.sh --resume_run /app/output/previous_run/checkpoint/latest.ckpt
```

### Training without wandb:
```bash
./run-docker.sh --config config/train_marigold_normals.yaml --no_wandb
```

## Environment Variables

The container uses these environment variables (automatically set):
- `BASE_DATA_DIR=/data` - mounted from `$DATASET_PATH`
- `BASE_CKPT_DIR=/checkpoints` - mounted from `$CHECKPOINT_PATH`

## Directory Mapping

```
Host                    →  Container
──────────────────────────────────────────
$DATASET_PATH           →  /data (read-only)
$CHECKPOINT_PATH        →  /checkpoints (read-only)
$OUTPUT_PATH            →  /app/output
$HF_CACHE_PATH          →  /opt/cache/hf
```

## Next Steps: Apptainer/Slurm

After testing with Docker, you can convert the Docker image to Apptainer format for use with Slurm (see SLURM_SETUP.md).

