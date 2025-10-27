#!/bin/bash
echo "TRAINING IS STARTING NOW"
cd /app
CONFIG=config/train_marigold_normals.yaml
wandb login


rm -r /tmp/Marigold_data/

python script/normals/train.py \
        --config "$CONFIG" \
        --base_data_dir /data \
        --base_ckpt_dir /ckpt \
        --output_dir /work/out \
        --add_datetime_prefix