#!/bin/bash
echo "BASELINE TRAINING IS STARTING NOW"
cd /app

nvidia-smi

CONFIG=config/train_baseline_normals.yaml

wandb login

cat "$CONFIG"
rm -rf /tmp/Marigold_data/

pip uninstall -y transformers diffusers
pip install transformers==4.49 diffusers==0.32.2 timm==0.9.16

python script/baseline/train.py \
        --base_data_dir /data \
        --output_dir /work/out \
        --add_datetime_prefix --do_not_copy_data

