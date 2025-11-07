#!/bin/bash
echo "TRAINING IS STARTING NOW"
cd /app

nvidia-smi

CONFIG=config/train_marigold_normals.yaml
##CKPT_PATH="/work/out/25_10_27-16_26_40-train_marigold_normals/checkpoint/latest"
wandb login

cat "$CONFIG"
rm -r /tmp/Marigold_data/

pip uninstall -y transformers diffusers
pip install transformers==4.49 diffusers==0.32.2

python script/normals/train.py \
##      --resume_run "$CKPT_PATH" \
        --base_data_dir /data \
        --base_ckpt_dir /ckpt \
        --output_dir /work/out \
        --do_not_copy_data  