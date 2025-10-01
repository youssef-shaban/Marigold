### Container authentication: Weights & Biases (W&B) and Hugging Face (HF)

This guide shows secure, repeatable ways to make your Docker/Apptainer training runs authenticate to W&B and the Hugging Face Hub without baking secrets into images.

- Tokens needed
  - **W&B**: `WANDB_API_KEY` (from your W&B account settings)
  - **Hugging Face**: `HUGGINGFACE_HUB_TOKEN` (a.k.a. HF access token)
- Never commit tokens to Git. Prefer environment variables or mounted, ignored files (e.g., `.env`).

### Option A: Local Docker (recommended)

- Put your tokens in a local `.env` file (not committed):
```bash
cat > .env <<'EOF'
WANDB_API_KEY=xxxxxx
# Optional: set project defaults
WANDB_ENTITY=my_team
WANDB_PROJECT=marigold-normals

# HF hub token
HUGGINGFACE_HUB_TOKEN=hf_xxxxxxxxxxxxxxxxx
# Optional: run without WANDB
# WANDB_MODE=offline
EOF
```

- Run the container with `--env-file` and bind caches/data:
```bash
docker run --rm --gpus all -it \
  --env-file .env \
  -e HF_HOME=/cache/hf -e TRANSFORMERS_CACHE=/cache/hf -e HUGGINGFACE_HUB_CACHE=/cache/hf \
  -v $PWD/.cache/hf:/cache/hf \
  -v /path/to/dataset:/data \
  -v /path/to/checkpoints:/ckpt \
  -v $PWD/out:/work/out \
  marigold-normals:cu124 \
  python script/normals/train.py \
    --config config/train_marigold_normals.yaml \
    --base_data_dir /data \
    --base_ckpt_dir /ckpt \
    --output_dir /work/out \
    --add_datetime_prefix
```

- One-time logins (alternative): You can also login interactively inside the container, but prefer env vars for non-interactive runs.
```bash
# Inside the container (not needed if you use env vars above)
wandb login $WANDB_API_KEY
huggingface-cli login --token $HUGGINGFACE_HUB_TOKEN --add-to-git-credential
```

### Option B: Slurm with Apptainer

Use environment variables in your sbatch script and pass/bind them into the container.

- Add to `script/slurm/apptainer_train_normals.sbatch` (edit these lines):
```bash
# --- User tokens (do NOT commit real tokens) ---
export WANDB_API_KEY=xxxxxx
export WANDB_ENTITY=my_team
export WANDB_PROJECT=marigold-normals
export HUGGINGFACE_HUB_TOKEN=hf_xxxxxxxxxxxxxxxxx
# Optional: run W&B offline on clusters without internet
# export WANDB_MODE=offline

# Caches (already present in the sample script)
export HF_HOME=/cache/hf
export TRANSFORMERS_CACHE=/cache/hf
export HUGGINGFACE_HUB_CACHE=/cache/hf

apptainer exec \
  --nv \
  --bind "$DATA_DIR:/data" \
  --bind "$CKPT_DIR:/ckpt" \
  --bind "$CACHE_DIR:/cache/hf" \
  --bind "$OUT_DIR:/work/out" \
  "$SIF" \
  env \
    WANDB_API_KEY="$WANDB_API_KEY" \
    WANDB_ENTITY="$WANDB_ENTITY" \
    WANDB_PROJECT="$WANDB_PROJECT" \
    HUGGINGFACE_HUB_TOKEN="$HUGGINGFACE_HUB_TOKEN" \
    HF_HOME="$HF_HOME" TRANSFORMERS_CACHE="$TRANSFORMERS_CACHE" HUGGINGFACE_HUB_CACHE="$HUGGINGFACE_HUB_CACHE" \
    python script/normals/train.py \
      --config "$CONFIG" \
      --base_data_dir /data \
      --base_ckpt_dir /ckpt \
      --output_dir /work/out \
      --add_datetime_prefix
```

- Notes
  - Apptainer inherits environment by default; explicitly passing with `env` makes it clear and auditable.
  - Keep tokens in cluster key/value stores or job secrets if available (site-specific), or load from a protected file (`source /secure/path/tokens.sh`).

### Verify authentication inside the container

- Hugging Face:
```bash
python - <<'PY'
from huggingface_hub import whoami
import os
print("HF token present:", bool(os.getenv("HUGGINGFACE_HUB_TOKEN")))
print("HF whoami:", whoami())
PY
```

- W&B:
```bash
python - <<'PY'
import os, wandb
print("WANDB_API_KEY present:", bool(os.getenv("WANDB_API_KEY")))
# Start a test run (offline if WANDB_MODE=offline)
run = wandb.init(project=os.getenv("WANDB_PROJECT", "marigold"))
print("Run URL:", run.url)
run.finish()
PY
```

### Good practices

- **Never** bake tokens into images or commit them to Git.
- Use env vars + bind-mounted cache dirs so downloads (e.g., models from HF) are reused across runs.
- Prefer `WANDB_MODE=offline` on clusters without outbound internet; you can later sync with `wandb sync`.
- Scope your HF token to only the permissions you need (read-only is often enough for inference/training from public models).

### Troubleshooting

- Missing token (HF): you may see 401/403 when downloading models. Ensure `HUGGINGFACE_HUB_TOKEN` is exported and the container sees it.
- W&B disabled: set `--no_wandb` on training or `WANDB_MODE=offline` if you cannot reach W&B.
- Cache permissions: if caches are shared, ensure the user inside the container can write to the bound cache directory. 