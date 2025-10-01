#!/usr/bin/env python3
import argparse
import os
import sys
from huggingface_hub import snapshot_download

DEFAULT_ALLOW = [
    "model_index.json",
    "unet/*",
    "vae/*",
    "text_encoder/*",
    "tokenizer/*",
    "scheduler/*",
]
DEFAULT_IGNORE = ["*.ckpt", "*.bin", "*.fb16.safetensors"]


def main():
    parser = argparse.ArgumentParser(
        description="Download minimal Diffusers files from a Hugging Face repo."
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default="stabilityai/stable-diffusion-2",
        help="Hugging Face repo id. Default: stabilityai/stable-diffusion-2",
    )
    parser.add_argument(
        "--dest_root",
        type=str,
        required=True,
        help="Destination root directory (e.g., your base_ckpt_dir).",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="stable-diffusion-2",
        help="Subfolder name to create under dest_root. Default: stable-diffusion-2",
    )
    parser.add_argument(
        "--allow_patterns",
        type=str,
        nargs="*",
        default=DEFAULT_ALLOW,
        help="Files/patterns to allow. Defaults to a minimal Diffusers set.",
    )
    parser.add_argument(
        "--ignore_patterns",
        type=str,
        nargs="*",
        default=DEFAULT_IGNORE,
        help="Glob patterns to ignore (e.g., legacy .ckpt/.bin).",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Optional repo revision (branch/tag/commit).",
    )

    args = parser.parse_args()

    dest_root = os.path.abspath(args.dest_root)
    local_dir = os.path.join(dest_root, args.model_name)
    os.makedirs(local_dir, exist_ok=True)

    token_present = bool(os.getenv("HUGGINGFACE_HUB_TOKEN"))
    if not token_present:
        print(
            "Warning: HUGGINGFACE_HUB_TOKEN not set. Private models may fail to download.",
            file=sys.stderr,
        )

    print(f"Repo: {args.repo_id}")
    print(f"Downloading to: {local_dir}")
    print(f"Allow: {args.allow_patterns}")
    print(f"Ignore: {args.ignore_patterns}")

    snapshot_download(
        repo_id=args.repo_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        allow_patterns=args.allow_patterns,
        ignore_patterns=args.ignore_patterns,
        revision=args.revision,
    )

    print("Done.")
    print(f"Model available at: {local_dir}")


if __name__ == "__main__":
    main() 