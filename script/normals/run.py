# Copyright 2023-2025 Marigold Team, ETH Zürich. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# --------------------------------------------------------------------------
# More information about Marigold:
#   https://marigoldmonodepth.github.io
#   https://marigoldcomputervision.github.io
# Efficient inference pipelines are now part of diffusers:
#   https://huggingface.co/docs/diffusers/using-diffusers/marigold_usage
#   https://huggingface.co/docs/diffusers/api/pipelines/marigold
# Examples of trained models and live demos:
#   https://huggingface.co/prs-eth
# Related projects:
#   https://rollingdepth.github.io/
#   https://marigolddepthcompletion.github.io/
# Citation (BibTeX):
#   https://github.com/prs-eth/Marigold#-citation
# If you find Marigold useful, we kindly ask you to cite our papers.
# --------------------------------------------------------------------------

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import argparse
import logging
import numpy as np
import os
import torch
from PIL import Image
from glob import glob
from tqdm.auto import tqdm

from marigold import MarigoldNormalsPipeline, MarigoldNormalsOutput

EXTENSION_LIST = [".jpg", ".jpeg", ".png"]
MASK_EXTENSION_LIST = [".png", ".jpg", ".jpeg"]


def tile_image_2x2(image: Image.Image) -> Image.Image:
    width, height = image.size
    tiled = Image.new("RGB", (width * 2, height * 2))
    tiled.paste(image, (0, 0))
    tiled.paste(image, (width, 0))
    tiled.paste(image, (0, height))
    tiled.paste(image, (width, height))
    return tiled


def pad_to_square(image: Image.Image, fill_color=(0, 0, 0)) -> Image.Image:
    width, height = image.size
    if width == height:
        return image
    target_size = max(width, height)
    padded = Image.new("RGB", (target_size, target_size), fill_color)
    left = (target_size - width) // 2
    top = (target_size - height) // 2
    padded.paste(image, (left, top))
    return padded


def get_pil_resample(method: str) -> int:
    if method == "bicubic":
        return Image.BICUBIC
    if method == "nearest":
        return Image.NEAREST
    return Image.BILINEAR


if "__main__" == __name__:
    logging.basicConfig(level=logging.INFO)

    # -------------------- Arguments --------------------
    parser = argparse.ArgumentParser(
        description="Marigold : Surface Normals Estimation : Multi-image Inference"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="prs-eth/marigold-normals-v1-1",
        help="Checkpoint path or hub name.",
    )
    parser.add_argument(
        "--input_rgb_dir",
        type=str,
        required=True,
        help="Path containing `image/` and `roof_intuitive_mask/` subfolders.",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Output directory."
    )
    parser.add_argument(
        "--denoise_steps",
        type=int,
        default=None,
        help="Diffusion denoising steps, more steps results in higher accuracy but slower inference speed. If set to "
        "`None`, default value will be read from checkpoint.",
    )
    parser.add_argument(
        "--processing_res",
        type=int,
        default=None,
        help="Resolution to which the input is resized before performing estimation. `0` uses the original input "
        "resolution; `None` resolves the best default from the model checkpoint. Default: `None`",
    )
    parser.add_argument(
        "--ensemble_size",
        type=int,
        default=1,
        help="Number of predictions to be ensembled. Default: `1`.",
    )
    parser.add_argument(
        "--half_precision",
        "--fp16",
        action="store_true",
        help="Run with half-precision (16-bit float), might lead to suboptimal result.",
    )
    parser.add_argument(
        "--output_processing_res",
        action="store_true",
        help="Setting this flag will output the result at the effective value of `processing_res`, otherwise the "
        "output will be resized to the input resolution.",
    )
    parser.add_argument(
        "--resample_method",
        choices=["bilinear", "bicubic", "nearest"],
        default="bilinear",
        help="Resampling method used to resize images and predictions. This can be one of `bilinear`, `bicubic` or "
        "`nearest`. Default: `bilinear`",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Reproducibility seed. Set to `None` for randomized inference. Default: `None`",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=0,
        help="Inference batch size. Default: 0 (will be set automatically).",
    )
    parser.add_argument(
        "--apple_silicon",
        action="store_true",
        help="Use Apple Silicon for faster inference (subject to availability).",
    )
    parser.add_argument(
        "--use_aspect_ratio",
        action="store_true",
        help="Enable aspect-ratio conditioning when supported by the model.",
    )
    parser.add_argument(
        "--multi_view",
        action="store_true",
        help="Tile the masked input 2x2 for multi-view inference.",
    )

    args = parser.parse_args()

    checkpoint_path = args.checkpoint
    input_rgb_dir = args.input_rgb_dir
    output_dir = args.output_dir

    denoise_steps = args.denoise_steps
    ensemble_size = args.ensemble_size
    if ensemble_size > 15:
        logging.warning("Running with large ensemble size will be slow.")
    half_precision = args.half_precision

    processing_res = args.processing_res
    match_input_res = not args.output_processing_res
    if 0 == processing_res and match_input_res is False:
        logging.warning(
            "Processing at native resolution without resizing output might NOT lead to exactly the same resolution, "
            "due to the padding and pooling properties of conv layers."
        )
    resample_method = args.resample_method

    seed = args.seed
    batch_size = args.batch_size
    apple_silicon = args.apple_silicon
    if apple_silicon and 0 == batch_size:
        batch_size = 1  # set default batchsize
    use_aspect_ratio = args.use_aspect_ratio
    multi_view = args.multi_view

    # -------------------- Preparation --------------------
    # Output directories
    output_dir_img = os.path.join(output_dir, "normals_png")
    output_dir_latent = os.path.join(output_dir, "normals_latent")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir_img, exist_ok=True)
    os.makedirs(output_dir_latent, exist_ok=True)
    logging.info(f"output dir = {output_dir}")

    # -------------------- Device --------------------
    if apple_silicon:
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
            logging.warning("MPS is not available. Running on CPU will be slow.")
    else:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
            logging.warning("CUDA is not available. Running on CPU will be slow.")
    logging.info(f"device = {device}")

    # -------------------- Data --------------------
    image_dir = os.path.join(input_rgb_dir, "image")
    mask_dir = os.path.join(input_rgb_dir, "roof_intuitive_mask")
    if not os.path.isdir(image_dir):
        logging.error(f"Expected image directory at '{image_dir}'.")
        exit(1)
    if not os.path.isdir(mask_dir):
        logging.error(f"Expected mask directory at '{mask_dir}'.")
        exit(1)

    rgb_filename_list = glob(os.path.join(image_dir, "*"))
    rgb_filename_list = [
        f
        for f in rgb_filename_list
        if os.path.splitext(f)[1].lower() in EXTENSION_LIST
    ]
    rgb_filename_list = sorted(rgb_filename_list)
    n_images = len(rgb_filename_list)
    if n_images > 0:
        logging.info(f"Found {n_images} images")
    else:
        logging.error(f"No image found in '{image_dir}'")
        exit(1)

    mask_lookup = {}
    for mask_path in glob(os.path.join(mask_dir, "*")):
        ext = os.path.splitext(mask_path)[1].lower()
        if ext in MASK_EXTENSION_LIST:
            mask_lookup[os.path.splitext(os.path.basename(mask_path))[0]] = mask_path

    # -------------------- Model --------------------
    if half_precision:
        dtype = torch.float16
        variant = "fp16"
        logging.info(
            f"Running with half precision ({dtype}), might lead to suboptimal result."
        )
    else:
        dtype = torch.float32
        variant = None

    pipe: MarigoldNormalsPipeline = MarigoldNormalsPipeline.from_pretrained(
        checkpoint_path, variant=variant, torch_dtype=dtype
    )

    try:
        pipe.enable_xformers_memory_efficient_attention()
    except ImportError:
        pass  # run without xformers

    pipe = pipe.to(device)

    if use_aspect_ratio:
        pipe.enable_aspect_ratio_conditioning(checkpoint_path)

    logging.info("Loaded normals pipeline")

    # Print out config
    logging.info(
        f"Inference settings: checkpoint = `{checkpoint_path}`, "
        f"with denoise_steps = {denoise_steps or pipe.default_denoising_steps}, "
        f"ensemble_size = {ensemble_size}, "
        f"processing resolution = {processing_res or pipe.default_processing_resolution}, "
        f"seed = {seed}; "
    )

    # -------------------- Inference and saving --------------------
    with torch.no_grad():
        os.makedirs(output_dir, exist_ok=True)

        for rgb_path in tqdm(
            rgb_filename_list, desc="Surface Normals Inference", leave=True
        ):
            # Read input image
            input_image = Image.open(rgb_path).convert("RGB")
            aspect_ratio_value = None
            if use_aspect_ratio:
                aspect_ratio_value = (
                    float(input_image.width) / float(input_image.height)
                    if input_image.height != 0
                    else 1.0
                )

            # Apply mask
            base_name = os.path.splitext(os.path.basename(rgb_path))[0]
            mask_path = mask_lookup.get(base_name)
            if mask_path is None:
                logging.warning(
                    f"No mask found for '{rgb_path}'. Skipping this image."
                )
                continue

            mask_image = Image.open(mask_path).convert("L")
            if mask_image.size != input_image.size:
                mask_image = mask_image.resize(input_image.size, Image.NEAREST)

            mask_np = np.array(mask_image)
            mask_binary = mask_np > 127
            if not mask_binary.any():
                logging.warning(
                    f"Mask for '{rgb_path}' is empty after thresholding. Skipping."
                )
                continue

            image_np = np.array(input_image)
            image_np[~mask_binary] = 0
            input_image = Image.fromarray(image_np)
            # Aspect ratio remains the same after masking

            input_image = pad_to_square(input_image)

            if multi_view:
                input_image = tile_image_2x2(input_image)

            # Random number generator
            if seed is None:
                generator = None
            else:
                generator = torch.Generator(device=device)
                generator.manual_seed(seed)

            # Perform inference
            pipe_out: MarigoldNormalsOutput = pipe(
                input_image,
                denoising_steps=denoise_steps,
                ensemble_size=ensemble_size,
                processing_res=processing_res,
                match_input_res=match_input_res,
                batch_size=batch_size,
                show_progress_bar=True,
                resample_method=resample_method,
                generator=generator,
                aspect_ratio=aspect_ratio_value,
            )

            normals_img: Image.Image = pipe_out.normals_img
            normals_latent: np.ndarray = pipe_out.normals_latent  # [4,h,w]

            # Save as npy
            rgb_name_base = os.path.splitext(os.path.basename(rgb_path))[0]

            img_save_path = os.path.join(output_dir_img, f"{rgb_name_base}.png")
            if os.path.exists(img_save_path):
                logging.warning(f"Existing file: '{img_save_path}' will be overwritten")
            normals_img.save(img_save_path)

            latent_save_path = os.path.join(output_dir_latent, f"{rgb_name_base}.npy")
            if os.path.exists(latent_save_path):
                logging.warning(
                    f"Existing file: '{latent_save_path}' will be overwritten"
                )
            np.save(latent_save_path, normals_latent)
