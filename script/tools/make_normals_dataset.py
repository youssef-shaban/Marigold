#!/usr/bin/env python3
import argparse
import os
import sys
import random
import pathlib
import numpy as np
from typing import List, Tuple, Dict
from PIL import Image
from tqdm import tqdm


def list_image_files(base_dir: str, exts: Tuple[str, ...]) -> List[str]:
    collected: List[str] = []
    for root, _, files in os.walk(base_dir):
        for f in files:
            if f.lower().endswith(exts):
                collected.append(os.path.join(root, f))
    collected.sort()
    return collected


def build_stem_to_relpath_map(files: List[str], rel_root: str) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for abs_path in files:
        rel_path = os.path.relpath(abs_path, rel_root)
        stem = pathlib.Path(rel_path).with_suffix("").as_posix()
        mapping[stem] = rel_path.replace(os.sep, "/")
    return mapping


def convert_normal_png_to_npy(png_path: str, npy_path: str) -> None:
    image = Image.open(png_path).convert("RGB")
    arr_uint8 = np.asarray(image)
    arr_float = arr_uint8.astype(np.float32) / 255.0 * 2.0 - 1.0
    normals_chw = np.transpose(arr_float, (2, 0, 1))
    norm = np.linalg.norm(normals_chw, axis=0, keepdims=True)
    norm = np.maximum(norm, 1e-6)
    normals_chw = normals_chw / norm
    normals_hwc = np.transpose(normals_chw, (1, 2, 0))
    os.makedirs(os.path.dirname(npy_path), exist_ok=True)
    np.save(npy_path, normals_hwc)


def write_split_file(pairs: List[Tuple[str, str]], save_path: str) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        for rgb_rel, normal_rel in pairs:
            f.write(f"{rgb_rel} {normal_rel}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare normals dataset: convert target PNG normals to NPY and create data_split files."
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        required=True,
        help="Root directory containing 'input/' (RGB) and 'target/' (normal PNGs).",
    )
    parser.add_argument(
        "--input_subdir",
        type=str,
        default="input",
        help="Subdirectory name under dataset_root for RGB images. Default: input",
    )
    parser.add_argument(
        "--target_subdir",
        type=str,
        default="target",
        help="Subdirectory name under dataset_root for normal images (PNG). Default: target",
    )
    parser.add_argument(
        "--output_normals_subdir",
        type=str,
        default="normals_npy",
        help="Subdirectory under dataset_root to write converted NPY normals. Default: normals_npy",
    )
    parser.add_argument(
        "--output_split_dir",
        type=str,
        default="data_split/custom_normals",
        help="Directory to write split files (train/val[/test].txt). Can be absolute or relative to CWD.",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="Validation split ratio in [0,1). Default: 0.1",
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.0,
        help="Test split ratio in [0,1). Default: 0.0 (no test split)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2024,
        help="Random seed for splitting. Default: 2024",
    )
    parser.add_argument(
        "--rgb_exts",
        type=str,
        default=".png,.jpg,.jpeg,.bmp",
        help="Comma-separated list of RGB file extensions. Default: .png,.jpg,.jpeg,.bmp",
    )
    parser.add_argument(
        "--normal_exts",
        type=str,
        default=".png",
        help="Comma-separated list of normal image extensions to convert. Default: .png",
    )

    args = parser.parse_args()

    dataset_root = os.path.abspath(args.dataset_root)
    input_dir = os.path.join(dataset_root, args.input_subdir)
    target_dir = os.path.join(dataset_root, args.target_subdir)
    output_normals_dir = os.path.join(dataset_root, args.output_normals_subdir)

    if not os.path.isdir(input_dir):
        print(f"Error: Input directory does not exist: {input_dir}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isdir(target_dir):
        print(f"Error: Target directory does not exist: {target_dir}", file=sys.stderr)
        sys.exit(1)

    if args.val_ratio < 0 or args.test_ratio < 0 or (args.val_ratio + args.test_ratio) >= 1.0:
        print("Error: val_ratio and test_ratio must be >=0 and sum to < 1.0", file=sys.stderr)
        sys.exit(1)

    rgb_exts = tuple([e.strip().lower() for e in args.rgb_exts.split(",") if e.strip()])
    normal_exts = tuple([e.strip().lower() for e in args.normal_exts.split(",") if e.strip()])

    rgb_files = list_image_files(input_dir, rgb_exts)
    normal_files = list_image_files(target_dir, normal_exts)

    rgb_map = build_stem_to_relpath_map(rgb_files, input_dir)
    normal_map = build_stem_to_relpath_map(normal_files, target_dir)

    common_stems = sorted(set(rgb_map.keys()) & set(normal_map.keys()))
    if len(common_stems) == 0:
        print("Error: No matching pairs found between input and target.", file=sys.stderr)
        sys.exit(1)

    converted_pairs: List[Tuple[str, str]] = []
    for stem in tqdm(common_stems, desc="Converting normals to NPY"):
        rgb_rel = os.path.join(args.input_subdir, rgb_map[stem]).replace(os.sep, "/")
        normal_png_rel = normal_map[stem]
        normal_png_abs = os.path.join(target_dir, normal_png_rel)
        npy_rel_under_output = pathlib.Path(normal_png_rel).with_suffix(".npy").as_posix()
        normal_npy_rel = os.path.join(args.output_normals_subdir, npy_rel_under_output).replace(os.sep, "/")
        normal_npy_abs = os.path.join(output_normals_dir, npy_rel_under_output)

        try:
            convert_normal_png_to_npy(normal_png_abs, normal_npy_abs)
            converted_pairs.append((rgb_rel, normal_npy_rel))
        except Exception as e:
            bad_name = os.path.basename(normal_png_abs)
            print(f"Error converting normal '{bad_name}': {e}. Skipping.", file=sys.stderr)
            continue

    random.seed(args.seed)
    random.shuffle(converted_pairs)

    n_total = len(converted_pairs)
    n_test = int(n_total * args.test_ratio)
    n_val = int(n_total * args.val_ratio)

    test_pairs = converted_pairs[:n_test]
    val_pairs = converted_pairs[n_test : n_test + n_val]
    train_pairs = converted_pairs[n_test + n_val :]

    write_split_file(train_pairs, os.path.join(args.output_split_dir, "train.txt"))
    write_split_file(val_pairs, os.path.join(args.output_split_dir, "val.txt"))
    if n_test > 0:
        write_split_file(test_pairs, os.path.join(args.output_split_dir, "test.txt"))

    # Visualization subset from validation
    n_vis = min(20, len(val_pairs))
    if n_vis > 0:
        vis_pairs = random.sample(val_pairs, n_vis)
        write_split_file(vis_pairs, os.path.join(args.output_split_dir, "vis.txt"))

    print(f"Done. Total pairs: {n_total}; train: {len(train_pairs)}, val: {len(val_pairs)}, test: {len(test_pairs)}")
    if n_vis > 0:
        print(f"Visualization subset: {n_vis} samples written to vis.txt")
    print(f"Split files written to: {os.path.abspath(args.output_split_dir)}")
    print(f"Converted NPY normals written under: {output_normals_dir}")


if __name__ == "__main__":
    main() 