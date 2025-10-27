#!/usr/bin/env python3
import argparse
import os
import sys
import random
import pathlib
from typing import List, Tuple, Dict
from collections import defaultdict
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


def extract_sample_name(stem: str) -> str:
    """
    Extract the sample name from a stem like 'input/3192_010'.
    For files named {sample_name}_{sample_num}.ext, returns the sample_name part.
    """
    basename = os.path.basename(stem)
    # Split by underscore and take all parts except the last one
    parts = basename.rsplit('_', 1)
    if len(parts) == 2:
        return parts[0]
    # If no underscore, return the whole basename
    return basename


def write_split_file(pairs: List[Tuple[str, str]], save_path: str) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        for rgb_rel, normal_rel in pairs:
            f.write(f"{rgb_rel} {normal_rel}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare normals dataset: match RGB and NPY normals pairs and create data_split files."
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        required=True,
        help="Root directory containing 'input/' (RGB) and 'target/' (normal NPY files).",
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
        default="target_npy",
        help="Subdirectory name under dataset_root for normal NPY files. Default: target",
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
        default=".npy",
        help="Comma-separated list of normal file extensions. Default: .npy",
    )

    args = parser.parse_args()

    dataset_root = os.path.abspath(args.dataset_root)
    input_dir = os.path.join(dataset_root, args.input_subdir)
    target_dir = os.path.join(dataset_root, args.target_subdir)

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

    # Group pairs by sample name to avoid data leakage
    # For files like "3192_010.png", group by "3192"
    sample_groups: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    
    for stem in tqdm(common_stems, desc="Building pairs"):
        rgb_rel = os.path.join(args.input_subdir, rgb_map[stem]).replace(os.sep, "/")
        normal_rel = os.path.join(args.target_subdir, normal_map[stem]).replace(os.sep, "/")
        
        # Extract sample name (e.g., "3192" from "3192_010")
        sample_name = extract_sample_name(stem)
        sample_groups[sample_name].append((rgb_rel, normal_rel))
    
    # Get list of unique sample names and shuffle them
    sample_names = sorted(sample_groups.keys())
    random.seed(args.seed)
    random.shuffle(sample_names)
    
    # Split by sample names (not individual pairs) to avoid data leakage
    n_samples = len(sample_names)
    n_test_samples = int(n_samples * args.test_ratio)
    n_val_samples = int(n_samples * args.val_ratio)
    
    test_sample_names = sample_names[:n_test_samples]
    val_sample_names = sample_names[n_test_samples : n_test_samples + n_val_samples]
    train_sample_names = sample_names[n_test_samples + n_val_samples :]
    
    # Collect all pairs for each split
    test_pairs = []
    for sample_name in test_sample_names:
        test_pairs.extend(sample_groups[sample_name])
    
    val_pairs = []
    for sample_name in val_sample_names:
        val_pairs.extend(sample_groups[sample_name])
    
    train_pairs = []
    for sample_name in train_sample_names:
        train_pairs.extend(sample_groups[sample_name])

    write_split_file(train_pairs, os.path.join(args.output_split_dir, "train.txt"))
    write_split_file(val_pairs, os.path.join(args.output_split_dir, "val.txt"))
    if len(test_pairs) > 0:
        write_split_file(test_pairs, os.path.join(args.output_split_dir, "test.txt"))

    # Visualization subset from validation
    n_vis = min(20, len(val_pairs))
    if n_vis > 0:
        vis_pairs = random.sample(val_pairs, n_vis)
        write_split_file(vis_pairs, os.path.join(args.output_split_dir, "vis.txt"))

    n_total_pairs = len(train_pairs) + len(val_pairs) + len(test_pairs)
    print(f"\nDone!")
    print(f"Unique samples: {n_samples} (train: {len(train_sample_names)}, val: {len(val_sample_names)}, test: {len(test_sample_names)})")
    print(f"Total pairs: {n_total_pairs} (train: {len(train_pairs)}, val: {len(val_pairs)}, test: {len(test_pairs)})")
    print(f"Average variations per sample: {n_total_pairs / n_samples:.2f}")
    if n_vis > 0:
        print(f"Visualization subset: {n_vis} samples written to vis.txt")
    print(f"Split files written to: {os.path.abspath(args.output_split_dir)}")
    print(f"\nNote: Samples with same name (e.g., '3192_010' and '3192_005') are kept in the same split to prevent data leakage.")


if __name__ == "__main__":
    main() 