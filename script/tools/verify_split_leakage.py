#!/usr/bin/env python3
"""
Verify that there's no data leakage between train/val/test splits.
Check if the same sample name appears in multiple splits.
For files like "3192_010.png", the sample name is "3192".
"""
import argparse
import os
from collections import defaultdict
from typing import Dict, List, Set


def extract_sample_name(file_path: str) -> str:
    """
    Extract the sample name from a file path like 'input/3192_010.png'.
    Returns the sample_name part (e.g., "3192").
    """
    basename = os.path.basename(file_path)
    # Remove extension
    name_without_ext = os.path.splitext(basename)[0]
    # Split by underscore and take all parts except the last one
    parts = name_without_ext.rsplit('_', 1)
    if len(parts) == 2:
        return parts[0]
    # If no underscore, return the whole name
    return name_without_ext


def read_split_file(file_path: str) -> List[str]:
    """Read a split file and return list of input file paths."""
    if not os.path.exists(file_path):
        return []
    
    samples = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                # Extract the first part (input file path)
                parts = line.split()
                if parts:
                    samples.append(parts[0])
    return samples


def main():
    parser = argparse.ArgumentParser(
        description="Verify data split integrity - check for sample name leakage between splits."
    )
    parser.add_argument(
        "--split_dir",
        type=str,
        required=True,
        help="Directory containing train.txt, val.txt, and optionally test.txt",
    )
    
    args = parser.parse_args()
    
    # Read split files
    train_file = os.path.join(args.split_dir, "train.txt")
    val_file = os.path.join(args.split_dir, "val.txt")
    test_file = os.path.join(args.split_dir, "test.txt")
    
    train_samples = read_split_file(train_file)
    val_samples = read_split_file(val_file)
    test_samples = read_split_file(test_file)
    
    print(f"Train samples: {len(train_samples)}")
    print(f"Val samples: {len(val_samples)}")
    print(f"Test samples: {len(test_samples)}")
    print()
    
    # Extract sample names
    train_names = {extract_sample_name(s) for s in train_samples}
    val_names = {extract_sample_name(s) for s in val_samples}
    test_names = {extract_sample_name(s) for s in test_samples}
    
    print(f"Unique train sample names: {len(train_names)}")
    print(f"Unique val sample names: {len(val_names)}")
    print(f"Unique test sample names: {len(test_names)}")
    print()
    
    # Check for overlaps
    train_val_overlap = train_names & val_names
    train_test_overlap = train_names & test_names
    val_test_overlap = val_names & test_names
    
    leakage_found = False
    
    if train_val_overlap:
        leakage_found = True
        print(f"⚠️  WARNING: Found {len(train_val_overlap)} sample names in both TRAIN and VAL!")
        print(f"Examples: {sorted(list(train_val_overlap))[:10]}")
        print()
        
        # Show specific examples
        print("Detailed examples:")
        for sample_name in sorted(list(train_val_overlap))[:5]:
            train_files = [s for s in train_samples if extract_sample_name(s) == sample_name]
            val_files = [s for s in val_samples if extract_sample_name(s) == sample_name]
            print(f"  Sample '{sample_name}':")
            print(f"    Train: {train_files[:3]}")
            print(f"    Val:   {val_files[:3]}")
        print()
    
    if train_test_overlap:
        leakage_found = True
        print(f"⚠️  WARNING: Found {len(train_test_overlap)} sample names in both TRAIN and TEST!")
        print(f"Examples: {sorted(list(train_test_overlap))[:10]}")
        print()
    
    if val_test_overlap:
        leakage_found = True
        print(f"⚠️  WARNING: Found {len(val_test_overlap)} sample names in both VAL and TEST!")
        print(f"Examples: {sorted(list(val_test_overlap))[:10]}")
        print()
    
    if not leakage_found:
        print("✅ No data leakage detected! All sample names are unique to their splits.")
    else:
        print("❌ Data leakage detected! Please regenerate the splits.")
        return 1
    
    # Show statistics about variations
    sample_counts: Dict[str, int] = defaultdict(int)
    for sample in train_samples + val_samples + test_samples:
        sample_name = extract_sample_name(sample)
        sample_counts[sample_name] += 1
    
    max_variations = max(sample_counts.values())
    min_variations = min(sample_counts.values())
    avg_variations = sum(sample_counts.values()) / len(sample_counts)
    
    print()
    print(f"Variation statistics:")
    print(f"  Min variations per sample: {min_variations}")
    print(f"  Max variations per sample: {max_variations}")
    print(f"  Avg variations per sample: {avg_variations:.2f}")
    
    return 0


if __name__ == "__main__":
    exit(main())


