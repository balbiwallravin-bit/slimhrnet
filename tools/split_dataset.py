"""
Split a pseudo-label CSV by video to avoid frame-level leakage.

Example:
  python tools/split_dataset.py \
      --input_csv data/pseudo_labels_clean.csv \
      --output_dir data/splits
"""

import argparse
import csv
import os
import random
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_csv",
        default="/home/lht/codexwork/slimhrnet/data_maked/pseudo_labels/clean/pseudo_labels_clean_v2.csv",
    )
    parser.add_argument(
        "--output_dir",
        default="/home/lht/codexwork/slimhrnet/data_maked/splits_v2",
    )
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)

    video_rows = defaultdict(list)
    fieldnames = None
    with open(args.input_csv) as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        for row in reader:
            video_rows[row["video_path"]].append(row)

    videos = list(video_rows.keys())
    random.shuffle(videos)

    num_videos = len(videos)
    num_train = int(num_videos * args.train_ratio)
    num_val = int(num_videos * args.val_ratio)

    splits = {
        "train": videos[:num_train],
        "val": videos[num_train : num_train + num_val],
        "test": videos[num_train + num_val :],
    }

    os.makedirs(args.output_dir, exist_ok=True)
    if not fieldnames:
        raise ValueError(f"no columns found in {args.input_csv}")

    for split_name, split_videos in splits.items():
        out_path = os.path.join(args.output_dir, f"{split_name}.csv")
        rows = []
        for video_path in split_videos:
            rows.extend(video_rows[video_path])
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"  {split_name}: {len(split_videos)} videos, {len(rows)} frames -> {out_path}")

    print("[Done] Dataset split complete")


if __name__ == "__main__":
    main()
