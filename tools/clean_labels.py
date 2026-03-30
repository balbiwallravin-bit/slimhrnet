"""
Filter raw pseudo labels into a cleaner CSV.

Example:
  python tools/clean_labels.py \
      --input_csv data/pseudo_labels_raw.csv \
      --output_csv data/pseudo_labels_clean.csv
"""

import argparse
import csv
import math
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", default="data/pseudo_labels_raw.csv")
    parser.add_argument("--output_csv", default="data/pseudo_labels_clean.csv")
    parser.add_argument(
        "--min_score",
        type=float,
        default=0.5,
        help="Drop frames below this confidence threshold.",
    )
    parser.add_argument(
        "--max_jump",
        type=float,
        default=0.15,
        help="Maximum normalized jump allowed between nearby frames.",
    )
    parser.add_argument(
        "--min_coverage",
        type=float,
        default=0.3,
        help="Drop entire videos below this effective coverage ratio.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    raw = defaultdict(list)
    with open(args.input_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw[row["video_path"]].append(
                {
                    "frame_idx": int(row["frame_idx"]),
                    "cx_norm": float(row["cx_norm"]),
                    "cy_norm": float(row["cy_norm"]),
                    "score": float(row["score"]),
                }
            )

    print(f"[INFO] Loaded raw labels from {len(raw)} videos")

    kept_rows = []
    stats = {
        "videos_total": len(raw),
        "videos_kept": 0,
        "frames_total": 0,
        "frames_kept": 0,
    }

    for video_path, frames in raw.items():
        frames.sort(key=lambda item: item["frame_idx"])
        stats["frames_total"] += len(frames)

        frames = [item for item in frames if item["score"] >= args.min_score]

        clean = [frames[0]] if frames else []
        for index in range(1, len(frames)):
            prev, cur = frames[index - 1], frames[index]
            if cur["frame_idx"] - prev["frame_idx"] <= 9:
                dx = cur["cx_norm"] - prev["cx_norm"]
                dy = cur["cy_norm"] - prev["cy_norm"]
                dist = math.sqrt(dx * dx + dy * dy)
                if dist > args.max_jump:
                    continue
            clean.append(cur)

        if not clean:
            continue

        frame_range = clean[-1]["frame_idx"] - clean[0]["frame_idx"] + 1
        coverage = len(clean) / max(frame_range, 1)
        if coverage < args.min_coverage:
            continue

        stats["frames_kept"] += len(clean)
        stats["videos_kept"] += 1
        for item in clean:
            kept_rows.append(
                {
                    "video_path": video_path,
                    "frame_idx": item["frame_idx"],
                    "cx_norm": item["cx_norm"],
                    "cy_norm": item["cy_norm"],
                    "score": item["score"],
                }
            )

    import os

    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["video_path", "frame_idx", "cx_norm", "cy_norm", "score"],
        )
        writer.writeheader()
        writer.writerows(kept_rows)

    print("[Result]")
    print(f"  Videos: {stats['videos_total']} -> kept {stats['videos_kept']}")
    print(f"  Frames: {stats['frames_total']} -> kept {stats['frames_kept']}")
    print(
        f"  Keep ratio: "
        f"{stats['frames_kept'] / max(stats['frames_total'], 1) * 100:.1f}%"
    )
    print(f"  Output: {args.output_csv}")


if __name__ == "__main__":
    main()
