"""
Turn raw teacher predictions into cleaner pseudo-label segments.

The new cleaner is segment-based:
- low-score frames break a segment
- large velocity residuals break a segment
- short / overly long segments are dropped
- short gaps inside a valid segment can be linearly interpolated and marked
"""

import argparse
import csv
import math
import os
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_csv",
        default="/home/lht/codexwork/slimhrnet/data_maked/pseudo_labels/raw/pseudo_labels_raw.csv",
    )
    parser.add_argument(
        "--output_csv",
        default="/home/lht/codexwork/slimhrnet/data_maked/pseudo_labels/clean/pseudo_labels_clean_v2.csv",
    )
    parser.add_argument(
        "--min_score",
        type=float,
        default=0.5,
        help="Rows below this score break the current segment.",
    )
    parser.add_argument(
        "--max_residual_px",
        type=float,
        default=150.0,
        help="Maximum constant-velocity residual before a segment is split.",
    )
    parser.add_argument(
        "--min_segment_frames",
        type=int,
        default=30,
        help="Drop segments shorter than this many sampled frames.",
    )
    parser.add_argument(
        "--max_segment_frames",
        type=int,
        default=480,
        help="Drop segments longer than this many sampled frames.",
    )
    parser.add_argument(
        "--max_frame_gap",
        type=int,
        default=1,
        help="Split when consecutive accepted rows are farther apart than this.",
    )
    parser.add_argument(
        "--interp_max_gap",
        type=int,
        default=2,
        help="Interpolate at most this many missing frames inside a kept segment.",
    )
    parser.add_argument(
        "--interpolated_weight",
        type=float,
        default=0.5,
        help="Weight attached to interpolated rows for later training.",
    )
    return parser.parse_args()


def _float(row, key, default=0.0):
    value = row.get(key, default)
    if value in ("", None):
        return float(default)
    return float(value)


def _int(row, key, default=0):
    value = row.get(key, default)
    if value in ("", None):
        return int(default)
    return int(value)


def load_rows(input_csv):
    grouped = defaultdict(list)
    with open(input_csv, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            grouped[row["video_path"]].append(
                {
                    "video_path": row["video_path"],
                    "frame_idx": _int(row, "frame_idx"),
                    "x_model": _float(row, "x_model"),
                    "y_model": _float(row, "y_model"),
                    "x_px": _float(row, "x_px"),
                    "y_px": _float(row, "y_px"),
                    "cx_norm": _float(row, "cx_norm"),
                    "cy_norm": _float(row, "cy_norm"),
                    "score": _float(row, "score"),
                    "angle_deg": _float(row, "angle_deg"),
                    "length_input_px": _float(row, "length_input_px"),
                    "length_px": _float(row, "length_px"),
                    "visible_teacher": _int(row, "visible_teacher", default=0),
                    "resize_mode": row.get("resize_mode", "stretch"),
                    "step": _int(row, "step", default=1),
                }
            )
    return grouped


def constant_velocity_residual(prev2, prev1, cur):
    dx = prev1["x_px"] - prev2["x_px"]
    dy = prev1["y_px"] - prev2["y_px"]
    frame_delta = prev1["frame_idx"] - prev2["frame_idx"]
    if frame_delta <= 0:
        return 0.0

    gap = cur["frame_idx"] - prev1["frame_idx"]
    scale = gap / frame_delta
    x_pred = prev1["x_px"] + dx * scale
    y_pred = prev1["y_px"] + dy * scale
    return math.hypot(cur["x_px"] - x_pred, cur["y_px"] - y_pred)


def filter_segments(rows, args):
    rows = sorted(rows, key=lambda item: item["frame_idx"])
    segments = []
    current = []

    for row in rows:
        if row["score"] < args.min_score:
            if args.min_segment_frames <= len(current) <= args.max_segment_frames:
                segments.append(current)
            current = []
            continue

        if not current:
            current = [row]
            continue

        gap = row["frame_idx"] - current[-1]["frame_idx"]
        if gap > args.max_frame_gap + 1:
            if args.min_segment_frames <= len(current) <= args.max_segment_frames:
                segments.append(current)
            current = [row]
            continue

        if len(current) >= 2:
            residual = constant_velocity_residual(current[-2], current[-1], row)
            if residual > args.max_residual_px:
                if args.min_segment_frames <= len(current) <= args.max_segment_frames:
                    segments.append(current)
                current = [row]
                continue

        current.append(row)

    if args.min_segment_frames <= len(current) <= args.max_segment_frames:
        segments.append(current)

    return segments


def interpolate_angle(a0, a1, t):
    delta = ((a1 - a0 + 180.0) % 360.0) - 180.0
    return a0 + delta * t


def densify_segment(segment, segment_id, args):
    dense_rows = []
    for idx, row in enumerate(segment):
        row_out = dict(row)
        row_out["segment_id"] = segment_id
        row_out["interpolated"] = 0
        row_out["label_weight"] = 1.0
        dense_rows.append(row_out)

        if idx == len(segment) - 1:
            continue

        nxt = segment[idx + 1]
        gap = nxt["frame_idx"] - row["frame_idx"] - 1
        if gap <= 0 or gap > args.interp_max_gap:
            continue

        for missing_offset in range(1, gap + 1):
            t = missing_offset / float(gap + 1)
            interp_row = {
                "video_path": row["video_path"],
                "frame_idx": row["frame_idx"] + missing_offset,
                "x_model": row["x_model"] + (nxt["x_model"] - row["x_model"]) * t,
                "y_model": row["y_model"] + (nxt["y_model"] - row["y_model"]) * t,
                "x_px": row["x_px"] + (nxt["x_px"] - row["x_px"]) * t,
                "y_px": row["y_px"] + (nxt["y_px"] - row["y_px"]) * t,
                "cx_norm": row["cx_norm"] + (nxt["cx_norm"] - row["cx_norm"]) * t,
                "cy_norm": row["cy_norm"] + (nxt["cy_norm"] - row["cy_norm"]) * t,
                "score": min(row["score"], nxt["score"]),
                "angle_deg": interpolate_angle(row["angle_deg"], nxt["angle_deg"], t),
                "length_input_px": row["length_input_px"]
                + (nxt["length_input_px"] - row["length_input_px"]) * t,
                "length_px": row["length_px"] + (nxt["length_px"] - row["length_px"]) * t,
                "visible_teacher": 1,
                "resize_mode": row["resize_mode"],
                "step": row["step"],
                "segment_id": segment_id,
                "interpolated": 1,
                "label_weight": float(args.interpolated_weight),
            }
            dense_rows.append(interp_row)

    dense_rows.sort(key=lambda item: item["frame_idx"])
    return dense_rows


def main():
    args = parse_args()
    raw = load_rows(args.input_csv)
    print(f"[INFO] Loaded raw labels from {len(raw)} videos")

    kept_rows = []
    stats = {
        "videos_total": len(raw),
        "videos_kept": 0,
        "rows_total": 0,
        "rows_kept": 0,
        "segments_kept": 0,
        "interpolated_rows": 0,
    }

    for video_index, (video_path, rows) in enumerate(sorted(raw.items())):
        stats["rows_total"] += len(rows)
        segments = filter_segments(rows, args)
        if not segments:
            continue

        stats["videos_kept"] += 1
        for segment_index, segment in enumerate(segments):
            segment_id = f"{video_index:05d}_{segment_index:04d}"
            dense_rows = densify_segment(segment, segment_id, args)
            kept_rows.extend(dense_rows)
            stats["segments_kept"] += 1
            stats["interpolated_rows"] += sum(
                int(item["interpolated"]) for item in dense_rows
            )

    stats["rows_kept"] = len(kept_rows)

    fieldnames = [
        "video_path",
        "segment_id",
        "frame_idx",
        "x_model",
        "y_model",
        "x_px",
        "y_px",
        "cx_norm",
        "cy_norm",
        "score",
        "angle_deg",
        "length_input_px",
        "length_px",
        "visible_teacher",
        "resize_mode",
        "step",
        "interpolated",
        "label_weight",
    ]
    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept_rows)

    print("[Result]")
    print(f"  Videos: {stats['videos_total']} -> kept {stats['videos_kept']}")
    print(f"  Rows:   {stats['rows_total']} -> kept {stats['rows_kept']}")
    print(f"  Segments kept: {stats['segments_kept']}")
    print(f"  Interpolated rows: {stats['interpolated_rows']}")
    print(
        "  Keep ratio: "
        f"{stats['rows_kept'] / max(stats['rows_total'], 1) * 100:.1f}%"
    )
    print(f"  Output: {args.output_csv}")


if __name__ == "__main__":
    main()
