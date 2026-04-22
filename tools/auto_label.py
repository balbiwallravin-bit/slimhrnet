"""
Auto-label videos with a pretrained BlurBall teacher and export richer pseudo labels.

The new workflow keeps every processed frame after the history buffer is full, then
lets later filtering decide which segments are trustworthy.
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "0")

import cv2
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
CONFIG_DIR = SRC_DIR / "configs"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from models import build_model  # noqa: E402
from utils.resize_ops import (  # noqa: E402
    MODEL_H,
    MODEL_W,
    build_resize_plan,
    model_to_normalized,
    model_to_original_pixels,
    resize_frame_with_plan,
)


FRAMES_IN = 3
VIDEO_EXTENSIONS = (".ts", ".mp4", ".avi", ".mov", ".mkv")

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)
ZERO_FRAME = torch.zeros(3, MODEL_H, MODEL_W, dtype=torch.float32)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_root", default="/home/lht/daqiu")
    parser.add_argument(
        "--weights",
        default="/home/lht/codexwork/blurball-mainyy/blurball_best.pth",
    )
    parser.add_argument(
        "--output_csv",
        default="/home/lht/codexwork/slimhrnet/data_maked/pseudo_labels/raw/pseudo_labels_raw.csv",
    )
    parser.add_argument(
        "--score_threshold",
        type=float,
        default=0.5,
        help="Confidence threshold recorded into visible_teacher for later filtering.",
    )
    parser.add_argument(
        "--component_threshold",
        type=float,
        default=0.35,
        help="Threshold used to estimate blur angle/length from the teacher heatmap blob.",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=1,
        help="Process every Nth frame. step=1 keeps the 60fps temporal structure.",
    )
    parser.add_argument(
        "--resize_mode",
        default="stretch",
        choices=["stretch", "letterbox"],
        help="How 1920x1080 frames are mapped to 512x288 before the teacher.",
    )
    parser.add_argument("--pad_value", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--max_videos",
        type=int,
        default=None,
        help="Debug option: only process the first N selected videos.",
    )
    parser.add_argument(
        "--num_shards",
        type=int,
        default=1,
        help="Split the video list so multiple processes can label different shards in parallel.",
    )
    parser.add_argument(
        "--shard_idx",
        type=int,
        default=0,
        help="Current shard index in [0, num_shards).",
    )
    return parser.parse_args()


def find_all_videos(root):
    videos = []
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            if filename.lower().endswith(VIDEO_EXTENSIONS):
                videos.append(os.path.join(dirpath, filename))
    videos.sort()
    return videos


def extract_state_dict(state):
    if isinstance(state, dict):
        for key in ["model", "state_dict", "model_state_dict", "net"]:
            if key in state:
                return state[key]
    return state


def load_runtime_config(config_name, device):
    with initialize_config_dir(config_dir=str(CONFIG_DIR.resolve()), version_base=None):
        cfg = compose(config_name=config_name)

    OmegaConf.set_struct(cfg, False)
    cfg.model.frames_in = FRAMES_IN
    cfg.model.frames_out = FRAMES_IN
    cfg.model.inp_width = MODEL_W
    cfg.model.inp_height = MODEL_H
    cfg.model.out_width = MODEL_W
    cfg.model.out_height = MODEL_H
    cfg.runner.device = "cuda" if device.type == "cuda" else "cpu"
    cfg.runner.gpus = [0]
    return cfg


def load_blurball_model(weights_path, device):
    cfg = load_runtime_config("inference_blurball", device)
    model = build_model(cfg)

    state = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(extract_state_dict(state), strict=False)
    model = model.to(device).eval()
    print(f"[INFO] Loaded BlurBall weights: {weights_path}")
    return model


def is_invalid_tensor(tensor):
    return tensor.abs().sum().item() < 1e-3


def preprocess_frame(frame_bgr, resize_mode, pad_value):
    if frame_bgr is None or frame_bgr.size == 0:
        return ZERO_FRAME.clone(), None

    plan = build_resize_plan(
        orig_w=frame_bgr.shape[1],
        orig_h=frame_bgr.shape[0],
        dst_w=MODEL_W,
        dst_h=MODEL_H,
        mode=resize_mode,
        pad_value=pad_value,
    )
    resized = resize_frame_with_plan(frame_bgr, plan)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = torch.from_numpy(rgb).permute(2, 0, 1)
    tensor = (tensor - IMAGENET_MEAN) / IMAGENET_STD
    if is_invalid_tensor(tensor):
        return ZERO_FRAME.clone(), plan
    return tensor, plan


def extract_primary_heatmap(preds):
    if isinstance(preds, dict):
        if 0 not in preds:
            raise KeyError("Model output does not contain scale=0")
        return preds[0]
    return preds


def _component_geometry(mask, weights):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None

    w = weights[ys, xs].astype(np.float32)
    weight_sum = float(w.sum())
    if weight_sum <= 1e-6:
        w = np.ones_like(w, dtype=np.float32)
        weight_sum = float(w.sum())

    x = float(np.sum(xs.astype(np.float32) * w) / weight_sum)
    y = float(np.sum(ys.astype(np.float32) * w) / weight_sum)

    if len(xs) < 2:
        return {
            "x_model": x,
            "y_model": y,
            "angle_deg": 0.0,
            "length_input_px": 0.0,
        }

    points = np.stack([xs, ys], axis=1).astype(np.float32)
    mean, eigenvectors = cv2.PCACompute(points, mean=None)
    principal_axis = eigenvectors[0]
    centered = points - mean
    projected = centered @ principal_axis
    length = float(projected.max() - projected.min())
    angle_deg = float(np.degrees(np.arctan2(principal_axis[1], principal_axis[0])))
    return {
        "x_model": x,
        "y_model": y,
        "angle_deg": angle_deg,
        "length_input_px": length,
    }


def decode_heatmap(hm_tensor, component_threshold):
    hm = torch.sigmoid(hm_tensor).detach().cpu().numpy()
    peak_score = float(hm.max())
    peak_y, peak_x = np.unravel_index(np.argmax(hm), hm.shape)

    mask = hm >= component_threshold
    if not mask[peak_y, peak_x]:
        mask = np.zeros_like(mask, dtype=np.uint8)
        mask[peak_y, peak_x] = 1
    else:
        _, labels = cv2.connectedComponents(mask.astype(np.uint8))
        label_id = int(labels[peak_y, peak_x])
        mask = labels == label_id

    geometry = _component_geometry(mask, hm)
    if geometry is None:
        geometry = {
            "x_model": float(peak_x),
            "y_model": float(peak_y),
            "angle_deg": 0.0,
            "length_input_px": 0.0,
        }

    geometry["score"] = peak_score
    return geometry


def process_video(model, video_path, step, score_threshold, component_threshold, device, resize_mode, pad_value):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[WARN] Cannot open video: {video_path}")
        return []

    records = []
    frame_buffer = []
    frame_idx = 0
    consecutive_failures = 0
    resize_plan = None

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            consecutive_failures += 1
            if consecutive_failures > 8:
                break
            frame_idx += 1
            continue
        consecutive_failures = 0

        if frame_idx % step == 0:
            tensor, plan = preprocess_frame(frame_bgr, resize_mode=resize_mode, pad_value=pad_value)
            resize_plan = plan if plan is not None else resize_plan
            if is_invalid_tensor(tensor):
                tensor = frame_buffer[-1].clone() if frame_buffer else ZERO_FRAME.clone()

            frame_buffer.append(tensor)
            if len(frame_buffer) > FRAMES_IN:
                frame_buffer.pop(0)

            if len(frame_buffer) == FRAMES_IN and resize_plan is not None:
                inp = torch.cat(frame_buffer, dim=0).unsqueeze(0).to(device)
                with torch.no_grad():
                    preds = model(inp)

                hm_out = extract_primary_heatmap(preds)
                result = decode_heatmap(hm_out[0, FRAMES_IN - 1], component_threshold)
                x_px, y_px = model_to_original_pixels(
                    result["x_model"],
                    result["y_model"],
                    resize_plan,
                )
                cx_norm, cy_norm = model_to_normalized(
                    result["x_model"],
                    result["y_model"],
                    resize_plan,
                )

                if resize_plan["mode"] == "stretch":
                    avg_scale = 0.5 * (
                        float(resize_plan["scale_x"]) + float(resize_plan["scale_y"])
                    )
                else:
                    avg_scale = float(resize_plan["scale"])
                length_px = result["length_input_px"] / max(avg_scale, 1e-6)

                records.append(
                    {
                        "frame_idx": frame_idx,
                        "x_model": round(float(result["x_model"]), 4),
                        "y_model": round(float(result["y_model"]), 4),
                        "x_px": round(float(x_px), 4),
                        "y_px": round(float(y_px), 4),
                        "cx_norm": round(float(cx_norm), 6),
                        "cy_norm": round(float(cy_norm), 6),
                        "score": round(float(result["score"]), 6),
                        "angle_deg": round(float(result["angle_deg"]), 4),
                        "length_input_px": round(float(result["length_input_px"]), 4),
                        "length_px": round(float(length_px), 4),
                        "visible_teacher": int(float(result["score"]) >= score_threshold),
                    }
                )

        frame_idx += 1

    cap.release()
    return records


def main():
    args = parse_args()
    if args.num_shards <= 0:
        raise ValueError("--num_shards must be >= 1")
    if not (0 <= args.shard_idx < args.num_shards):
        raise ValueError("--shard_idx must be within [0, num_shards)")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)

    all_videos = find_all_videos(args.video_root)
    all_videos = [
        video_path
        for video_index, video_path in enumerate(all_videos)
        if video_index % args.num_shards == args.shard_idx
    ]
    if args.max_videos:
        all_videos = all_videos[: args.max_videos]
    print(f"[INFO] Found {len(all_videos)} videos in shard {args.shard_idx}/{args.num_shards}")

    if not all_videos:
        print("[WARN] No videos found; exiting")
        return

    model = load_blurball_model(args.weights, device)

    fieldnames = [
        "video_path",
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
    ]
    total_rows = 0
    t0 = time.time()

    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for video_index, video_path in enumerate(all_videos):
            records = process_video(
                model=model,
                video_path=video_path,
                step=args.step,
                score_threshold=args.score_threshold,
                component_threshold=args.component_threshold,
                device=device,
                resize_mode=args.resize_mode,
                pad_value=args.pad_value,
            )
            for record in records:
                writer.writerow(
                    {
                        "video_path": video_path,
                        "frame_idx": record["frame_idx"],
                        "x_model": record["x_model"],
                        "y_model": record["y_model"],
                        "x_px": record["x_px"],
                        "y_px": record["y_px"],
                        "cx_norm": record["cx_norm"],
                        "cy_norm": record["cy_norm"],
                        "score": record["score"],
                        "angle_deg": record["angle_deg"],
                        "length_input_px": record["length_input_px"],
                        "length_px": record["length_px"],
                        "visible_teacher": record["visible_teacher"],
                        "resize_mode": args.resize_mode,
                        "step": args.step,
                    }
                )
            total_rows += len(records)

            if (video_index + 1) % 25 == 0 or video_index == len(all_videos) - 1:
                elapsed = time.time() - t0
                eta = elapsed / max(video_index + 1, 1) * (len(all_videos) - video_index - 1)
                print(
                    f"[{video_index + 1}/{len(all_videos)}] "
                    f"rows={total_rows} | "
                    f"elapsed={elapsed / 60:.1f}min | "
                    f"eta={eta / 60:.1f}min"
                )

    print(f"\n[Done] Saved pseudo labels to: {args.output_csv}")
    print(f"       Total rows: {total_rows} from {len(all_videos)} videos")


if __name__ == "__main__":
    main()
