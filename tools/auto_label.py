"""
Auto-label TS videos with a pretrained BlurBall teacher and export pseudo labels.

Example:
  python tools/auto_label.py \
      --video_root /home/lht/daqiu \
      --weights /home/lht/codexwork/blurball-mainyy/blurball_best.pth \
      --output_csv data/pseudo_labels_raw.csv \
      --device cuda
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path

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


CROP_X1, CROP_Y1 = 367, 100
CROP_X2, CROP_Y2 = 1760, 750
CROP_W = CROP_X2 - CROP_X1
CROP_H = CROP_Y2 - CROP_Y1

MODEL_W, MODEL_H = 512, 288
FRAMES_IN = 3

# In this repo, HRNet/BlurBall defaults to full-resolution heatmaps.
HM_W = MODEL_W
HM_H = MODEL_H

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_root", default="/home/lht/daqiu")
    parser.add_argument(
        "--weights",
        default="/home/lht/codexwork/blurball-mainyy/blurball_best.pth",
    )
    parser.add_argument("--output_csv", default="data/pseudo_labels_raw.csv")
    parser.add_argument(
        "--score_threshold",
        type=float,
        default=0.4,
        help="Mark frames below this confidence as no-ball.",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=3,
        help="Process every Nth frame. step=3 is about 20 fps for 60 fps videos.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--max_videos",
        type=int,
        default=None,
        help="Debug option: only process the first N videos.",
    )
    return parser.parse_args()


def find_all_ts_videos(root):
    videos = []
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            if filename.lower().endswith(".ts"):
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
    """
    Build the local BlurBall architecture and load teacher weights into it.
    """
    cfg = load_runtime_config("inference_blurball", device)
    model = build_model(cfg)

    state = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(extract_state_dict(state), strict=False)
    model = model.to(device).eval()
    print(f"[INFO] Loaded BlurBall weights: {weights_path}")
    return model


def preprocess_frame(frame_bgr):
    """
    Read an OpenCV BGR frame, crop, resize, and normalize to [3, MODEL_H, MODEL_W].
    """
    cropped = frame_bgr[CROP_Y1:CROP_Y2, CROP_X1:CROP_X2]
    resized = cv2.resize(cropped, (MODEL_W, MODEL_H), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = torch.from_numpy(rgb).permute(2, 0, 1)
    tensor = (tensor - IMAGENET_MEAN) / IMAGENET_STD
    return tensor


def extract_primary_heatmap(preds):
    """
    Current BlurBall/HRNet returns {scale: tensor}. We only use scale 0.
    Shape: [B, C, H, W]
    """
    if isinstance(preds, dict):
        if 0 not in preds:
            raise KeyError("Model output does not contain scale=0")
        return preds[0]
    return preds


def decode_heatmap(hm_tensor, score_threshold):
    """
    Decode a single heatmap into normalized coordinates.
    Returns (cx_norm, cy_norm, score) or None.
    """
    hm = torch.sigmoid(hm_tensor)
    score = hm.max().item()
    if score < score_threshold:
        return None

    height, width = hm.shape
    device = hm.device
    weights = hm * (hm >= score_threshold)
    if weights.sum().item() <= 0:
        weights = hm

    grid_y, grid_x = torch.meshgrid(
        torch.arange(height, dtype=torch.float32, device=device),
        torch.arange(width, dtype=torch.float32, device=device),
        indexing="ij",
    )
    total = weights.sum() + 1e-6
    cx_hm = (weights * grid_x).sum() / total
    cy_hm = (weights * grid_y).sum() / total

    cx_norm = cx_hm.item() / max(width - 1, 1)
    cy_norm = cy_hm.item() / max(height - 1, 1)
    return cx_norm, cy_norm, score


def process_video(model, video_path, step, score_threshold, device):
    """
    Return a list of label rows for one video.
    Each row contains frame_idx, cx_norm, cy_norm, and score.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[WARN] Cannot open video: {video_path}")
        return []

    records = []
    frame_buffer = []
    frame_idx = 0

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        if frame_idx % step == 0:
            tensor = preprocess_frame(frame_bgr)
            frame_buffer.append(tensor)
            if len(frame_buffer) > FRAMES_IN:
                frame_buffer.pop(0)

            if len(frame_buffer) == FRAMES_IN:
                inp = torch.cat(frame_buffer, dim=0).unsqueeze(0).to(device)
                with torch.no_grad():
                    preds = model(inp)

                hm_out = extract_primary_heatmap(preds)
                # Keep only the current frame, i.e. the last output channel.
                result = decode_heatmap(hm_out[0, FRAMES_IN - 1], score_threshold)
                if result is not None:
                    cx_norm, cy_norm, score = result
                    records.append(
                        {
                            "frame_idx": frame_idx,
                            "cx_norm": round(cx_norm, 6),
                            "cy_norm": round(cy_norm, 6),
                            "score": round(score, 4),
                        }
                    )

        frame_idx += 1

    cap.release()
    return records


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)

    all_videos = find_all_ts_videos(args.video_root)
    if args.max_videos:
        all_videos = all_videos[: args.max_videos]
    print(f"[INFO] Found {len(all_videos)} .ts videos")

    if not all_videos:
        print("[WARN] No .ts videos found; exiting")
        return

    model = load_blurball_model(args.weights, device)

    fieldnames = ["video_path", "frame_idx", "cx_norm", "cy_norm", "score"]
    total_labeled = 0
    t0 = time.time()

    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for video_index, video_path in enumerate(all_videos):
            records = process_video(
                model,
                video_path,
                args.step,
                args.score_threshold,
                device,
            )
            for record in records:
                writer.writerow(
                    {
                        "video_path": video_path,
                        "frame_idx": record["frame_idx"],
                        "cx_norm": record["cx_norm"],
                        "cy_norm": record["cy_norm"],
                        "score": record["score"],
                    }
                )
            total_labeled += len(records)

            if (video_index + 1) % 50 == 0 or video_index == len(all_videos) - 1:
                elapsed = time.time() - t0
                eta = elapsed / max(video_index + 1, 1) * (len(all_videos) - video_index - 1)
                print(
                    f"[{video_index + 1}/{len(all_videos)}] "
                    f"labeled_frames={total_labeled} | "
                    f"elapsed={elapsed / 60:.1f}min | "
                    f"eta={eta / 60:.1f}min"
                )

    print(f"\n[Done] Saved pseudo labels to: {args.output_csv}")
    print(f"       Total labeled frames: {total_labeled} from {len(all_videos)} videos")


if __name__ == "__main__":
    main()
