import csv
import os
import random
from pathlib import Path

os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "0")

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


CROP_X1, CROP_Y1 = 367, 100
CROP_X2, CROP_Y2 = 1760, 750
MODEL_W, MODEL_H = 512, 288
OUT_W, OUT_H = MODEL_W, MODEL_H
FRAMES_IN = 3

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)
ZERO_FRAME = torch.zeros(3, MODEL_H, MODEL_W, dtype=torch.float32)


def generate_gaussian_heatmap(cx_norm, cy_norm, out_w, out_h, sigma=3.0):
    if not (0.0 <= cx_norm <= 1.0 and 0.0 <= cy_norm <= 1.0):
        return np.zeros((out_h, out_w), dtype=np.float32)

    cx = np.clip(cx_norm, 0.0, 1.0) * max(out_w - 1, 1)
    cy = np.clip(cy_norm, 0.0, 1.0) * max(out_h - 1, 1)
    grid_x, grid_y = np.meshgrid(np.arange(out_w), np.arange(out_h))
    dist2 = (grid_x - cx) ** 2 + (grid_y - cy) ** 2
    return np.exp(-dist2 / (2.0 * sigma ** 2)).astype(np.float32)


def is_invalid_tensor(tensor):
    return tensor.abs().sum().item() < 1e-3


def normalize_resized_bgr_frame(frame_bgr):
    if frame_bgr is None or frame_bgr.size == 0:
        return ZERO_FRAME.clone()

    if frame_bgr.shape[0] != MODEL_H or frame_bgr.shape[1] != MODEL_W:
        frame_bgr = cv2.resize(frame_bgr, (MODEL_W, MODEL_H), interpolation=cv2.INTER_LINEAR)

    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = torch.from_numpy(rgb).permute(2, 0, 1)
    tensor = (tensor - IMAGENET_MEAN) / IMAGENET_STD
    if is_invalid_tensor(tensor):
        return ZERO_FRAME.clone()
    return tensor


def normalize_video_bgr_frame(frame_bgr):
    if frame_bgr is None or frame_bgr.size == 0:
        return ZERO_FRAME.clone()

    cropped = frame_bgr[CROP_Y1:CROP_Y2, CROP_X1:CROP_X2]
    if cropped.size == 0:
        return ZERO_FRAME.clone()

    resized = cv2.resize(cropped, (MODEL_W, MODEL_H), interpolation=cv2.INTER_LINEAR)
    return normalize_resized_bgr_frame(resized)


def build_frame_path(video_path, frame_idx, video_root=None, frame_root=None):
    if not video_root or not frame_root:
        return None

    video_path = Path(video_path)
    video_root = Path(video_root)
    frame_root = Path(frame_root)
    try:
        rel = video_path.relative_to(video_root)
    except ValueError:
        return None
    return frame_root / rel.with_suffix("") / f"{frame_idx:06d}.jpg"


def load_and_preprocess_frame(video_path, frame_idx, video_root=None, frame_root=None):
    frame_path = build_frame_path(video_path, frame_idx, video_root=video_root, frame_root=frame_root)
    if frame_path is not None and frame_path.exists():
        frame_bgr = cv2.imread(str(frame_path))
        tensor = normalize_resized_bgr_frame(frame_bgr)
        if not is_invalid_tensor(tensor):
            return tensor

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame_bgr = cap.read()
    cap.release()

    if not ret:
        return ZERO_FRAME.clone()
    return normalize_video_bgr_frame(frame_bgr)


class BallDataset(Dataset):
    """
    Training dataset.

    Each sample returns:
      - input:   [FRAMES_IN * 3, MODEL_H, MODEL_W] -> [9, 288, 512]
      - heatmap: [FRAMES_IN, OUT_H, OUT_W]         -> [3, 288, 512]
    """

    def __init__(self, csv_path, augment=True, sigma=3.0, step=3, video_root=None, frame_root=None):
        self.augment = augment
        self.sigma = sigma
        self.step = step
        self.video_root = video_root
        self.frame_root = frame_root
        self.samples = []

        video_labels = {}
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                video_path = row["video_path"]
                frame_idx = int(row["frame_idx"])
                video_labels.setdefault(video_path, {})[frame_idx] = (
                    float(row["cx_norm"]),
                    float(row["cy_norm"]),
                )

        skipped_missing_history = 0
        for video_path, label_map in video_labels.items():
            for frame_idx in sorted(label_map.keys()):
                frame_ids = [frame_idx - 2 * self.step, frame_idx - self.step, frame_idx]
                if any(frame_id not in label_map for frame_id in frame_ids):
                    skipped_missing_history += 1
                    continue

                coords = [label_map[frame_id] for frame_id in frame_ids]
                self.samples.append((video_path, frame_ids, coords))

        print(
            f"[Dataset] Loaded {len(self.samples)} samples from {csv_path} "
            f"(skipped {skipped_missing_history} samples without label history)"
        )
        if self.frame_root:
            print(f"[Dataset] frame_root enabled: {self.frame_root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, frame_ids, coords = self.samples[idx]

        frames = []
        for frame_id in frame_ids:
            frame = load_and_preprocess_frame(
                video_path,
                frame_id,
                video_root=self.video_root,
                frame_root=self.frame_root,
            )
            if is_invalid_tensor(frame):
                frame = frames[-1].clone() if frames else ZERO_FRAME.clone()
            frames.append(frame)

        flip = self.augment and random.random() < 0.5
        if flip:
            frames = [frame.flip(-1) for frame in frames]
            coords = [(1.0 - cx_norm, cy_norm) for cx_norm, cy_norm in coords]

        input_tensor = torch.cat(frames, dim=0)

        heatmaps = []
        for cx_norm, cy_norm in coords:
            heatmaps.append(
                generate_gaussian_heatmap(
                    cx_norm,
                    cy_norm,
                    OUT_W,
                    OUT_H,
                    sigma=self.sigma,
                )
            )
        heatmap = torch.from_numpy(np.stack(heatmaps, axis=0))

        return input_tensor, heatmap
