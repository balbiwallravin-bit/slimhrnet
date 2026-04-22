import csv
import os
import random
from pathlib import Path

os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "0")

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from utils.resize_ops import MODEL_H, MODEL_W, build_resize_plan, resize_frame_with_plan


OUT_W, OUT_H = MODEL_W, MODEL_H
FRAMES_IN = 3

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)
ZERO_FRAME = torch.zeros(3, MODEL_H, MODEL_W, dtype=torch.float32)


def generate_gaussian_heatmap(x_model, y_model, out_w, out_h, sigma=3.0):
    if x_model < 0 or y_model < 0 or x_model >= out_w or y_model >= out_h:
        return np.zeros((out_h, out_w), dtype=np.float32)

    grid_x, grid_y = np.meshgrid(np.arange(out_w), np.arange(out_h))
    dist2 = (grid_x - x_model) ** 2 + (grid_y - y_model) ** 2
    heatmap = np.exp(-dist2 / (2.0 * sigma ** 2)).astype(np.float32)
    max_value = float(heatmap.max())
    if max_value > 1e-6:
        heatmap /= max_value
    return heatmap


def encode_angle(angle_deg):
    theta = np.deg2rad(angle_deg)
    return float(np.sin(2.0 * theta)), float(np.cos(2.0 * theta))


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


def normalize_video_bgr_frame(frame_bgr, resize_mode="stretch", pad_value=0):
    if frame_bgr is None or frame_bgr.size == 0:
        return ZERO_FRAME.clone()

    plan = build_resize_plan(
        frame_bgr.shape[1],
        frame_bgr.shape[0],
        dst_w=MODEL_W,
        dst_h=MODEL_H,
        mode=resize_mode,
        pad_value=pad_value,
    )
    resized = resize_frame_with_plan(frame_bgr, plan)
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


def load_and_preprocess_frame(
    video_path,
    frame_idx,
    video_root=None,
    frame_root=None,
    resize_mode="stretch",
    pad_value=0,
):
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
    return normalize_video_bgr_frame(frame_bgr, resize_mode=resize_mode, pad_value=pad_value)


def sigma_from_length(length_input_px, default_sigma, dynamic_sigma, min_sigma, max_sigma):
    if not dynamic_sigma:
        return default_sigma
    sigma = float(length_input_px) / 4.0
    sigma = max(float(min_sigma), min(float(max_sigma), sigma))
    return sigma


class BallDataset(Dataset):
    """
    Distillation dataset.

    Returns:
      - input: [9, 288, 512]
      - target dict:
          heatmap: [3, 288, 512]
          angle: [2, 288, 512]
          length: [1, 288, 512]
          mask: [1, 288, 512]
          sample_weight: scalar
          aux_valid: scalar
    """

    def __init__(
        self,
        csv_path,
        augment=True,
        sigma=3.0,
        step=None,
        video_root=None,
        frame_root=None,
        dynamic_sigma=False,
        min_sigma=2.0,
        max_sigma=5.0,
        length_norm_max=64.0,
        resize_mode="stretch",
        pad_value=0,
    ):
        self.augment = augment
        self.sigma = float(sigma)
        self.step = step
        self.video_root = video_root
        self.frame_root = frame_root
        self.dynamic_sigma = bool(dynamic_sigma)
        self.min_sigma = float(min_sigma)
        self.max_sigma = float(max_sigma)
        self.length_norm_max = float(length_norm_max)
        self.resize_mode = resize_mode
        self.pad_value = int(pad_value)
        self.samples = []

        video_labels = {}
        with open(csv_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                video_path = row["video_path"]
                frame_idx = int(row["frame_idx"])
                x_model = float(row.get("x_model", row.get("cx_norm", 0.0))) * (
                    1.0 if "x_model" in row else (OUT_W - 1)
                )
                y_model = float(row.get("y_model", row.get("cy_norm", 0.0))) * (
                    1.0 if "y_model" in row else (OUT_H - 1)
                )
                video_labels.setdefault(video_path, {})[frame_idx] = {
                    "x_model": x_model,
                    "y_model": y_model,
                    "angle_deg": float(row.get("angle_deg", 0.0)),
                    "length_input_px": float(row.get("length_input_px", 0.0)),
                    "score": float(row.get("score", 1.0)),
                    "segment_id": row.get("segment_id", ""),
                    "interpolated": int(row.get("interpolated", 0)),
                    "label_weight": float(row.get("label_weight", 1.0)),
                    "resize_mode": row.get("resize_mode", resize_mode),
                    "step": int(row.get("step", 1)),
                }

        skipped_missing_history = 0
        skipped_mixed_segments = 0
        for video_path, label_map in video_labels.items():
            for frame_idx in sorted(label_map.keys()):
                current = label_map[frame_idx]
                sample_step = int(self.step if self.step is not None else current["step"])
                frame_ids = [frame_idx - 2 * sample_step, frame_idx - sample_step, frame_idx]
                if any(frame_id not in label_map for frame_id in frame_ids):
                    skipped_missing_history += 1
                    continue

                metas = [label_map[frame_id] for frame_id in frame_ids]
                segment_ids = {meta["segment_id"] for meta in metas if meta["segment_id"]}
                if len(segment_ids) > 1:
                    skipped_mixed_segments += 1
                    continue

                self.samples.append((video_path, frame_ids, metas))

        print(
            f"[Dataset] Loaded {len(self.samples)} samples from {csv_path} "
            f"(skipped {skipped_missing_history} without history, {skipped_mixed_segments} across segments)"
        )
        if self.frame_root:
            print(f"[Dataset] frame_root enabled: {self.frame_root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, frame_ids, metas = self.samples[idx]

        frames = []
        for frame_id, meta in zip(frame_ids, metas):
            frame = load_and_preprocess_frame(
                video_path,
                frame_id,
                video_root=self.video_root,
                frame_root=self.frame_root,
                resize_mode=meta["resize_mode"] or self.resize_mode,
                pad_value=self.pad_value,
            )
            if is_invalid_tensor(frame):
                frame = frames[-1].clone() if frames else ZERO_FRAME.clone()
            frames.append(frame)

        flip = self.augment and random.random() < 0.5
        if flip:
            frames = [frame.flip(-1) for frame in frames]

        input_tensor = torch.cat(frames, dim=0)

        heatmaps = []
        label_weights = []
        metas_out = []
        for meta in metas:
            x_model = float(meta["x_model"])
            y_model = float(meta["y_model"])
            angle_deg = float(meta["angle_deg"])
            if flip:
                x_model = (OUT_W - 1) - x_model
                angle_deg = 180.0 - angle_deg

            sigma = sigma_from_length(
                meta["length_input_px"],
                default_sigma=self.sigma,
                dynamic_sigma=self.dynamic_sigma,
                min_sigma=self.min_sigma,
                max_sigma=self.max_sigma,
            )
            heatmaps.append(
                generate_gaussian_heatmap(
                    x_model,
                    y_model,
                    OUT_W,
                    OUT_H,
                    sigma=sigma,
                )
            )
            label_weights.append(float(meta["label_weight"]))
            metas_out.append(
                {
                    "x_model": x_model,
                    "y_model": y_model,
                    "angle_deg": angle_deg,
                    "length_input_px": float(meta["length_input_px"]),
                    "score": float(meta["score"]),
                }
            )

        heatmap = torch.from_numpy(np.stack(heatmaps, axis=0))

        current = metas_out[-1]
        current_hm = heatmap[-1].numpy()
        aux_mask = (current_hm > 0.1).astype(np.float32)
        angle_sin, angle_cos = encode_angle(current["angle_deg"])
        angle_target = np.stack(
            [
                np.full((OUT_H, OUT_W), angle_sin, dtype=np.float32),
                np.full((OUT_H, OUT_W), angle_cos, dtype=np.float32),
            ],
            axis=0,
        )
        length_value = np.clip(
            current["length_input_px"] / max(self.length_norm_max, 1e-6),
            0.0,
            1.0,
        )
        length_target = np.full((1, OUT_H, OUT_W), length_value, dtype=np.float32)

        target = {
            "heatmap": heatmap,
            "angle": torch.from_numpy(angle_target),
            "length": torch.from_numpy(length_target),
            "mask": torch.from_numpy(aux_mask[None, ...]),
            "sample_weight": torch.tensor(
                float(sum(label_weights) / max(len(label_weights), 1)),
                dtype=torch.float32,
            ),
            "aux_valid": torch.tensor(1.0, dtype=torch.float32),
        }
        return input_tensor, target
