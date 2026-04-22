from __future__ import annotations

from typing import Dict, Tuple

import cv2
import numpy as np


MODEL_W = 512
MODEL_H = 288


def build_resize_plan(
    orig_w: int,
    orig_h: int,
    dst_w: int = MODEL_W,
    dst_h: int = MODEL_H,
    mode: str = "stretch",
    pad_value: int = 0,
) -> Dict[str, float]:
    if orig_w <= 0 or orig_h <= 0:
        raise ValueError(f"invalid source size: {orig_w}x{orig_h}")

    mode = mode.lower()
    if mode not in {"stretch", "letterbox"}:
        raise ValueError(f"unsupported resize mode: {mode}")

    plan = {
        "mode": mode,
        "orig_w": int(orig_w),
        "orig_h": int(orig_h),
        "dst_w": int(dst_w),
        "dst_h": int(dst_h),
        "pad_value": int(pad_value),
    }

    if mode == "stretch":
        plan.update(
            {
                "scale_x": float(max(dst_w - 1, 1)) / float(max(orig_w - 1, 1)),
                "scale_y": float(max(dst_h - 1, 1)) / float(max(orig_h - 1, 1)),
                "scale": None,
                "new_w": int(dst_w),
                "new_h": int(dst_h),
                "pad_x": 0.0,
                "pad_y": 0.0,
            }
        )
        return plan

    scale = min(float(dst_w) / float(orig_w), float(dst_h) / float(orig_h))
    new_w = max(1, int(round(orig_w * scale)))
    new_h = max(1, int(round(orig_h * scale)))
    pad_x = (dst_w - new_w) / 2.0
    pad_y = (dst_h - new_h) / 2.0
    plan.update(
        {
            "scale_x": scale,
            "scale_y": scale,
            "scale": scale,
            "new_w": int(new_w),
            "new_h": int(new_h),
            "pad_x": float(pad_x),
            "pad_y": float(pad_y),
        }
    )
    return plan


def resize_frame_with_plan(frame_bgr: np.ndarray, plan: Dict[str, float]) -> np.ndarray:
    dst_w = int(plan["dst_w"])
    dst_h = int(plan["dst_h"])
    mode = str(plan["mode"])

    if mode == "stretch":
        return cv2.resize(frame_bgr, (dst_w, dst_h), interpolation=cv2.INTER_LINEAR)

    resized = cv2.resize(
        frame_bgr,
        (int(plan["new_w"]), int(plan["new_h"])),
        interpolation=cv2.INTER_LINEAR,
    )
    canvas = np.full(
        (dst_h, dst_w, frame_bgr.shape[2]),
        int(plan["pad_value"]),
        dtype=frame_bgr.dtype,
    )
    x1 = int(round(plan["pad_x"]))
    y1 = int(round(plan["pad_y"]))
    x2 = x1 + resized.shape[1]
    y2 = y1 + resized.shape[0]
    canvas[y1:y2, x1:x2] = resized
    return canvas


def model_to_original_pixels(
    x_model: float,
    y_model: float,
    plan: Dict[str, float],
) -> Tuple[float, float]:
    if plan["mode"] == "stretch":
        x_orig = x_model / max(plan["scale_x"], 1e-6)
        y_orig = y_model / max(plan["scale_y"], 1e-6)
    else:
        scale = max(plan["scale"], 1e-6)
        x_orig = (x_model - plan["pad_x"]) / scale
        y_orig = (y_model - plan["pad_y"]) / scale
    return float(x_orig), float(y_orig)


def original_pixels_to_model(
    x_orig: float,
    y_orig: float,
    plan: Dict[str, float],
) -> Tuple[float, float]:
    if plan["mode"] == "stretch":
        x_model = x_orig * plan["scale_x"]
        y_model = y_orig * plan["scale_y"]
    else:
        x_model = x_orig * plan["scale"] + plan["pad_x"]
        y_model = y_orig * plan["scale"] + plan["pad_y"]
    return float(x_model), float(y_model)


def normalized_to_model(
    cx_norm: float,
    cy_norm: float,
    plan: Dict[str, float],
) -> Tuple[float, float]:
    x_orig = np.clip(cx_norm, 0.0, 1.0) * max(plan["orig_w"] - 1, 1)
    y_orig = np.clip(cy_norm, 0.0, 1.0) * max(plan["orig_h"] - 1, 1)
    return original_pixels_to_model(float(x_orig), float(y_orig), plan)


def model_to_normalized(
    x_model: float,
    y_model: float,
    plan: Dict[str, float],
) -> Tuple[float, float]:
    x_orig, y_orig = model_to_original_pixels(x_model, y_model, plan)
    cx_norm = x_orig / max(plan["orig_w"] - 1, 1)
    cy_norm = y_orig / max(plan["orig_h"] - 1, 1)
    return float(cx_norm), float(cy_norm)


def build_affine_from_resize_plan(plan: Dict[str, float], dtype=np.float32) -> np.ndarray:
    if plan["mode"] == "stretch":
        sx = 1.0 / max(plan["scale_x"], 1e-6)
        sy = 1.0 / max(plan["scale_y"], 1e-6)
        return np.array([[sx, 0.0, 0.0], [0.0, sy, 0.0]], dtype=dtype)

    scale = max(plan["scale"], 1e-6)
    return np.array(
        [
            [1.0 / scale, 0.0, -plan["pad_x"] / scale],
            [0.0, 1.0 / scale, -plan["pad_y"] / scale],
        ],
        dtype=dtype,
    )
