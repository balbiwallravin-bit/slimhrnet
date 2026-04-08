"""
Sample videos from a directory, run tracking inference with async sequential video reads,
optionally skip nearly static frames, optionally render a result video as a post-process
step, and write a speed summary CSV.

Example:
  python tools/test_videos.py \
      --config-name inference_slimhrnet \
      --model-path checkpoints_balanced/best.pth \
      --video-dir /home/lht/ceshi \
      --include-video /home/lht/ceshi/D1_S20251015112330_E20251015112400.mp4 \
      --num-random 5 \
      --gpus 0 \
      --output-dir video_test_outputs
"""

import argparse
import csv
import os
import queue
import random
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "0")
os.environ.setdefault("AV_LOG_FORCE_NOCOLOR", "1")

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from detectors import build_detector  # noqa: E402
from trackers import build_tracker  # noqa: E402


VIDEO_EXTENSIONS = [".mp4", ".ts", ".avi", ".mov", ".mkv"]
CROP_X1, CROP_Y1 = 367, 100
CROP_X2, CROP_Y2 = 1760, 750
DIFF_W, DIFF_H = 96, 54
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)


class AsyncVideoReader:
    def __init__(self, video_path, queue_size=8):
        self.video_path = str(video_path)
        self.queue_size = max(int(queue_size), 1)
        self.frame_queue = queue.Queue(maxsize=self.queue_size)
        self.stop_event = threading.Event()
        self.error = None
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"failed to open video: {self.video_path}")
        self.thread = threading.Thread(target=self._worker, daemon=True)

    def start(self):
        self.thread.start()
        return self

    def _put_with_retry(self, item):
        while not self.stop_event.is_set():
            try:
                self.frame_queue.put(item, timeout=0.1)
                return
            except queue.Full:
                continue

    def _worker(self):
        frame_idx = 0
        try:
            while not self.stop_event.is_set():
                ret, frame = self.cap.read()
                if not ret:
                    break
                self._put_with_retry((frame_idx, frame))
                frame_idx += 1
        except Exception as exc:
            self.error = exc
        finally:
            self.cap.release()
            self._put_with_retry(None)

    def read(self):
        item = self.frame_queue.get()
        if self.error is not None:
            raise self.error
        return item

    def close(self):
        self.stop_event.set()
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)
        try:
            self.cap.release()
        except Exception:
            pass


def parse_args():
    parser = argparse.ArgumentParser(description="Batch video test helper for visual inspection.")
    parser.add_argument("--config-name", default="inference_slimhrnet")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--video-dir", required=True)
    parser.add_argument("--include-video", default=None)
    parser.add_argument("--num-random", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--score-threshold", type=float, default=None)
    parser.add_argument("--output-dir", default="video_test_outputs")
    parser.add_argument("--summary-csv", default=None)
    parser.add_argument("--skip-render", action="store_true", help="Skip post-rendered result.mp4 generation.")
    parser.add_argument("--decode-queue-size", type=int, default=8, help="Frame prefetch queue size for async decoding.")
    parser.add_argument("--static-diff-threshold", type=float, default=3.0, help="Mean absolute grayscale diff threshold (0-255 scale).")
    parser.add_argument("--disable-static-skip", action="store_true", help="Disable frame-difference based static-frame skipping.")
    return parser.parse_args()


def strip_yaml_suffix(name):
    return name[:-5] if name.endswith(".yaml") else name


def parse_gpus(text):
    gpu_ids = []
    for item in text.split(","):
        item = item.strip()
        if item:
            gpu_ids.append(int(item))
    if not gpu_ids:
        raise ValueError("at least one GPU id is required")
    return gpu_ids


def get_config(args):
    config_dir = str((ROOT / "src" / "configs").resolve())
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name=strip_yaml_suffix(args.config_name))

    OmegaConf.set_struct(cfg, False)
    cfg.runner.device = args.device
    cfg.runner.gpus = parse_gpus(args.gpus)
    cfg.runner.vis_result = False
    cfg.runner.vis_hm = False
    cfg.runner.vis_traj = False
    cfg.detector.model_path = str(Path(args.model_path).resolve())
    if args.step is not None:
        cfg.detector.step = args.step
    if args.score_threshold is not None:
        cfg.detector.postprocessor.score_threshold = args.score_threshold
    return cfg


def build_runtime(cfg):
    detector = build_detector(cfg, model=None)
    tracker = build_tracker(cfg)
    return detector, tracker


def list_candidate_videos(video_dir):
    video_dir = Path(video_dir).resolve()
    videos = []
    for extension in VIDEO_EXTENSIONS:
        videos.extend(video_dir.rglob(f"*{extension}"))
        videos.extend(video_dir.rglob(f"*{extension.upper()}"))
    return sorted({video.resolve() for video in videos})


def select_videos(args):
    rng = random.Random(args.seed)
    candidates = list_candidate_videos(args.video_dir)
    if not candidates:
        raise FileNotFoundError(f"No videos found under {Path(args.video_dir).resolve()}")

    selected = []
    include_video = None
    if args.include_video:
        include_video = Path(args.include_video).resolve()
        if not include_video.exists():
            raise FileNotFoundError(f"include video not found: {include_video}")
        selected.append(include_video)

    remaining = [video for video in candidates if video != include_video]
    if args.num_random > 0:
        selected.extend(rng.sample(remaining, min(args.num_random, len(remaining))))

    if not selected:
        raise ValueError("No videos selected")

    return sorted(dict.fromkeys(selected))


def sync_if_needed(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def safe_relative(path, root):
    try:
        return path.resolve().relative_to(Path(root).resolve())
    except ValueError:
        return Path(path.name)


def safe_coord(value):
    if value is None or not np.isfinite(value):
        return 0
    return int(round(float(value)))


def safe_score(value):
    if value is None or not np.isfinite(value):
        return 0.0
    return float(value)


def build_crop_affine_tensor(cfg):
    inp_w = int(cfg.model.inp_width)
    inp_h = int(cfg.model.inp_height)
    crop_w = CROP_X2 - CROP_X1
    crop_h = CROP_Y2 - CROP_Y1
    sx = (crop_w - 1) / max(inp_w - 1, 1)
    sy = (crop_h - 1) / max(inp_h - 1, 1)
    trans_single = np.array(
        [
            [sx, 0.0, float(CROP_X1)],
            [0.0, sy, float(CROP_Y1)],
        ],
        dtype=np.float32,
    )
    trans = np.stack([trans_single for _ in range(int(cfg.model.frames_out))], axis=0)
    return torch.tensor(trans, dtype=torch.float32).unsqueeze(0)


def preprocess_frame_gpu(frame_bgr, device, input_h, input_w, mean, std):
    raw_cpu = torch.from_numpy(frame_bgr).permute(2, 0, 1).contiguous()
    if device.type == "cuda":
        raw_cpu = raw_cpu.pin_memory()
    raw_gpu = raw_cpu.to(device, non_blocking=(device.type == "cuda"))

    _, height, width = raw_gpu.shape
    x1 = max(0, min(CROP_X1, width - 1))
    x2 = max(x1 + 1, min(CROP_X2, width))
    y1 = max(0, min(CROP_Y1, height - 1))
    y2 = max(y1 + 1, min(CROP_Y2, height))

    cropped = raw_gpu[:, y1:y2, x1:x2]
    rgb = cropped[[2, 1, 0], ...]
    resized = TF.resize(
        rgb,
        [input_h, input_w],
        interpolation=TF.InterpolationMode.BILINEAR,
        antialias=False,
    )
    resized = resized.float() / 255.0
    resized = (resized - mean) / std
    return resized


def build_diff_frame(frame_bgr):
    cropped = frame_bgr[CROP_Y1:CROP_Y2, CROP_X1:CROP_X2]
    if cropped.size == 0:
        cropped = frame_bgr
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (DIFF_W, DIFF_H), interpolation=cv2.INTER_AREA)


def mean_frame_diff(prev_frame, cur_frame):
    return float(np.mean(cv2.absdiff(prev_frame, cur_frame)))


def write_traj_csv(rows, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["Frame", "X", "Y", "Visibility", "Score"]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def render_video_postprocess(video_path, traj_rows, output_path):
    pred_map = {int(row["Frame"]): row for row in traj_rows}

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video for rendering: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"failed to open video writer: {output_path}")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        pred = pred_map.get(frame_idx)
        if pred is not None and int(pred["Visibility"]) == 1:
            x = safe_coord(pred["X"])
            y = safe_coord(pred["Y"])
            x = max(0, min(x, width - 1))
            y = max(0, min(y, height - 1))
            score = safe_score(pred.get("Score", 0.0))
            cv2.circle(frame, (x, y), 8, (0, 0, 255), 2)
            cv2.putText(
                frame,
                f"ball {score:.2f}",
                (max(0, x - 30), max(20, y - 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()


def build_result_row(frame_id, result):
    return {
        "Frame": frame_id,
        "X": safe_coord(result.get("x")),
        "Y": safe_coord(result.get("y")),
        "Visibility": int(bool(result.get("visi", False))),
        "Score": round(safe_score(result.get("score")), 6),
    }


def run_single_video(cfg, detector, tracker, video_path, output_dir, skip_render, decode_queue_size, static_diff_threshold, disable_static_skip):
    video_path = Path(video_path).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    traj_csv = output_dir / "traj.csv"
    result_video = output_dir / "result.mp4"

    device = torch.device(detector._device)
    mean = IMAGENET_MEAN.to(device)
    std = IMAGENET_STD.to(device)
    input_h = int(cfg.model.inp_height)
    input_w = int(cfg.model.inp_width)
    step = int(cfg.detector.step)
    frames_in = int(cfg.model.frames_in)
    affine = build_crop_affine_tensor(cfg)

    if step != 1 and not disable_static_skip:
        print(f"[WARN] static-skip is optimized for step=1, but detector.step={step}; disabling it")
        disable_static_skip = True

    reader = AsyncVideoReader(video_path, queue_size=decode_queue_size).start()
    det_results = defaultdict(list)
    frame_ids_buffer = []
    frame_tensors_buffer = []
    skipped_static_ids = set()
    prev_diff_frame = None
    last_preprocessed = None
    actual_frame_count = 0
    preprocess_sec = 0.0
    decode_wait_sec = 0.0
    had_model_output = False

    infer_start = time.perf_counter()
    try:
        while True:
            t_wait = time.perf_counter()
            item = reader.read()
            decode_wait_sec += time.perf_counter() - t_wait
            if item is None:
                break

            frame_id, frame_bgr = item
            actual_frame_count = frame_id + 1

            diff_frame = build_diff_frame(frame_bgr)
            static_skip = False
            if (
                not disable_static_skip
                and prev_diff_frame is not None
                and had_model_output
                and last_preprocessed is not None
            ):
                static_skip = mean_frame_diff(prev_diff_frame, diff_frame) < static_diff_threshold
            prev_diff_frame = diff_frame

            if static_skip:
                frame_tensor = last_preprocessed
            else:
                sync_if_needed(device)
                t_pre = time.perf_counter()
                frame_tensor = preprocess_frame_gpu(frame_bgr, device, input_h, input_w, mean, std)
                sync_if_needed(device)
                preprocess_sec += time.perf_counter() - t_pre
                last_preprocessed = frame_tensor

            frame_ids_buffer.append(frame_id)
            frame_tensors_buffer.append(frame_tensor)

            if len(frame_tensors_buffer) == frames_in:
                if static_skip and step == 1:
                    skipped_static_ids.add(frame_id)
                    frame_ids_buffer.pop(0)
                    frame_tensors_buffer.pop(0)
                    continue

                input_tensor = torch.cat(frame_tensors_buffer, dim=0).unsqueeze(0)
                batch_results, _ = detector.run_tensor(input_tensor, affine)
                had_model_output = True
                for elem_idx in sorted(batch_results[0].keys()):
                    det_results[frame_ids_buffer[elem_idx]].extend(batch_results[0][elem_idx])

                if step == 1:
                    frame_ids_buffer.pop(0)
                    frame_tensors_buffer.pop(0)
                elif step == 3:
                    frame_ids_buffer = []
                    frame_tensors_buffer = []
                else:
                    raise ValueError(f"unsupported detector.step={step}")
    finally:
        reader.close()

    tracker.refresh()
    track_start = time.perf_counter()
    traj_rows = []
    last_row = None
    for frame_id in range(actual_frame_count):
        if frame_id in det_results:
            result = tracker.update(det_results[frame_id])
            row = build_result_row(frame_id, result)
            traj_rows.append(row)
            last_row = row
        elif frame_id in skipped_static_ids and last_row is not None:
            copied = dict(last_row)
            copied["Frame"] = frame_id
            traj_rows.append(copied)
            last_row = copied
    track_sec = time.perf_counter() - track_start
    inference_sec = time.perf_counter() - infer_start

    write_traj_csv(traj_rows, traj_csv)

    render_sec = 0.0
    if not skip_render:
        render_start = time.perf_counter()
        render_video_postprocess(video_path, traj_rows, result_video)
        render_sec = time.perf_counter() - render_start

    wall_total_sec = inference_sec + render_sec
    predicted_frames = len(traj_rows)

    row = {
        "video": str(video_path),
        "status": "ok",
        "error": "",
        "raw_frames": actual_frame_count,
        "unique_frames": actual_frame_count,
        "predicted_frames": predicted_frames,
        "skipped_static_frames": len(skipped_static_ids),
        "decode_wait_sec": round(decode_wait_sec, 6),
        "preprocess_sec": round(preprocess_sec, 6),
        "inference_sec": round(inference_sec, 6),
        "tracking_sec": round(track_sec, 6),
        "render_sec": round(render_sec, 6),
        "wall_total_sec": round(wall_total_sec, 6),
        "fps_inference": round(predicted_frames / max(inference_sec, 1e-6), 4),
        "fps_wall": round(predicted_frames / max(wall_total_sec, 1e-6), 4),
        "result_video": "" if skip_render else str(result_video),
        "traj_csv": str(traj_csv),
    }
    return row


def write_summary(rows, output_csv):
    output_path = Path(output_csv).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "video",
        "status",
        "error",
        "raw_frames",
        "unique_frames",
        "predicted_frames",
        "skipped_static_frames",
        "decode_wait_sec",
        "preprocess_sec",
        "inference_sec",
        "tracking_sec",
        "render_sec",
        "wall_total_sec",
        "fps_inference",
        "fps_wall",
        "result_video",
        "traj_csv",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def main():
    args = parse_args()
    cfg = get_config(args)
    detector, tracker = build_runtime(cfg)

    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    summary_csv = Path(args.summary_csv).resolve() if args.summary_csv else output_root / "summary.csv"

    selected = select_videos(args)
    selected_txt = output_root / "selected_videos.txt"
    selected_txt.write_text("\n".join(str(video) for video in selected) + "\n", encoding="utf-8")

    print(f"[INFO] Selected {len(selected)} videos")
    for video in selected:
        print(f"  - {video}")

    rows = []
    video_root = Path(args.video_dir).resolve()
    for video_path in selected:
        rel = safe_relative(video_path, video_root)
        video_out_dir = output_root / rel.with_suffix("")
        try:
            row = run_single_video(
                cfg,
                detector,
                tracker,
                video_path,
                video_out_dir,
                skip_render=args.skip_render,
                decode_queue_size=args.decode_queue_size,
                static_diff_threshold=args.static_diff_threshold,
                disable_static_skip=args.disable_static_skip,
            )
            print(
                f"[OK] {Path(video_path).name} | "
                f"predicted_frames={row['predicted_frames']} | "
                f"skipped_static={row['skipped_static_frames']} | "
                f"fps_inference={row['fps_inference']:.2f} | "
                f"fps_wall={row['fps_wall']:.2f}"
            )
        except Exception as exc:
            row = {
                "video": str(Path(video_path).resolve()),
                "status": "error",
                "error": str(exc),
                "raw_frames": 0,
                "unique_frames": 0,
                "predicted_frames": 0,
                "skipped_static_frames": 0,
                "decode_wait_sec": 0.0,
                "preprocess_sec": 0.0,
                "inference_sec": 0.0,
                "tracking_sec": 0.0,
                "render_sec": 0.0,
                "wall_total_sec": 0.0,
                "fps_inference": 0.0,
                "fps_wall": 0.0,
                "result_video": "",
                "traj_csv": "",
            }
            print(f"[ERROR] {video_path}: {exc}")
        rows.append(row)

    summary_path = write_summary(rows, summary_csv)
    print(f"\n[Done] Summary CSV: {summary_path}")
    print(f"[Done] Selected videos list: {selected_txt}")


if __name__ == "__main__":
    main()
