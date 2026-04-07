"""
Sample videos from a directory, run tracking inference, save visualized videos,
and write a speed summary CSV.

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
import random
import shutil
import sys
import time
from pathlib import Path

os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "0")
os.environ.setdefault("AV_LOG_FORCE_NOCOLOR", "1")

import cv2
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from detectors import build_detector  # noqa: E402
from runners.inference import inference_video  # noqa: E402
from trackers import build_tracker  # noqa: E402
from utils.preprocess import process_video  # noqa: E402


VIDEO_EXTENSIONS = [".mp4", ".ts", ".avi", ".mov", ".mkv"]


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
    parser.add_argument("--keep-temp-frames", action="store_true")
    parser.add_argument("--filter-duplicates", action="store_true")
    parser.add_argument("--save-heatmaps", action="store_true")
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
    cfg.runner.vis_result = True
    cfg.runner.vis_hm = args.save_heatmaps
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
    videos = sorted({video.resolve() for video in videos})
    return videos


def select_videos(args):
    rng = random.Random(args.seed)
    video_dir = Path(args.video_dir).resolve()
    candidates = list_candidate_videos(video_dir)
    if not candidates:
        raise FileNotFoundError(f"No videos found under {video_dir}")

    selected = []
    include_video = None
    if args.include_video:
        include_video = Path(args.include_video).resolve()
        if not include_video.exists():
            raise FileNotFoundError(f"include video not found: {include_video}")
        selected.append(include_video)

    remaining = [video for video in candidates if video != include_video]
    if args.num_random > 0:
        sample_count = min(args.num_random, len(remaining))
        selected.extend(rng.sample(remaining, sample_count))

    if not selected:
        raise ValueError("No videos selected")

    selected = sorted(dict.fromkeys(selected))
    return selected


def count_video_frames(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frame_count


def count_csv_rows(csv_path):
    if not csv_path.exists():
        return 0
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return sum(1 for _ in reader)


def cleanup_dir(path):
    path = Path(path)
    if path.exists() and path.is_dir():
        shutil.rmtree(path)


def safe_relative(path, root):
    try:
        return path.resolve().relative_to(root.resolve())
    except ValueError:
        return Path(path.name)


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
        "preprocess_sec",
        "inference_sec",
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
        video_out_dir.mkdir(parents=True, exist_ok=True)

        temp_frame_dir = None
        temp_vis_dir = video_out_dir / "result_frames"
        temp_hm_dir = video_out_dir / "heatmaps"
        traj_copy = video_out_dir / "traj.csv"
        result_video = video_out_dir / "result.mp4"

        row = {
            "video": str(video_path),
            "status": "ok",
            "error": "",
            "raw_frames": 0,
            "unique_frames": 0,
            "predicted_frames": 0,
            "preprocess_sec": 0.0,
            "inference_sec": 0.0,
            "wall_total_sec": 0.0,
            "fps_inference": 0.0,
            "fps_wall": 0.0,
            "result_video": "",
            "traj_csv": "",
        }

        wall_start = time.perf_counter()
        try:
            row["raw_frames"] = count_video_frames(video_path)
            cleanup_dir(temp_vis_dir)
            cleanup_dir(temp_hm_dir)

            prep_start = time.perf_counter()
            temp_frame_dir = Path(process_video(str(video_path), filter=args.filter_duplicates))
            row["preprocess_sec"] = time.perf_counter() - prep_start
            row["unique_frames"] = len(list(temp_frame_dir.glob("*.png")))

            infer_ret = inference_video(
                detector,
                tracker,
                str(video_path),
                str(temp_frame_dir),
                cfg,
                vis_frame_dir=str(temp_vis_dir),
                vis_hm_dir=str(temp_hm_dir) if args.save_heatmaps else None,
            )
            row["inference_sec"] = float(infer_ret.get("t_elapsed", 0.0))
            row["wall_total_sec"] = time.perf_counter() - wall_start

            traj_src = temp_frame_dir / "traj.csv"
            if traj_src.exists():
                shutil.copy2(traj_src, traj_copy)
                row["traj_csv"] = str(traj_copy)
                row["predicted_frames"] = count_csv_rows(traj_copy)

            generated_video = Path(str(temp_vis_dir) + ".mp4")
            if generated_video.exists():
                if result_video.exists():
                    result_video.unlink()
                shutil.move(str(generated_video), str(result_video))
                row["result_video"] = str(result_video)

            if row["inference_sec"] > 0:
                row["fps_inference"] = row["predicted_frames"] / row["inference_sec"]
            if row["wall_total_sec"] > 0:
                row["fps_wall"] = row["predicted_frames"] / row["wall_total_sec"]

            print(
                f"[OK] {video_path.name} | "
                f"predicted_frames={row['predicted_frames']} | "
                f"fps_inference={row['fps_inference']:.2f} | "
                f"fps_wall={row['fps_wall']:.2f}"
            )
        except Exception as exc:
            row["status"] = "error"
            row["error"] = str(exc)
            row["wall_total_sec"] = time.perf_counter() - wall_start
            print(f"[ERROR] {video_path}: {exc}")
        finally:
            if not args.keep_temp_frames and temp_frame_dir is not None:
                cleanup_dir(temp_frame_dir)
            if not args.keep_temp_frames:
                cleanup_dir(temp_vis_dir)
                if args.save_heatmaps:
                    cleanup_dir(temp_hm_dir)

        rows.append(row)

    summary_path = write_summary(rows, summary_csv)
    print(f"\n[Done] Summary CSV: {summary_path}")
    print(f"[Done] Selected videos list: {selected_txt}")


if __name__ == "__main__":
    main()
