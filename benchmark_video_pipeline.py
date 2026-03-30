import argparse
import csv
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from torch import nn


ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from detectors import build_detector  # noqa: E402
from models import build_model  # noqa: E402
from trackers import build_tracker  # noqa: E402
from utils.image import get_affine_transform  # noqa: E402
from utils.preprocess import process_video  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark full video pipeline with per-stage timings."
    )
    parser.add_argument(
        "--config-name",
        type=str,
        required=True,
        help="Hydra config name, for example inference_blurball or inference_slimhrnet.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Checkpoint path. If omitted, random weights are used.",
    )
    parser.add_argument(
        "--videos-dir",
        type=str,
        default=None,
        help="Directory containing input videos.",
    )
    parser.add_argument(
        "--video-glob",
        type=str,
        default="*.ts",
        help="Glob pattern under videos-dir, default: *.ts",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda"],
        help="This repository only supports CUDA inference paths.",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default="0",
        help="Comma-separated GPU ids, for example 0 or 0,1",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="benchmark_video_pipeline_results.csv",
        help="Where to save the per-video timing table.",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=None,
        help="Optional override for detector.step.",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=None,
        help="Optional override for detector.postprocessor.score_threshold.",
    )
    parser.add_argument(
        "--filter-duplicates",
        action="store_true",
        help="Enable SSIM duplicate-frame filtering in process_video.",
    )
    parser.add_argument(
        "--keep-frames",
        action="store_true",
        help="Keep extracted frame folders after benchmarking.",
    )
    return parser.parse_args()


def strip_yaml_suffix(name):
    return name[:-5] if name.endswith(".yaml") else name


def parse_gpus(text):
    values = []
    for item in text.split(","):
        item = item.strip()
        if item:
            values.append(int(item))
    if not values:
        raise ValueError("at least one GPU id is required")
    return values


def sync_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


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
    cfg.output_dir = str((ROOT / "benchmark_outputs").resolve())
    if args.model_path is not None:
        cfg.detector.model_path = str(Path(args.model_path).resolve())
    if args.step is not None:
        cfg.detector.step = args.step
    if args.score_threshold is not None:
        cfg.detector.postprocessor.score_threshold = args.score_threshold
    return cfg


def build_runtime(cfg):
    model = None
    if cfg.detector.model_path is None:
        model = build_model(cfg).to(cfg.runner.device)
        model = nn.DataParallel(model, device_ids=cfg.runner.gpus)
        model.eval()

    detector = build_detector(cfg, model=model)
    tracker = build_tracker(cfg)
    return detector, tracker


def list_videos(args):
    if args.videos_dir is None:
        raise ValueError("--videos-dir is required")
    video_dir = Path(args.videos_dir).resolve()
    videos = sorted(video_dir.glob(args.video_glob))
    if not videos:
        raise FileNotFoundError(
            "no videos found in {} with pattern {}".format(video_dir, args.video_glob)
        )
    return videos


def frame_dir_for_video(video_path):
    return video_path.parent / ("frames_" + video_path.stem)


def cleanup_frame_dir(frame_dir):
    frame_dir = Path(frame_dir)
    if frame_dir.exists() and frame_dir.is_dir() and frame_dir.name.startswith("frames_"):
        shutil.rmtree(frame_dir)


def unpack_pp_results(pp_results):
    results = {}
    hms_vis = {}
    for bid in sorted(pp_results.keys()):
        results[bid] = {}
        hms_vis[bid] = {}
        for eid in sorted(pp_results[bid].keys()):
            results[bid][eid] = []
            hms_vis[bid][eid] = []
            for scale in sorted(pp_results[bid][eid].keys()):
                scale_result = pp_results[bid][eid][scale]
                scores = scale_result["scores"]
                xys = scale_result["xys"]
                angles = scale_result.get("angles")
                lengths = scale_result.get("lengths")

                for idx, (xy, score) in enumerate(zip(xys, scores)):
                    det = {"xy": xy, "score": score, "scale": scale}
                    if angles is not None:
                        det["angle"] = angles[idx]
                    if lengths is not None:
                        det["length"] = lengths[idx]
                    results[bid][eid].append(det)

                hms_vis[bid][eid].append(
                    {
                        "hm": scale_result["hm"],
                        "scale": scale,
                        "trans": scale_result["trans"],
                    }
                )
    return results, hms_vis


def run_detector_timed(detector, input_tensor, trans):
    imgs = input_tensor.to(detector._device)

    sync_cuda()
    t0 = time.perf_counter()
    preds = detector._model(imgs)
    sync_cuda()
    model_sec = time.perf_counter() - t0

    sync_cuda()
    t1 = time.perf_counter()
    pp_results = detector._postprocessor.run(preds, trans)
    sync_cuda()
    post_sec = time.perf_counter() - t1

    batch_results, hms_vis = unpack_pp_results(pp_results)
    return batch_results, hms_vis, model_sec, post_sec


def build_affine_tensor(cfg, video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("failed to open video {}".format(video_path))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    raw_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    center = np.array([width / 2.0, height / 2.0], dtype=np.float32)
    scale = max(height, width) * 1.0
    trans = np.stack(
        [
            get_affine_transform(
                center,
                scale,
                0,
                [cfg.model.inp_width, cfg.model.inp_height],
                inv=1,
            )
            for _ in range(cfg.model.frames_out)
        ],
        axis=0,
    )
    return torch.tensor(trans)[None, :], raw_frames


def run_single_video(cfg, detector, tracker, video_path, filter_duplicates=False):
    video_path = Path(video_path).resolve()
    frame_dir = frame_dir_for_video(video_path)
    if frame_dir.exists():
        cleanup_frame_dir(frame_dir)

    total_start = time.perf_counter()

    t0 = time.perf_counter()
    processed_frame_dir = process_video(str(video_path), filter=filter_duplicates)
    extract_sec = time.perf_counter() - t0

    processed_frame_dir = Path(processed_frame_dir)
    imgs_paths = sorted(processed_frame_dir.glob("*.png"))
    trans, raw_frames = build_affine_tensor(cfg, video_path)

    preprocess_frame = T.Compose(
        [
            T.ToPILImage(),
            T.Resize((cfg.model.inp_height, cfg.model.inp_width)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    step = cfg.detector.step
    det_results = defaultdict(list)
    img_paths_buffer = []
    frames_buffer = []

    prep_tensor_sec = 0.0
    model_sec = 0.0
    post_sec = 0.0
    windows = 0

    for img_path in imgs_paths:
        t_read = time.perf_counter()
        frame = cv2.imread(str(img_path))
        prep_tensor_sec += time.perf_counter() - t_read

        frames_buffer.append(frame)
        img_paths_buffer.append(str(img_path))

        if len(frames_buffer) == cfg.model.frames_in:
            t_prep = time.perf_counter()
            frames_processed = [preprocess_frame(f) for f in frames_buffer]
            input_tensor = torch.cat(frames_processed, dim=0).unsqueeze(0)
            prep_tensor_sec += time.perf_counter() - t_prep

            batch_results, _, model_tmp, post_tmp = run_detector_timed(
                detector, input_tensor, trans
            )
            model_sec += model_tmp
            post_sec += post_tmp
            windows += 1

            for ie in batch_results[0].keys():
                path = img_paths_buffer[ie]
                preds = batch_results[0][ie]
                det_results[path].extend(preds)

            if step == 1:
                frames_buffer.pop(0)
                img_paths_buffer.pop(0)
            elif step == 3:
                img_paths_buffer = []
                frames_buffer = []
            else:
                raise ValueError("unsupported detector.step={}".format(step))

    tracker.refresh()
    sync_cuda()
    t_track = time.perf_counter()
    result_dict = {}
    for img_path, preds in det_results.items():
        result_dict[img_path] = tracker.update(preds)
    tracking_sec = time.perf_counter() - t_track

    total_sec = time.perf_counter() - total_start
    unique_frames = len(imgs_paths)
    predicted_frames = len(result_dict)

    return {
        "video": str(video_path),
        "raw_frames": raw_frames,
        "unique_frames": unique_frames,
        "predicted_frames": predicted_frames,
        "windows": windows,
        "extract_sec": extract_sec,
        "prep_tensor_sec": prep_tensor_sec,
        "model_sec": model_sec,
        "post_sec": post_sec,
        "tracking_sec": tracking_sec,
        "total_sec": total_sec,
        "fps_total": (predicted_frames / total_sec) if total_sec > 0 else 0.0,
        "prep_ms_per_frame": (
            (extract_sec + prep_tensor_sec) * 1000.0 / max(unique_frames, 1)
        ),
        "model_ms_per_window": (model_sec * 1000.0 / max(windows, 1)),
        "post_ms_per_window": (post_sec * 1000.0 / max(windows, 1)),
        "tracking_ms_per_frame": (tracking_sec * 1000.0 / max(predicted_frames, 1)),
    }, processed_frame_dir


def write_csv(rows, output_csv):
    output_path = Path(output_csv).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "video",
        "raw_frames",
        "unique_frames",
        "predicted_frames",
        "windows",
        "extract_sec",
        "prep_tensor_sec",
        "model_sec",
        "post_sec",
        "tracking_sec",
        "total_sec",
        "fps_total",
        "prep_ms_per_frame",
        "model_ms_per_window",
        "post_ms_per_window",
        "tracking_ms_per_frame",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return output_path


def print_summary(rows):
    total_raw_frames = sum(row["raw_frames"] for row in rows)
    total_unique_frames = sum(row["unique_frames"] for row in rows)
    total_predicted_frames = sum(row["predicted_frames"] for row in rows)
    total_windows = sum(row["windows"] for row in rows)
    total_extract = sum(row["extract_sec"] for row in rows)
    total_prep_tensor = sum(row["prep_tensor_sec"] for row in rows)
    total_model = sum(row["model_sec"] for row in rows)
    total_post = sum(row["post_sec"] for row in rows)
    total_tracking = sum(row["tracking_sec"] for row in rows)
    total_time = sum(row["total_sec"] for row in rows)

    mean_video_fps = sum(row["fps_total"] for row in rows) / max(len(rows), 1)
    weighted_fps = total_predicted_frames / total_time if total_time > 0 else 0.0

    print("\nPer-video results")
    for row in rows:
        print(
            "{} | total {:.2f}s | fps {:.2f} | prep {:.2f} ms/frame | model {:.2f} ms/window | post {:.2f} ms/window | tracking {:.2f} ms/frame".format(
                Path(row["video"]).name,
                row["total_sec"],
                row["fps_total"],
                row["prep_ms_per_frame"],
                row["model_ms_per_window"],
                row["post_ms_per_window"],
                row["tracking_ms_per_frame"],
            )
        )

    print("\nOverall summary")
    print("videos: {}".format(len(rows)))
    print("raw frames: {}".format(total_raw_frames))
    print("unique frames: {}".format(total_unique_frames))
    print("predicted frames: {}".format(total_predicted_frames))
    print("windows: {}".format(total_windows))
    print("extract total: {:.2f}s".format(total_extract))
    print("tensor prep total: {:.2f}s".format(total_prep_tensor))
    print("model total: {:.2f}s".format(total_model))
    print("postprocess total: {:.2f}s".format(total_post))
    print("tracking total: {:.2f}s".format(total_tracking))
    print("pipeline total: {:.2f}s".format(total_time))
    print("mean video FPS: {:.2f}".format(mean_video_fps))
    print("weighted FPS: {:.2f}".format(weighted_fps))
    print(
        "avg preprocess: {:.2f} ms/frame".format(
            (total_extract + total_prep_tensor) * 1000.0 / max(total_unique_frames, 1)
        )
    )
    print(
        "avg model forward: {:.2f} ms/window".format(
            total_model * 1000.0 / max(total_windows, 1)
        )
    )
    print(
        "avg postprocess: {:.2f} ms/window".format(
            total_post * 1000.0 / max(total_windows, 1)
        )
    )
    print(
        "avg tracking: {:.2f} ms/frame".format(
            total_tracking * 1000.0 / max(total_predicted_frames, 1)
        )
    )


def main():
    args = parse_args()
    cfg = get_config(args)
    detector, tracker = build_runtime(cfg)
    videos = list_videos(args)

    rows = []
    for video_path in videos:
        row, frame_dir = run_single_video(
            cfg,
            detector,
            tracker,
            video_path,
            filter_duplicates=args.filter_duplicates,
        )
        rows.append(row)
        if not args.keep_frames:
            cleanup_frame_dir(frame_dir)

    output_csv = write_csv(rows, args.output_csv)
    print_summary(rows)
    print("\nSaved CSV: {}".format(output_csv))


if __name__ == "__main__":
    main()
