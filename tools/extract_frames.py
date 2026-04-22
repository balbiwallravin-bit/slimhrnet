"""
Extract sampled frames from videos and save them as JPEG images.

This helper now shares the same resize policy as pseudo-label generation so the
student sees exactly the same 512x288 inputs during training.
"""

import argparse
import os
from pathlib import Path

os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "0")

import cv2


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in os.sys.path:
    os.sys.path.insert(0, str(SRC_DIR))

from utils.resize_ops import MODEL_H, MODEL_W, build_resize_plan, resize_frame_with_plan  # noqa: E402


VIDEO_EXTENSIONS = (".ts", ".mp4", ".avi", ".mov", ".mkv")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_root", default="/home/lht/daqiu")
    parser.add_argument(
        "--output_root",
        default="/home/lht/codexwork/slimhrnet/data_maked/frames",
    )
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--jpeg_quality", type=int, default=92)
    parser.add_argument(
        "--resize_mode",
        default="stretch",
        choices=["stretch", "letterbox"],
    )
    parser.add_argument("--pad_value", type=int, default=0)
    return parser.parse_args()


def iter_videos(video_root):
    root = Path(video_root)
    videos = []
    for ext in VIDEO_EXTENSIONS:
        videos.extend(root.rglob(f"*{ext}"))
        videos.extend(root.rglob(f"*{ext.upper()}"))
    return sorted({video.resolve() for video in videos})


def main():
    args = parse_args()
    videos = iter_videos(args.video_root)
    print(f"[INFO] Found {len(videos)} videos")

    if not videos:
        print("[WARN] No videos found; exiting")
        return

    for video_index, video_path in enumerate(videos):
        rel = video_path.relative_to(Path(args.video_root).resolve())
        out_dir = Path(args.output_root) / rel.with_suffix("")
        out_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"[WARN] Cannot open video: {video_path}")
            continue

        frame_idx = 0
        saved = 0
        resize_plan = None
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            if frame_idx % args.step == 0:
                out_path = out_dir / f"{frame_idx:06d}.jpg"
                if not out_path.exists():
                    if resize_plan is None:
                        resize_plan = build_resize_plan(
                            frame_bgr.shape[1],
                            frame_bgr.shape[0],
                            dst_w=MODEL_W,
                            dst_h=MODEL_H,
                            mode=args.resize_mode,
                            pad_value=args.pad_value,
                        )
                    resized = resize_frame_with_plan(frame_bgr, resize_plan)
                    cv2.imwrite(
                        str(out_path),
                        resized,
                        [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality],
                    )
                    saved += 1
            frame_idx += 1

        cap.release()

        if (video_index + 1) % 100 == 0 or video_index == len(videos) - 1:
            print(
                f"[{video_index + 1}/{len(videos)}] "
                f"saved={saved} | "
                f"video={video_path.name}"
            )

    print(f"[Done] Extracted frames into: {args.output_root}")


if __name__ == "__main__":
    main()
