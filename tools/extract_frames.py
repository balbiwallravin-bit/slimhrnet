"""
Extract sampled frames from TS videos and save them as JPEG images.

Example:
  python tools/extract_frames.py \
      --video_root /home/lht/daqiu \
      --output_root /home/lht/daqiu_frames \
      --step 3
"""

import argparse
import os
from pathlib import Path

os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "0")

import cv2


CROP_X1, CROP_Y1 = 367, 100
CROP_X2, CROP_Y2 = 1760, 750
MODEL_W, MODEL_H = 512, 288


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_root", default="/home/lht/daqiu")
    parser.add_argument("--output_root", default="/home/lht/daqiu_frames")
    parser.add_argument("--step", type=int, default=3)
    parser.add_argument("--jpeg_quality", type=int, default=92)
    return parser.parse_args()


def iter_videos(video_root):
    return sorted(Path(video_root).rglob("*.ts"))


def main():
    args = parse_args()
    videos = iter_videos(args.video_root)
    print(f"[INFO] Found {len(videos)} .ts videos")

    if not videos:
        print("[WARN] No .ts videos found; exiting")
        return

    for video_index, video_path in enumerate(videos):
        rel = video_path.relative_to(args.video_root)
        out_dir = Path(args.output_root) / rel.with_suffix("")
        out_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"[WARN] Cannot open video: {video_path}")
            continue

        frame_idx = 0
        saved = 0
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            if frame_idx % args.step == 0:
                out_path = out_dir / f"{frame_idx:06d}.jpg"
                if not out_path.exists():
                    cropped = frame_bgr[CROP_Y1:CROP_Y2, CROP_X1:CROP_X2]
                    if cropped.size != 0:
                        resized = cv2.resize(cropped, (MODEL_W, MODEL_H), interpolation=cv2.INTER_LINEAR)
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
