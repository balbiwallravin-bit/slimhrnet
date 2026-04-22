# SlimHRNet Distillation Workflow

This document records the custom pseudo-label, distillation, testing, and benchmarking flow added in this fork.

## What Was Added

The following scripts were added or updated for the custom SlimHRNet student workflow:

- `tools/auto_label.py`: generate pseudo labels from TS videos with the BlurBall teacher.
- `tools/clean_labels.py`: clean the raw pseudo labels.
- `tools/split_dataset.py`: split labels by video into train/val/test CSVs.
- `tools/extract_frames.py`: extract cropped+resized JPEG frames for stable training.
- `src/datasets/ball_dataset.py`: dataset for distilled training, preferring extracted JPEGs and falling back to videos.
- `tools/train_distill.py`: multi-GPU distillation training.
- `tools/test_videos.py`: batch video test helper with async sequential decoding, GPU crop/resize, optional static-frame skipping, post-rendered `result.mp4`, and per-video speed summary.
- `benchmark_speed.py`: synthetic preprocessing/model/postprocess latency benchmark.
- `benchmark_video_pipeline.py`: end-to-end timing benchmark on real videos.
- `tools/profile_inference.py`: day-1 inference profiling helper for FP32 / FP16 / channels_last / torch.compile.

## Current V2 Retrain Status

The repository now contains the first-pass implementation for the new retraining plan:

- pseudo labels can store `x_model`, `y_model`, `score`, `angle_deg`, `length_input_px`, `length_px`
- label cleaning is now segment-based, with low-score breaking, velocity-residual splitting, and optional interpolation
- the student dataset can consume interpolated weights and auxiliary blur labels
- a `slimhrnet_v2` config is available with widened high-resolution branches and optional angle/length heads
- `gaussian_postprocessor.py` supports soft-argmax decoding
- a `kalman_ball` tracker config is available for the new inference path

What is still intentionally not automated in-repo:

- manual suspicious-segment screening UI
- TensorRT engine export / runtime wrapper
- GT labeling workflow

## Why The Frame Extraction Step Exists

Random `cv2.VideoCapture(...); cap.set(CAP_PROP_POS_FRAMES, idx)` seeks are fragile on `.ts` videos and can trigger FFmpeg warnings such as missing reference frames. The custom workflow therefore prefers:

1. Extract JPEGs once with `tools/extract_frames.py`.
2. Train from JPEGs with `--frame_root`.
3. Fall back to video decoding only if a JPEG is missing.

This makes training more stable and usually faster.

## Training Workflow

Assumed paths used below:

- repo: `/home/lht/codexwork/slimhrnet`
- teacher weights: `/home/lht/codexwork/blurball-mainyy/blurball_best.pth`
- raw videos: `/home/lht/daqiu`
- generated workspace: `/home/lht/codexwork/slimhrnet/data_maked`
- extracted JPEGs: `/home/lht/codexwork/slimhrnet/data_maked/frames`

Activate the environment first:

```bash
cd /home/lht/codexwork/slimhrnet
conda activate blurball
```

### 1. Generate raw pseudo labels

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONUNBUFFERED=1 \
OPENCV_FFMPEG_LOGLEVEL=0 \
AV_LOG_FORCE_NOCOLOR=1 \
python tools/auto_label.py \
    --video_root /home/lht/daqiu \
    --weights /home/lht/codexwork/blurball-mainyy/blurball_best.pth \
    --output_csv /home/lht/codexwork/slimhrnet/data_maked/pseudo_labels/raw/pseudo_labels_raw.csv \
    --step 1 \
    --resize_mode stretch \
    --score_threshold 0.5 \
    --device cuda | tee auto_label.log
```

### 2. Clean the pseudo labels

Segment-based cleaning settings used in practice:

```bash
python tools/clean_labels.py \
    --input_csv /home/lht/codexwork/slimhrnet/data_maked/pseudo_labels/raw/pseudo_labels_raw.csv \
    --output_csv /home/lht/codexwork/slimhrnet/data_maked/pseudo_labels/clean/pseudo_labels_clean_v2.csv \
    --min_score 0.50 \
    --max_residual_px 150 \
    --min_segment_frames 30 \
    --max_segment_frames 480 \
    --interp_max_gap 2 | tee clean_labels_v2.log
```

### 3. Split train / val / test by video

```bash
python tools/split_dataset.py \
    --input_csv /home/lht/codexwork/slimhrnet/data_maked/pseudo_labels/clean/pseudo_labels_clean_v2.csv \
    --output_dir /home/lht/codexwork/slimhrnet/data_maked/splits_v2 | tee split_dataset_v2.log
```

### 4. Extract cropped JPEG frames

```bash
PYTHONUNBUFFERED=1 \
OPENCV_FFMPEG_LOGLEVEL=0 \
AV_LOG_FORCE_NOCOLOR=1 \
python tools/extract_frames.py \
    --video_root /home/lht/daqiu \
    --output_root /home/lht/codexwork/slimhrnet/data_maked/frames \
    --step 1 \
    --resize_mode stretch | tee extract_frames.log
```

### 5. Train the student with distillation

Example using 3 GPUs:

```bash
CUDA_VISIBLE_DEVICES=4,5,7 \
PYTHONUNBUFFERED=1 \
OPENCV_FFMPEG_LOGLEVEL=0 \
AV_LOG_FORCE_NOCOLOR=1 \
python tools/train_distill.py \
    --train_csv /home/lht/codexwork/slimhrnet/data_maked/splits_v2/train.csv \
    --val_csv /home/lht/codexwork/slimhrnet/data_maked/splits_v2/val.csv \
    --teacher_weights /home/lht/codexwork/blurball-mainyy/blurball_best.pth \
    --student_config inference_slimhrnet_v2 \
    --save_dir /home/lht/codexwork/slimhrnet/data_maked/checkpoints_v2 \
    --video_root /home/lht/daqiu \
    --frame_root /home/lht/codexwork/slimhrnet/data_maked/frames \
    --epochs 80 \
    --batch_size 24 \
    --lr 1e-3 \
    --num_workers 8 \
    --alpha 0.7 \
    --beta 0.3 \
    --sigma 3.0 \
    --dynamic_sigma \
    --ohem_ratio 0.3 \
    --ohem_start_epoch 40 \
    --device cuda | tee train_distill_balanced.log
```

Resume training:

```bash
CUDA_VISIBLE_DEVICES=4,5,7 \
PYTHONUNBUFFERED=1 \
OPENCV_FFMPEG_LOGLEVEL=0 \
AV_LOG_FORCE_NOCOLOR=1 \
python tools/train_distill.py \
    --train_csv /home/lht/codexwork/slimhrnet/data_maked/splits_v2/train.csv \
    --val_csv /home/lht/codexwork/slimhrnet/data_maked/splits_v2/val.csv \
    --teacher_weights /home/lht/codexwork/blurball-mainyy/blurball_best.pth \
    --student_config inference_slimhrnet_v2 \
    --save_dir /home/lht/codexwork/slimhrnet/data_maked/checkpoints_v2 \
    --video_root /home/lht/daqiu \
    --frame_root /home/lht/codexwork/slimhrnet/data_maked/frames \
    --epochs 80 \
    --batch_size 24 \
    --lr 1e-3 \
    --num_workers 8 \
    --alpha 0.7 \
    --beta 0.3 \
    --sigma 3.0 \
    --dynamic_sigma \
    --ohem_ratio 0.3 \
    --ohem_start_epoch 40 \
    --device cuda \
    --resume /home/lht/codexwork/slimhrnet/data_maked/checkpoints_v2/latest.pth | tee train_distill_v2_resume.log
```

## Day-1 Speed Profiling

```bash
CUDA_VISIBLE_DEVICES=0 \
python tools/profile_inference.py \
    --config-name inference_slimhrnet_v2 \
    --weights /home/lht/codexwork/slimhrnet/data_maked/checkpoints_v2/best.pth \
    --do-profiler
```

## Visual Testing On Real Videos

`tools/test_videos.py` now uses a faster testing pipeline:

- sequential `cap.read()` video decoding through an async prefetch thread
- crop + resize on GPU with `torch` and `torchvision.transforms.functional`
- optional frame-difference based static-frame skipping for nearly unchanged frames
- no intermediate `result_frames/` dump during inference
- `traj.csv` is written immediately after tracking
- `result.mp4` is rendered in a separate post-process pass, so it does not slow the main inference loop

### Example: one fixed video + 5 random videos from `/home/lht/ceshi`

```bash
CUDA_VISIBLE_DEVICES=4 \
PYTHONUNBUFFERED=1 \
OPENCV_FFMPEG_LOGLEVEL=0 \
AV_LOG_FORCE_NOCOLOR=1 \
python tools/test_videos.py \
    --config-name inference_slimhrnet \
    --model-path checkpoints_balanced/best.pth \
    --video-dir /home/lht/ceshi \
    --include-video /home/lht/ceshi/D1_S20251015112330_E20251015112400.mp4 \
    --num-random 5 \
    --gpus 0 \
    --decode-queue-size 8 \
    --static-diff-threshold 3.0 \
    --output-dir video_test_outputs/slimhrnet_check | tee test_videos.log
```

Notes:

- `--num-random 5` means the fixed video plus 5 additional random videos, so up to 6 outputs total.
- If you want exactly 5 total including the fixed video, use `--num-random 4`.
- Add `--skip-render` when you only want speed and `traj.csv`, without generating `result.mp4`.
- Add `--disable-static-skip` if you want to isolate async decode without the static-frame optimization.
- `--decode-queue-size` controls how many decoded frames are prefetched on CPU while the GPU is busy.
- `--static-diff-threshold` is the grayscale mean-difference threshold on a 0-255 scale; lower values make the skip logic more conservative.
- Outputs are stored under the chosen `--output-dir`.
- The script writes `selected_videos.txt` and `summary.csv`.

The generated layout looks like this:

```text
video_test_outputs/slimhrnet_check/
  selected_videos.txt
  summary.csv
  <relative_video_path_without_suffix>/
    result.mp4
    traj.csv
```

### Speed fields in `summary.csv`

- `decode_wait_sec`: time spent by the main loop waiting for the async decoder queue.
- `preprocess_sec`: GPU upload + crop + resize + normalize time accumulated across the video.
- `inference_sec`: end-to-end inference loop time excluding post-render.
- `tracking_sec`: tracker-only portion.
- `render_sec`: optional post-render video generation time.
- `skipped_static_frames`: how many frames reused the previous result instead of running the model.
- `fps_inference`: effective FPS for the main inference loop.
- `fps_wall`: effective FPS including post-render when enabled.

## Speed Benchmarks

### 1. Synthetic end-to-end latency benchmark

This uses a synthetic input image and reports preprocess/model/postprocess latency.

```bash
CUDA_VISIBLE_DEVICES=4 \
python benchmark_speed.py \
    --model slimhrnet \
    --weights checkpoints_balanced/best.pth \
    --device cuda \
    --full_pipeline \
    --input_orig_h 1080 \
    --input_orig_w 1920
```

### 2. Real video pipeline timing benchmark

This measures real preprocessing + model + postprocess + tracking time on videos.

```bash
CUDA_VISIBLE_DEVICES=4 \
python benchmark_video_pipeline.py \
    --config-name inference_slimhrnet \
    --model-path checkpoints_balanced/best.pth \
    --videos-dir /home/lht/ceshi \
    --video-glob "**/*.mp4" \
    --gpus 0 \
    --output-csv benchmark_video_pipeline_slimhrnet.csv
```

## Quick Smoke Test

Before a long run, it can be useful to verify that one batch can load and backpropagate:

```bash
CUDA_VISIBLE_DEVICES=4 \
PYTHONUNBUFFERED=1 \
OPENCV_FFMPEG_LOGLEVEL=0 \
AV_LOG_FORCE_NOCOLOR=1 \
python - <<'PY'
import torch
from torch.utils.data import DataLoader
from tools.train_distill import BallDataset, load_teacher, load_student, teacher_soft_label, extract_heatmap_tensor, distill_loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ds = BallDataset(
    "data/splits_balanced/train.csv",
    augment=False,
    sigma=3.0,
    video_root="/home/lht/daqiu",
    frame_root="/home/lht/daqiu_frames",
)
loader = DataLoader(ds, batch_size=6, shuffle=True, num_workers=2, pin_memory=(device.type == "cuda"))
inp, gt_hm = next(iter(loader))
valid_mask = inp.abs().sum(dim=(1, 2, 3)) > 1.0
inp = inp[valid_mask].to(device)
gt_hm = gt_hm[valid_mask].to(device)
teacher = load_teacher("/home/lht/codexwork/blurball-mainyy/blurball_best.pth", device)
student = load_student(device)
soft_lbl = teacher_soft_label(teacher, inp, sigma=3.0)
pred_raw = extract_heatmap_tensor(student(inp))[:, :3]
loss = distill_loss(pred_raw, soft_lbl, gt_hm, 0.7, 0.3)
loss.backward()
print("SMOKE_TEST_OK")
print("input_shape=", tuple(inp.shape))
print("gt_shape=", tuple(gt_hm.shape))
print("loss=", float(loss))
PY
```

### Isolating the new optimizations

If you want to measure async decode alone without static-frame skipping, keep the queue but disable the skip path:

```bash
CUDA_VISIBLE_DEVICES=4 \
PYTHONUNBUFFERED=1 \
OPENCV_FFMPEG_LOGLEVEL=0 \
AV_LOG_FORCE_NOCOLOR=1 \
python tools/test_videos.py \
    --config-name inference_slimhrnet \
    --model-path checkpoints_balanced/best.pth \
    --video-dir /home/lht/ceshi \
    --include-video /home/lht/ceshi/D1_S20251015112330_E20251015112400.mp4 \
    --num-random 3 \
    --gpus 0 \
    --decode-queue-size 8 \
    --disable-static-skip \
    --output-dir video_test_outputs/slimhrnet_check_async_only | tee test_videos_async_only.log
```

For the combined async-decode + static-skip path, remove `--disable-static-skip` and optionally tune `--static-diff-threshold`.
