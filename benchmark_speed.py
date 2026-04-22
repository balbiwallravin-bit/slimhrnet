import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from detectors.blurball_postprocessor import BlurBallPostprocessor  # noqa: E402
from detectors.gaussian_postprocessor import GaussianPostprocessor  # noqa: E402
from models import build_model  # noqa: E402
from utils.resize_ops import build_affine_from_resize_plan, build_resize_plan  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark BlurBall-family models.")
    parser.add_argument(
        "--model",
        type=str,
        default="slimhrnet",
        choices=["slimhrnet", "blurball"],
        help="Model config to benchmark.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Optional checkpoint path. If omitted, random weights are used.",
    )
    parser.add_argument("--input_orig_h", type=int, default=1080)
    parser.add_argument("--input_orig_w", type=int, default=1920)
    parser.add_argument("--input_h", type=int, default=288)
    parser.add_argument("--input_w", type=int, default=512)
    parser.add_argument("--frames_in", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--fp16", action="store_true", help="Run benchmark in FP16.")
    parser.add_argument(
        "--full_pipeline",
        action="store_true",
        help="Benchmark numpy image preprocessing + model forward + postprocess.",
    )
    return parser.parse_args()


def sync_if_needed(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def extract_state_dict(state):
    if isinstance(state, dict):
        for key in ["model", "state_dict", "model_state_dict"]:
            if key in state:
                return state[key]
    return state


def load_runtime_config(args):
    config_dir = str((ROOT / "src" / "configs").resolve())
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name=f"inference_{args.model}")

    OmegaConf.set_struct(cfg, False)
    cfg.model.frames_in = args.frames_in
    cfg.model.inp_height = args.input_h
    cfg.model.inp_width = args.input_w
    cfg.model.out_height = args.input_h
    cfg.model.out_width = args.input_w
    cfg.runner.device = args.device
    cfg.runner.gpus = [0]
    return cfg


def load_model(args):
    cfg = load_runtime_config(args)
    cfg.model.frames_in = args.frames_in
    model = build_model(cfg)

    if args.weights is not None:
        state = torch.load(args.weights, map_location="cpu")
        state_dict = extract_state_dict(state)
        model.load_state_dict(state_dict, strict=False)
        print(f"[INFO] Loaded weights: {args.weights}")
    else:
        print("[INFO] No weights provided; benchmarking random initialization only.")

    return model, cfg


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_postprocessor(cfg):
    name = cfg.detector.postprocessor.name
    if name == "gaussian":
        return GaussianPostprocessor(cfg)
    if name == "blurball":
        return BlurBallPostprocessor(cfg)
    raise KeyError(f"Unsupported postprocessor for benchmark: {name}")


def to_postprocess_dtype(preds):
    if isinstance(preds, dict):
        return {k: v.float() for k, v in preds.items()}
    return preds.float()


def build_dummy_pipeline_inputs(args, cfg, dtype, device):
    del dtype
    plan = build_resize_plan(
        args.input_orig_w,
        args.input_orig_h,
        dst_w=cfg.model.inp_width,
        dst_h=cfg.model.inp_height,
        mode="stretch",
    )
    affine_single = build_affine_from_resize_plan(plan, dtype=np.float32)
    affine_mats = np.stack(
        [affine_single for _ in range(cfg.model.frames_out)],
        axis=0,
    )
    affine_mats = torch.tensor(affine_mats, dtype=torch.float32, device=device).unsqueeze(0)
    return affine_mats


def build_gpu_preprocessed_input(args, device, dtype):
    raw_np = np.random.randint(
        0,
        255,
        (args.input_orig_h, args.input_orig_w, 3),
        dtype=np.uint8,
    )

    sync_if_needed(device)
    t0 = time.perf_counter()
    raw_cpu = torch.from_numpy(raw_np).permute(2, 0, 1).contiguous()
    if device.type == "cuda":
        raw_cpu = raw_cpu.pin_memory()
    raw_gpu = raw_cpu.to(device, non_blocking=(device.type == "cuda"))
    frame_gpu = TF.resize(
        raw_gpu,
        [args.input_h, args.input_w],
        interpolation=TF.InterpolationMode.BILINEAR,
        antialias=False,
    )
    frame_gpu = frame_gpu.float() / 255.0
    if dtype == torch.float16:
        frame_gpu = frame_gpu.half()
    input_tensor = torch.cat([frame_gpu] * args.frames_in, dim=0).unsqueeze(0)
    sync_if_needed(device)
    t1 = time.perf_counter()

    return input_tensor, (t1 - t0) * 1000.0


def benchmark_forward_only(model, args, device, dtype):
    in_ch = args.frames_in * 3
    dummy = torch.randn(1, in_ch, args.input_h, args.input_w, dtype=dtype, device=device)

    with torch.no_grad():
        for _ in range(args.warmup):
            _ = model(dummy)
        sync_if_needed(device)

        latencies = []
        for _ in range(args.runs):
            sync_if_needed(device)
            t0 = time.perf_counter()
            _ = model(dummy)
            sync_if_needed(device)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)

    return {
        "mode": "forward_only",
        "input_channels": in_ch,
        "latencies_ms": np.array(latencies),
    }


def benchmark_full_pipeline(model, cfg, args, device, dtype):
    postprocessor = build_postprocessor(cfg)
    affine_mats = build_dummy_pipeline_inputs(args, cfg, dtype, device)

    with torch.no_grad():
        for _ in range(args.warmup):
            input_tensor, _ = build_gpu_preprocessed_input(args, device, dtype)
            preds = model(input_tensor)
            _ = postprocessor.run(to_postprocess_dtype(preds), affine_mats)
        sync_if_needed(device)

        preprocess_ms = []
        model_ms = []
        post_ms = []
        total_ms = []
        for _ in range(args.runs):
            input_tensor, preprocess_time_ms = build_gpu_preprocessed_input(
                args, device, dtype
            )
            t1 = time.perf_counter()

            preds = model(input_tensor)
            sync_if_needed(device)
            t2 = time.perf_counter()

            _ = postprocessor.run(to_postprocess_dtype(preds), affine_mats)
            sync_if_needed(device)
            t3 = time.perf_counter()

            preprocess_ms.append(preprocess_time_ms)
            model_ms.append((t2 - t1) * 1000.0)
            post_ms.append((t3 - t2) * 1000.0)
            total_ms.append(preprocess_time_ms + (t3 - t1) * 1000.0)

    return {
        "mode": "full_pipeline",
        "input_channels": args.frames_in * 3,
        "preprocess_ms": np.array(preprocess_ms),
        "model_ms": np.array(model_ms),
        "post_ms": np.array(post_ms),
        "latencies_ms": np.array(total_ms),
    }


def summarize_latency(name, values):
    return (
        f"{name:<14}"
        f"mean {values.mean():6.2f} ms | "
        f"median {np.median(values):6.2f} ms | "
        f"p95 {np.percentile(values, 95):6.2f} ms | "
        f"min {values.min():6.2f} ms"
    )


def benchmark(model, cfg, args):
    use_cuda = args.device == "cuda" and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model = model.to(device).eval()
    try:
        gpu_info = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
    except FileNotFoundError:
        gpu_info = ""
    if gpu_info:
        print(f"[INFO] GPU: {gpu_info}")
    else:
        print("[INFO] GPU: nvidia-smi unavailable or no visible GPU reported")

    if args.fp16 and device.type == "cuda":
        model = model.half()
        dtype = torch.float16
        print("[INFO] Running with FP16")
    else:
        dtype = torch.float32

    if args.full_pipeline:
        stats = benchmark_full_pipeline(model, cfg, args, device, dtype)
    else:
        stats = benchmark_forward_only(model, args, device, dtype)

    latencies = stats["latencies_ms"]
    print("\n" + "=" * 50)
    print(f"Model:          {args.model}")
    print(f"Mode:           {stats['mode']}")
    print(f"Device:         {device} | FP16: {args.fp16 and device.type == 'cuda'}")
    print(f"Input:          {stats['input_channels']}ch x {args.input_h} x {args.input_w}")
    print(f"Trainable params: {count_params(model) / 1e6:.2f} M")
    if args.full_pipeline:
        print(summarize_latency("Preprocess", stats["preprocess_ms"]))
        print(summarize_latency("Model", stats["model_ms"]))
        print(summarize_latency("Postprocess", stats["post_ms"]))
        print(summarize_latency("Total", latencies))
        print(f"Pipeline FPS:   {1000.0 / latencies.mean():.1f}")
    else:
        print(summarize_latency("Forward", latencies))
        print(f"FPS:            {1000.0 / latencies.mean():.1f}")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    args = parse_args()
    model, cfg = load_model(args)
    benchmark(model, cfg, args)
