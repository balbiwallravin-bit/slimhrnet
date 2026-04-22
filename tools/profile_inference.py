"""
Profile SlimHRNet / BlurBall inference the way the Day-1 speed plan asks for:
- PyTorch profiler top CUDA ops
- FP32 / FP16 / channels_last / torch.compile latency table
"""

import argparse
import sys
import time
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from torch.profiler import ProfilerActivity, profile, record_function


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
CONFIG_DIR = SRC_DIR / "configs"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from models import build_model  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", default="inference_slimhrnet_v2")
    parser.add_argument("--weights", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--runs", type=int, default=500)
    parser.add_argument("--do-profiler", action="store_true")
    return parser.parse_args()


def extract_state_dict(state):
    if isinstance(state, dict):
        for key in ["model", "state_dict", "model_state_dict", "net"]:
            if key in state:
                return state[key]
    return state


def load_runtime_config(config_name, device):
    with initialize_config_dir(config_dir=str(CONFIG_DIR.resolve()), version_base=None):
        cfg = compose(config_name=config_name)
    OmegaConf.set_struct(cfg, False)
    cfg.runner.device = "cuda" if device.type == "cuda" else "cpu"
    cfg.runner.gpus = [0]
    return cfg


def load_model(args, device):
    cfg = load_runtime_config(args.config_name, device)
    model = build_model(cfg)
    if args.weights:
        state = torch.load(args.weights, map_location="cpu")
        model.load_state_dict(extract_state_dict(state), strict=False)
    return model.to(device).eval(), cfg


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark_variant(model, x, warmup, runs):
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
        sync(x.device)

        t0 = time.perf_counter()
        for _ in range(runs):
            _ = model(x)
        sync(x.device)
        elapsed = time.perf_counter() - t0
    latency_ms = elapsed / runs * 1000.0
    return latency_ms, 1000.0 / latency_ms


def maybe_channels_last(model, x):
    model = model.to(memory_format=torch.channels_last)
    x = x.to(memory_format=torch.channels_last)
    return model, x


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = device.type == "cuda"

    model, cfg = load_model(args, device)
    in_ch = int(cfg.model.frames_in) * 3
    input_shape = (1, in_ch, int(cfg.model.inp_height), int(cfg.model.inp_width))
    print(f"torch: {torch.__version__}")
    print(f"cuda: {torch.version.cuda}")
    print(f"cudnn: {torch.backends.cudnn.version()}")
    print(f"config: {args.config_name}")
    print(f"input: {input_shape}")

    variants = []

    x_fp32 = torch.randn(*input_shape, device=device, dtype=torch.float32)
    latency, fps = benchmark_variant(model, x_fp32, args.warmup, args.runs)
    variants.append(("fp32", latency, fps))

    if device.type == "cuda":
        model_fp16 = build_model(cfg).to(device).eval().half()
        if args.weights:
            state = torch.load(args.weights, map_location="cpu")
            model_fp16.load_state_dict(extract_state_dict(state), strict=False)
        x_fp16 = x_fp32.half()
        latency, fps = benchmark_variant(model_fp16, x_fp16, args.warmup, args.runs)
        variants.append(("fp16", latency, fps))

        model_cl, x_cl = maybe_channels_last(model_fp16, x_fp16)
        latency, fps = benchmark_variant(model_cl, x_cl, args.warmup, args.runs)
        variants.append(("fp16+channels_last", latency, fps))

        if hasattr(torch, "compile"):
            model_compiled = torch.compile(model_cl, mode="reduce-overhead")
            latency, fps = benchmark_variant(model_compiled, x_cl, max(args.warmup, 200), args.runs)
            variants.append(("fp16+channels_last+compile", latency, fps))

    print("\nLatency table")
    print(f"{'variant':<30} {'latency_ms':>12} {'fps':>12}")
    for name, latency, fps in variants:
        print(f"{name:<30} {latency:>12.3f} {fps:>12.2f}")

    if args.do_profiler:
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
        ) as prof:
            with record_function("inference"):
                with torch.no_grad():
                    for _ in range(20):
                        _ = model(x_fp32)
                    sync(device)
        print("\nProfiler top ops")
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))


if __name__ == "__main__":
    main()
