"""
Distillation training for SlimHRNet using a BlurBall teacher.

Example:
  python tools/train_distill.py \
      --train_csv data/splits/train.csv \
      --val_csv data/splits/val.csv \
      --teacher_weights /home/lht/codexwork/blurball-mainyy/blurball_best.pth \
      --save_dir checkpoints \
      --epochs 80 \
      --device cuda
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "0")
os.environ.setdefault("AV_LOG_FORCE_NOCOLOR", "1")

import torch
import torch.nn.functional as F
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
CONFIG_DIR = SRC_DIR / "configs"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from datasets.ball_dataset import BallDataset  # noqa: E402
from models import build_model  # noqa: E402


MODEL_W, MODEL_H = 512, 288
FRAMES_IN = 3


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", default="data/splits/train.csv")
    parser.add_argument("--val_csv", default="data/splits/val.csv")
    parser.add_argument(
        "--teacher_weights",
        default="/home/lht/codexwork/blurball-mainyy/blurball_best.pth",
    )
    parser.add_argument("--save_dir", default="checkpoints")
    parser.add_argument("--video_root", default="/home/lht/daqiu")
    parser.add_argument("--frame_root", default=None)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=0.7, help="Teacher soft-target weight.")
    parser.add_argument("--beta", type=float, default=0.3, help="Pseudo-GT target weight.")
    parser.add_argument("--sigma", type=float, default=3.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", default=None, help="Resume from a checkpoint.")
    return parser.parse_args()


def extract_state_dict(state):
    if isinstance(state, dict):
        for key in ["model", "state_dict", "model_state_dict", "net"]:
            if key in state:
                return state[key]
    return state


def extract_heatmap_tensor(preds):
    if isinstance(preds, dict):
        if 0 not in preds:
            raise KeyError("Model output does not contain scale=0")
        return preds[0]
    return preds


def unwrap_model(model):
    if isinstance(model, torch.nn.DataParallel):
        return model.module
    return model


def get_visible_gpu_ids(device):
    if device.type != "cuda" or not torch.cuda.is_available():
        return []
    return list(range(torch.cuda.device_count()))


def maybe_wrap_data_parallel(model, gpu_ids, model_name):
    if len(gpu_ids) > 1:
        print(f"[INFO] Using DataParallel for {model_name} on GPUs: {gpu_ids}")
        return torch.nn.DataParallel(model, device_ids=gpu_ids, output_device=gpu_ids[0])
    if len(gpu_ids) == 1:
        print(f"[INFO] Using a single GPU for {model_name}")
    return model


def load_runtime_config(config_name, device):
    with initialize_config_dir(config_dir=str(CONFIG_DIR.resolve()), version_base=None):
        cfg = compose(config_name=config_name)

    OmegaConf.set_struct(cfg, False)
    cfg.model.frames_in = FRAMES_IN
    cfg.model.frames_out = FRAMES_IN
    cfg.model.inp_width = MODEL_W
    cfg.model.inp_height = MODEL_H
    cfg.model.out_width = MODEL_W
    cfg.model.out_height = MODEL_H
    cfg.runner.device = "cuda" if device.type == "cuda" else "cpu"
    cfg.runner.gpus = [0]
    return cfg


def load_teacher(weights_path, device):
    cfg = load_runtime_config("inference_blurball", device)
    teacher = build_model(cfg)

    state = torch.load(weights_path, map_location="cpu")
    teacher.load_state_dict(extract_state_dict(state), strict=False)
    teacher = teacher.to(device).eval()
    for param in teacher.parameters():
        param.requires_grad_(False)
    print("[INFO] Loaded and froze the BlurBall teacher")
    return teacher


def load_student(device):
    cfg = load_runtime_config("inference_slimhrnet", device)
    student = build_model(cfg)
    student = student.to(device)
    total_params = sum(param.numel() for param in student.parameters())
    print(f"[INFO] Loaded SlimHRNet student with {total_params / 1e6:.2f}M params")
    return student


def teacher_soft_label(teacher, inp, sigma=3.0):
    with torch.no_grad():
        hm_teacher = extract_heatmap_tensor(teacher(inp))[:, :FRAMES_IN]
        hm_sig = torch.sigmoid(hm_teacher)

    batch_size, channels, height, width = hm_sig.shape
    device = hm_sig.device
    grid_y, grid_x = torch.meshgrid(
        torch.arange(height, dtype=torch.float32, device=device),
        torch.arange(width, dtype=torch.float32, device=device),
        indexing="ij",
    )

    soft_labels = torch.zeros_like(hm_sig)
    for batch_index in range(batch_size):
        for channel_index in range(channels):
            hm = hm_sig[batch_index, channel_index]
            if hm.max().item() < 0.3:
                continue

            weights = hm * (hm >= 0.3)
            if weights.sum().item() <= 0:
                weights = hm

            total = weights.sum() + 1e-6
            cx = (weights * grid_x).sum() / total
            cy = (weights * grid_y).sum() / total
            dist2 = (grid_x - cx) ** 2 + (grid_y - cy) ** 2
            soft_labels[batch_index, channel_index] = torch.exp(-dist2 / (2.0 * sigma ** 2))

    return soft_labels


def distill_loss(pred_raw, soft_label, gt_label, alpha, beta):
    pred = torch.sigmoid(pred_raw)
    loss_soft = F.mse_loss(pred, soft_label)
    loss_gt = F.mse_loss(pred, gt_label)
    return alpha * loss_soft + beta * loss_gt


def filter_valid_batch(inp, gt_hm):
    valid_mask = inp.abs().sum(dim=(1, 2, 3)) > 1.0
    if valid_mask.sum().item() == 0:
        return None, None, 0
    return inp[valid_mask], gt_hm[valid_mask], int(valid_mask.sum().item())


def evaluate(student, val_loader, device):
    student.eval()
    total_loss = 0.0
    total_err = 0.0
    count = 0
    valid_batches = 0

    with torch.no_grad():
        for inp, gt_hm in val_loader:
            inp = inp.to(device, non_blocking=(device.type == "cuda"))
            gt_hm = gt_hm.to(device, non_blocking=(device.type == "cuda"))
            inp, gt_hm, kept = filter_valid_batch(inp, gt_hm)
            if kept == 0:
                continue

            pred_raw = extract_heatmap_tensor(student(inp))[:, :FRAMES_IN]
            pred_hm = torch.sigmoid(pred_raw)

            loss = F.mse_loss(pred_hm, gt_hm)
            total_loss += loss.item()
            valid_batches += 1

            _, _, height, width = pred_hm.shape
            grid_y, grid_x = torch.meshgrid(
                torch.arange(height, dtype=torch.float32, device=device),
                torch.arange(width, dtype=torch.float32, device=device),
                indexing="ij",
            )
            for batch_index in range(pred_hm.shape[0]):
                pred_current = pred_hm[batch_index, -1]
                gt_current = gt_hm[batch_index, -1]
                if pred_current.max().item() < 0.3:
                    continue
                if gt_current.sum().item() <= 0:
                    continue

                pred_weights = pred_current * (pred_current >= 0.3)
                if pred_weights.sum().item() <= 0:
                    pred_weights = pred_current

                pred_total = pred_weights.sum() + 1e-6
                gt_total = gt_current.sum() + 1e-6
                pred_x = (pred_weights * grid_x).sum() / pred_total
                pred_y = (pred_weights * grid_y).sum() / pred_total
                gt_x = (gt_current * grid_x).sum() / gt_total
                gt_y = (gt_current * grid_y).sum() / gt_total
                err = ((pred_x - gt_x) ** 2 + (pred_y - gt_y) ** 2).sqrt().item()
                total_err += err
                count += 1

    mean_loss = total_loss / max(valid_batches, 1)
    mean_err = total_err / max(count, 1)
    return mean_loss, mean_err


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    gpu_ids = get_visible_gpu_ids(device)
    if len(gpu_ids) > 1 and args.batch_size < len(gpu_ids):
        print(
            f"[WARN] batch_size={args.batch_size} is smaller than visible_gpus={len(gpu_ids)}; "
            "some GPUs may stay idle"
        )

    train_ds = BallDataset(
        args.train_csv,
        augment=True,
        sigma=args.sigma,
        video_root=args.video_root,
        frame_root=args.frame_root,
    )
    val_ds = BallDataset(
        args.val_csv,
        augment=False,
        sigma=args.sigma,
        video_root=args.video_root,
        frame_root=args.frame_root,
    )

    if len(train_ds) == 0:
        raise ValueError(f"Training dataset is empty: {args.train_csv}")
    if len(val_ds) == 0:
        raise ValueError(f"Validation dataset is empty: {args.val_csv}")

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        drop_last=len(train_ds) >= args.batch_size,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=args.num_workers > 0,
    )

    teacher = load_teacher(args.teacher_weights, device)
    student = load_student(device)

    start_epoch = 0
    best_val_loss = float("inf")
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location="cpu")
        student.load_state_dict(extract_state_dict(ckpt), strict=False)
        start_epoch = ckpt.get("epoch", -1) + 1
        best_val_loss = ckpt.get("best_val_loss", best_val_loss)
        print(f"[INFO] Resumed model weights from epoch {start_epoch}")

    teacher = maybe_wrap_data_parallel(teacher, gpu_ids, "teacher")
    student = maybe_wrap_data_parallel(student, gpu_ids, "student")

    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.01,
    )

    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location="cpu")
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])

    log_path = os.path.join(args.save_dir, "train_log.csv")
    log_fields = ["epoch", "train_loss", "val_loss", "val_err_hm_px", "lr", "time_min"]
    if start_epoch == 0:
        with open(log_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=log_fields).writeheader()

    for epoch in range(start_epoch, args.epochs):
        student.train()
        t0 = time.time()
        total_train_loss = 0.0
        valid_train_batches = 0
        skipped_train_batches = 0

        for batch_index, (inp, gt_hm) in enumerate(train_loader):
            inp = inp.to(device, non_blocking=pin_memory)
            gt_hm = gt_hm.to(device, non_blocking=pin_memory)
            inp, gt_hm, kept = filter_valid_batch(inp, gt_hm)
            if kept == 0:
                skipped_train_batches += 1
                continue

            soft_lbl = teacher_soft_label(teacher, inp, sigma=args.sigma)
            pred_raw = extract_heatmap_tensor(student(inp))[:, :FRAMES_IN]
            loss = distill_loss(pred_raw, soft_lbl, gt_hm, args.alpha, args.beta)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()

            total_train_loss += loss.item()
            valid_train_batches += 1

            if (batch_index + 1) % 50 == 0:
                avg_loss = total_train_loss / max(valid_train_batches, 1)
                print(
                    f"  Epoch {epoch + 1} [{batch_index + 1}/{len(train_loader)}] "
                    f"loss={avg_loss:.5f} | skipped={skipped_train_batches}"
                )

        scheduler.step()

        val_loss, val_err = evaluate(student, val_loader, device)
        train_loss = total_train_loss / max(valid_train_batches, 1)
        elapsed_min = (time.time() - t0) / 60.0

        print(
            f"Epoch {epoch + 1:3d}/{args.epochs} | "
            f"train_loss={train_loss:.5f} | "
            f"val_loss={val_loss:.5f} | "
            f"val_err={val_err:.2f}px | "
            f"lr={scheduler.get_last_lr()[0]:.2e} | "
            f"skipped={skipped_train_batches} | "
            f"{elapsed_min:.1f}min"
        )

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss

        state_dict = unwrap_model(student).state_dict()
        ckpt = {
            "epoch": epoch,
            "model": state_dict,
            "model_state_dict": state_dict,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_val_loss": best_val_loss,
        }
        torch.save(ckpt, os.path.join(args.save_dir, "latest.pth"))

        if improved:
            torch.save(ckpt, os.path.join(args.save_dir, "best.pth"))
            print(f"  [*] Saved new best checkpoint (val_loss={best_val_loss:.5f})")

        with open(log_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=log_fields).writerow(
                {
                    "epoch": epoch + 1,
                    "train_loss": round(train_loss, 6),
                    "val_loss": round(val_loss, 6),
                    "val_err_hm_px": round(val_err, 3),
                    "lr": round(scheduler.get_last_lr()[0], 8),
                    "time_min": round(elapsed_min, 2),
                }
            )

    print(f"\n[Done] Training finished. Best checkpoint: {args.save_dir}/best.pth")


if __name__ == "__main__":
    main()
