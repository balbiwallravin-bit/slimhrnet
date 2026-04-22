"""
Distillation training for SlimHRNet using a BlurBall teacher.

This version upgrades the original training loop with:
- direct teacher heatmap KD
- focal heatmap loss
- optional OHEM
- optional angle/length auxiliary supervision
- dynamic sigma and interpolated-label weights through BallDataset
"""

import argparse
import csv
import math
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
    parser.add_argument(
        "--train_csv",
        default="/home/lht/codexwork/slimhrnet/data_maked/splits_v2/train.csv",
    )
    parser.add_argument(
        "--val_csv",
        default="/home/lht/codexwork/slimhrnet/data_maked/splits_v2/val.csv",
    )
    parser.add_argument(
        "--teacher_weights",
        default="/home/lht/codexwork/blurball-mainyy/blurball_best.pth",
    )
    parser.add_argument("--teacher_config", default="inference_blurball")
    parser.add_argument("--student_config", default="inference_slimhrnet_v2")
    parser.add_argument(
        "--save_dir",
        default="/home/lht/codexwork/slimhrnet/data_maked/checkpoints_v2",
    )
    parser.add_argument("--video_root", default="/home/lht/daqiu")
    parser.add_argument(
        "--frame_root",
        default="/home/lht/codexwork/slimhrnet/data_maked/frames",
    )
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=0.7, help="Teacher KD weight.")
    parser.add_argument("--beta", type=float, default=0.3, help="Pseudo-GT heatmap weight.")
    parser.add_argument("--sigma", type=float, default=3.0)
    parser.add_argument("--dynamic_sigma", action="store_true")
    parser.add_argument("--min_sigma", type=float, default=2.0)
    parser.add_argument("--max_sigma", type=float, default=5.0)
    parser.add_argument("--length_norm_max", type=float, default=64.0)
    parser.add_argument("--heatmap_loss", default="focal", choices=["focal", "mse"])
    parser.add_argument("--angle_weight", type=float, default=0.3)
    parser.add_argument("--length_weight", type=float, default=0.2)
    parser.add_argument("--disable_aux", action="store_true")
    parser.add_argument("--ohem_ratio", type=float, default=0.3)
    parser.add_argument("--ohem_start_epoch", type=int, default=40)
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


def extract_aux_tensor(preds, key):
    if isinstance(preds, dict):
        return preds.get(key)
    return None


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


def load_teacher(weights_path, device, config_name):
    cfg = load_runtime_config(config_name, device)
    teacher = build_model(cfg)

    state = torch.load(weights_path, map_location="cpu")
    teacher.load_state_dict(extract_state_dict(state), strict=False)
    teacher = teacher.to(device).eval()
    for param in teacher.parameters():
        param.requires_grad_(False)
    print("[INFO] Loaded and froze the BlurBall teacher")
    return teacher


def load_student(device, config_name):
    cfg = load_runtime_config(config_name, device)
    student = build_model(cfg)
    student = student.to(device)
    total_params = sum(param.numel() for param in student.parameters())
    print(f"[INFO] Loaded student with {total_params / 1e6:.2f}M params from {config_name}")
    return student


def teacher_soft_label(teacher, inp):
    with torch.no_grad():
        hm_teacher = extract_heatmap_tensor(teacher(inp))[:, :FRAMES_IN]
    return torch.sigmoid(hm_teacher)


def focal_heatmap_loss_per_sample(pred_raw, target, alpha=2.0, beta=4.0):
    pred = torch.sigmoid(pred_raw).clamp(1e-7, 1.0 - 1e-7)
    pos_mask = (target >= 0.999).float()
    neg_mask = (target < 0.999).float()
    neg_weights = (1.0 - target) ** beta

    pos_loss = -torch.log(pred) * ((1.0 - pred) ** alpha) * pos_mask
    neg_loss = -torch.log(1.0 - pred) * (pred ** alpha) * neg_weights * neg_mask

    pos_loss = pos_loss.flatten(1).sum(dim=1)
    neg_loss = neg_loss.flatten(1).sum(dim=1)
    num_pos = pos_mask.flatten(1).sum(dim=1)
    return torch.where(num_pos > 0, (pos_loss + neg_loss) / num_pos.clamp_min(1.0), neg_loss)


def mse_loss_per_sample(pred_raw, target):
    pred = torch.sigmoid(pred_raw)
    return F.mse_loss(pred, target, reduction="none").flatten(1).mean(dim=1)


def kd_loss_per_sample(pred_raw, teacher_label):
    pred = torch.sigmoid(pred_raw)
    return F.mse_loss(pred, teacher_label, reduction="none").flatten(1).mean(dim=1)


def masked_aux_loss_per_sample(pred_aux, target_aux, mask, aux_valid):
    if pred_aux is None:
        return torch.zeros_like(aux_valid)
    weight = mask.expand_as(pred_aux)
    numer = ((pred_aux - target_aux) ** 2 * weight).flatten(1).sum(dim=1)
    denom = weight.flatten(1).sum(dim=1).clamp_min(1e-6)
    return (numer / denom) * aux_valid


def decode_soft_argmax_xy(heatmap, beta=100.0):
    batch, _, height, width = heatmap.shape
    scores = heatmap.amax(dim=(1, 2, 3))
    weights = torch.softmax(heatmap.view(batch, -1) * beta, dim=1).view(batch, 1, height, width)
    ys = torch.arange(height, device=heatmap.device, dtype=heatmap.dtype).view(1, 1, height, 1)
    xs = torch.arange(width, device=heatmap.device, dtype=heatmap.dtype).view(1, 1, 1, width)
    x = (weights * xs).sum(dim=(1, 2, 3))
    y = (weights * ys).sum(dim=(1, 2, 3))
    return x, y, scores


def select_loss_per_sample(pred_raw, gt_hm, heatmap_loss):
    if heatmap_loss == "focal":
        return focal_heatmap_loss_per_sample(pred_raw, gt_hm)
    return mse_loss_per_sample(pred_raw, gt_hm)


def apply_ohem(loss_per_sample, ratio, enabled):
    if not enabled or ratio >= 1.0:
        return loss_per_sample.mean()
    k = max(1, int(math.ceil(loss_per_sample.shape[0] * ratio)))
    hard_losses, _ = torch.topk(loss_per_sample, k)
    return hard_losses.mean()


def evaluate(student, val_loader, device, args):
    student.eval()
    total_loss = 0.0
    total_err = 0.0
    count = 0
    valid_batches = 0

    with torch.no_grad():
        for inp, target in val_loader:
            inp = inp.to(device, non_blocking=(device.type == "cuda"))
            gt_hm = target["heatmap"].to(device, non_blocking=(device.type == "cuda"))
            angle_t = target["angle"].to(device, non_blocking=(device.type == "cuda"))
            length_t = target["length"].to(device, non_blocking=(device.type == "cuda"))
            mask = target["mask"].to(device, non_blocking=(device.type == "cuda"))
            sample_weight = target["sample_weight"].to(device, non_blocking=(device.type == "cuda"))
            aux_valid = target["aux_valid"].to(device, non_blocking=(device.type == "cuda"))

            preds = student(inp)
            pred_raw = extract_heatmap_tensor(preds)[:, :FRAMES_IN]
            angle_pred = extract_aux_tensor(preds, "angle")
            length_pred = extract_aux_tensor(preds, "length")

            gt_loss = select_loss_per_sample(pred_raw, gt_hm, args.heatmap_loss)
            angle_loss = masked_aux_loss_per_sample(angle_pred, angle_t, mask, aux_valid)
            length_loss = masked_aux_loss_per_sample(length_pred, length_t, mask, aux_valid)
            loss = gt_loss * sample_weight
            if not args.disable_aux:
                loss = loss + args.angle_weight * angle_loss * sample_weight
                loss = loss + args.length_weight * length_loss * sample_weight

            total_loss += loss.mean().item()
            valid_batches += 1

            pred_current = torch.sigmoid(pred_raw[:, -1:, ...])
            gt_current = gt_hm[:, -1:, ...]
            pred_x, pred_y, scores = decode_soft_argmax_xy(pred_current)
            gt_x, gt_y, _ = decode_soft_argmax_xy(gt_current)
            valid = scores >= 0.3
            if valid.any():
                err = ((pred_x[valid] - gt_x[valid]) ** 2 + (pred_y[valid] - gt_y[valid]) ** 2).sqrt()
                total_err += err.sum().item()
                count += int(valid.sum().item())

    mean_loss = total_loss / max(valid_batches, 1)
    mean_err = total_err / max(count, 1)
    return mean_loss, mean_err


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = device.type == "cuda"
    os.makedirs(args.save_dir, exist_ok=True)

    gpu_ids = get_visible_gpu_ids(device)
    if len(gpu_ids) > 1 and args.batch_size < len(gpu_ids):
        print(
            f"[WARN] batch_size={args.batch_size} is smaller than visible_gpus={len(gpu_ids)}; "
            "some GPUs may stay idle"
        )

    dataset_kwargs = {
        "sigma": args.sigma,
        "video_root": args.video_root,
        "frame_root": args.frame_root,
        "dynamic_sigma": args.dynamic_sigma,
        "min_sigma": args.min_sigma,
        "max_sigma": args.max_sigma,
        "length_norm_max": args.length_norm_max,
    }
    train_ds = BallDataset(args.train_csv, augment=True, **dataset_kwargs)
    val_ds = BallDataset(args.val_csv, augment=False, **dataset_kwargs)

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

    teacher = load_teacher(args.teacher_weights, device, args.teacher_config)
    student = load_student(device, args.student_config)

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
    log_fields = [
        "epoch",
        "train_loss",
        "val_loss",
        "val_err_hm_px",
        "lr",
        "time_min",
    ]
    if start_epoch == 0:
        with open(log_path, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=log_fields).writeheader()

    for epoch in range(start_epoch, args.epochs):
        student.train()
        t0 = time.time()
        total_train_loss = 0.0
        valid_train_batches = 0

        for batch_index, (inp, target) in enumerate(train_loader):
            inp = inp.to(device, non_blocking=pin_memory)
            gt_hm = target["heatmap"].to(device, non_blocking=pin_memory)
            angle_t = target["angle"].to(device, non_blocking=pin_memory)
            length_t = target["length"].to(device, non_blocking=pin_memory)
            mask = target["mask"].to(device, non_blocking=pin_memory)
            sample_weight = target["sample_weight"].to(device, non_blocking=pin_memory)
            aux_valid = target["aux_valid"].to(device, non_blocking=pin_memory)

            teacher_lbl = teacher_soft_label(teacher, inp)
            preds = student(inp)
            pred_raw = extract_heatmap_tensor(preds)[:, :FRAMES_IN]
            angle_pred = extract_aux_tensor(preds, "angle")
            length_pred = extract_aux_tensor(preds, "length")

            kd_loss = kd_loss_per_sample(pred_raw, teacher_lbl)
            gt_loss = select_loss_per_sample(pred_raw, gt_hm, args.heatmap_loss) * sample_weight
            total_per_sample = args.alpha * kd_loss + args.beta * gt_loss

            if not args.disable_aux:
                angle_loss = masked_aux_loss_per_sample(angle_pred, angle_t, mask, aux_valid) * sample_weight
                length_loss = masked_aux_loss_per_sample(length_pred, length_t, mask, aux_valid) * sample_weight
                total_per_sample = total_per_sample + args.angle_weight * angle_loss
                total_per_sample = total_per_sample + args.length_weight * length_loss

            use_ohem = args.ohem_ratio < 1.0 and epoch >= args.ohem_start_epoch
            loss = apply_ohem(total_per_sample, args.ohem_ratio, enabled=use_ohem)

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
                    f"loss={avg_loss:.5f} | ohem={'on' if use_ohem else 'off'}"
                )

        scheduler.step()

        val_loss, val_err = evaluate(student, val_loader, device, args)
        train_loss = total_train_loss / max(valid_train_batches, 1)
        elapsed_min = (time.time() - t0) / 60.0

        print(
            f"Epoch {epoch + 1:3d}/{args.epochs} | "
            f"train_loss={train_loss:.5f} | "
            f"val_loss={val_loss:.5f} | "
            f"val_err={val_err:.2f}px | "
            f"lr={scheduler.get_last_lr()[0]:.2e} | "
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
            "student_config": args.student_config,
        }
        torch.save(ckpt, os.path.join(args.save_dir, "latest.pth"))

        if improved:
            torch.save(ckpt, os.path.join(args.save_dir, "best.pth"))
            print(f"  [*] Saved new best checkpoint (val_loss={best_val_loss:.5f})")

        with open(log_path, "a", newline="", encoding="utf-8") as f:
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
