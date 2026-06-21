#!/usr/bin/env python3
"""
ConformalDepth Training Script.

Usage:
    python scripts/train.py \
        --c3vd_root /data/c3vd \
        --output_dir outputs/run_01 \
        --model depth-anything/Depth-Anything-V2-Large-hf \
        --epochs 30 \
        --batch_size 8 \
        --lr 1e-4 \
        --alpha 0.10 \
        --lora_r 16 \
        [--use_v2] [--log_wandb]
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import wandb

from conformal_depth.models import ConformalDepthModel
from conformal_depth.data import build_c3vd_dataloaders
from conformal_depth.training import get_trainer
ConformalDepthTrainer = get_trainer()


def parse_args():
    p = argparse.ArgumentParser(description="Train ConformalDepth")
    p.add_argument("--c3vd_root",   required=True,  help="C3VD dataset root")
    p.add_argument("--output_dir",  default="outputs/run_01")
    p.add_argument("--model",       default="depth-anything/Depth-Anything-V2-Large-hf")
    p.add_argument("--epochs",      type=int,   default=30)
    p.add_argument("--batch_size",  type=int,   default=8)
    p.add_argument("--num_workers", type=int,   default=4)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--alpha",       type=float, default=0.10,
                   help="Target miscoverage level (e.g. 0.10 = 90% coverage)")
    p.add_argument("--lora_r",      type=int,   default=16)
    p.add_argument("--lora_alpha",  type=int,   default=32)
    p.add_argument("--use_v2",      action="store_true", help="Include C3VDv2 sequences")
    p.add_argument("--use_amp",     action="store_true", default=True)
    p.add_argument("--log_wandb",   action="store_true")
    p.add_argument("--resume_from", default=None, help="Path to checkpoint to resume from")
    p.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()

    if args.log_wandb:
        wandb.init(
            project="conformal-depth-endoscopy",
            config=vars(args),
            name=os.path.basename(args.output_dir),
        )

    print(f"[train] Loading model: {args.model}")
    model = ConformalDepthModel.from_pretrained(
        model_name=args.model,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
    )

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total     = sum(p.numel() for p in model.parameters())
    print(f"[train] Trainable params: {n_trainable:,} / {n_total:,} "
          f"({100*n_trainable/n_total:.2f}%)")

    print(f"[train] Building C3VD dataloaders from {args.c3vd_root}")
    train_loader, val_loader, calib_loader = build_c3vd_dataloaders(
        root=args.c3vd_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_v2=args.use_v2,
    )
    print(f"[train] train={len(train_loader.dataset)} | "
          f"val={len(val_loader.dataset)} | "
          f"calib={len(calib_loader.dataset)} frames")

    trainer = ConformalDepthTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        calib_loader=calib_loader,
        output_dir=args.output_dir,
        lr=args.lr,
        alpha=args.alpha,
        use_amp=args.use_amp,
        log_wandb=args.log_wandb,
    )

    trainer.train(
        epochs=args.epochs,
        device=args.device,
        resume_from=args.resume_from,
    )

    print(f"[train] Done. Best val loss: {trainer.best_val_loss:.6f}")
    print(f"[train] Checkpoints saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
