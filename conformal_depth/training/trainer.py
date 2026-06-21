"""
Training loop for ConformalDepth.

Trains the DV-LoRA adapter + quantile head + scale parameters while keeping
the ViT encoder backbone and DPT neck frozen.

Supports:
  - Mixed-precision (bfloat16) via torch.amp
  - Gradient checkpointing
  - Logging to TensorBoard and W&B
  - Checkpoint saving/resumption
  - Periodic conformal calibration during training to monitor q̂
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ..models import ConformalDepthModel
from .losses import TotalLoss
from ..conformal.scores import QuantileResidualScore, collect_model_outputs, compute_calibration_scores
from ..conformal.calibration import run_calibration


class ConformalDepthTrainer:
    """
    Encapsulates the full training loop.

    Args:
        model:          ConformalDepthModel instance.
        train_loader:   DataLoader for training set.
        val_loader:     DataLoader for validation set.
        calib_loader:   DataLoader for conformal calibration (held-out set).
        output_dir:     Directory for checkpoints and logs.
        lr:             Learning rate for LoRA + quantile head parameters.
        weight_decay:   AdamW weight decay.
        alpha:          Conformal miscoverage level.
        use_amp:        Enable bfloat16 mixed precision.
        log_wandb:      Log to Weights & Biases.
        calib_every_n:  Run conformal calibration every N epochs.
    """

    def __init__(
        self,
        model: ConformalDepthModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        calib_loader: DataLoader,
        output_dir: str = "outputs/",
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        alpha: float = 0.10,
        use_amp: bool = True,
        log_wandb: bool = False,
        calib_every_n: int = 5,
        max_grad_norm: float = 1.0,
    ):
        self.model        = model
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.calib_loader = calib_loader
        self.output_dir   = Path(output_dir)
        self.alpha        = alpha
        self.use_amp      = use_amp
        self.log_wandb    = log_wandb
        self.calib_every  = calib_every_n
        self.max_grad_norm = max_grad_norm

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Only optimize LoRA + quantile head + scale params
        trainable = model.trainable_parameters()
        self.optimizer = torch.optim.AdamW(
            trainable, lr=lr, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=lr,
            steps_per_epoch=len(train_loader),
            epochs=100,               # will be overridden in train()
            pct_start=0.05,
        )
        self.criterion = TotalLoss(alpha=alpha)
        self.scaler    = GradScaler(enabled=use_amp)

        self.writer    = SummaryWriter(log_dir=str(self.output_dir / "tb_logs"))
        self.global_step = 0
        self.best_val_loss = float("inf")

        if log_wandb:
            try:
                import wandb
                self.wandb = wandb
            except ImportError:
                self.wandb = None
        else:
            self.wandb = None

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def train(
        self,
        epochs: int = 30,
        device: str | torch.device = "cuda",
        resume_from: str | None = None,
    ):
        device = torch.device(device)
        self.model.to(device)

        # Rebuild scheduler with correct epoch count
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.optimizer.param_groups[0]["lr"],
            steps_per_epoch=len(self.train_loader),
            epochs=epochs,
            pct_start=0.05,
        )

        start_epoch = 0
        if resume_from:
            start_epoch = self._load_checkpoint(resume_from, device)

        for epoch in range(start_epoch, epochs):
            train_losses = self._train_epoch(epoch, device)
            val_losses   = self._val_epoch(epoch, device)

            self._log_epoch(epoch, train_losses, val_losses)

            # Save best checkpoint
            if val_losses["total"] < self.best_val_loss:
                self.best_val_loss = val_losses["total"]
                self._save_checkpoint(epoch, tag="best")

            # Periodic conformal calibration
            if (epoch + 1) % self.calib_every == 0:
                q_hat = self._run_conformal_calibration(device)
                self.writer.add_scalar("conformal/q_hat", q_hat, epoch)

            self._save_checkpoint(epoch, tag="latest")

        self.writer.close()

    # ------------------------------------------------------------------
    # Epoch loops
    # ------------------------------------------------------------------

    def _train_epoch(self, epoch: int, device: torch.device) -> dict[str, float]:
        self.model.train()
        running = {}

        pbar = tqdm(self.train_loader, desc=f"Train [{epoch}]", leave=False)
        for batch in pbar:
            imgs  = batch["image"].to(device)
            depth = batch["depth"].to(device)
            mask  = batch["depth_mask"].to(device)

            self.optimizer.zero_grad()
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                 enabled=self.use_amp):
                pred   = self.model(pixel_values=imgs)
                losses = self.criterion(pred, depth, mask, imgs)

            self.scaler.scale(losses["total"]).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.max_grad_norm
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            for k, v in losses.items():
                running[k] = running.get(k, 0.0) + v.item()

            self.global_step += 1
            pbar.set_postfix({"loss": f"{losses['total'].item():.4f}"})

        n = len(self.train_loader)
        return {k: v / n for k, v in running.items()}

    @torch.no_grad()
    def _val_epoch(self, epoch: int, device: torch.device) -> dict[str, float]:
        self.model.eval()
        running = {}

        for batch in tqdm(self.val_loader, desc=f"Val   [{epoch}]", leave=False):
            imgs  = batch["image"].to(device)
            depth = batch["depth"].to(device)
            mask  = batch["depth_mask"].to(device)

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                 enabled=self.use_amp):
                pred   = self.model(pixel_values=imgs)
                losses = self.criterion(pred, depth, mask, imgs)

            for k, v in losses.items():
                running[k] = running.get(k, 0.0) + v.item()

        n = len(self.val_loader)
        return {k: v / n for k, v in running.items()}

    # ------------------------------------------------------------------
    # Conformal calibration during training
    # ------------------------------------------------------------------

    def _run_conformal_calibration(self, device: torch.device) -> float:
        outputs = collect_model_outputs(self.model, self.calib_loader, device)
        score_fn = QuantileResidualScore(pixel_quantile=0.9)
        scores   = compute_calibration_scores(outputs, score_fn)
        result   = run_calibration(scores, alpha=self.alpha)
        print(f"\n  [Conformal] n_calib={result.n_calib}, "
              f"q̂={result.q_hat:.4f} mm, λ*={result.lambda_star:.4f} mm")
        return result.q_hat

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log_epoch(
        self,
        epoch: int,
        train: dict[str, float],
        val:   dict[str, float],
    ):
        for k, v in train.items():
            self.writer.add_scalar(f"train/{k}", v, epoch)
        for k, v in val.items():
            self.writer.add_scalar(f"val/{k}", v, epoch)

        if self.wandb:
            self.wandb.log({
                **{f"train/{k}": v for k, v in train.items()},
                **{f"val/{k}":   v for k, v in val.items()},
                "epoch": epoch,
            })

        lr = self.optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:3d} | "
            f"train_loss={train['total']:.4f} | "
            f"val_loss={val['total']:.4f} | "
            f"lr={lr:.2e}"
        )

    # ------------------------------------------------------------------
    # Checkpoint I/O
    # ------------------------------------------------------------------

    def _save_checkpoint(self, epoch: int, tag: str = "latest"):
        path = self.output_dir / f"checkpoint_{tag}.pt"
        torch.save(
            {
                "epoch":           epoch,
                "model_state":     self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "scheduler_state": self.scheduler.state_dict(),
                "scaler_state":    self.scaler.state_dict(),
                "best_val_loss":   self.best_val_loss,
            },
            path,
        )

    def _load_checkpoint(self, path: str, device: torch.device) -> int:
        ckpt = torch.load(path, map_location=device, weights_only=True)
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.scheduler.load_state_dict(ckpt["scheduler_state"])
        self.scaler.load_state_dict(ckpt["scaler_state"])
        self.best_val_loss = ckpt.get("best_val_loss", float("inf"))
        epoch = ckpt.get("epoch", 0) + 1
        print(f"Resumed from checkpoint: {path} (epoch {epoch})")
        return epoch
