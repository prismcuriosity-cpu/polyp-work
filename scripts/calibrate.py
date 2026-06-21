#!/usr/bin/env python3
"""
Conformal Calibration Script.

Loads a trained ConformalDepthModel, runs it on the C3VD calibration split,
computes per-frame nonconformity scores, and saves the calibration result.

Optionally estimates the video-exchangeability TV slack and outputs a
shift-adjusted threshold.

Usage:
    python scripts/calibrate.py \
        --checkpoint outputs/run_01/checkpoint_best.pt \
        --c3vd_root /data/c3vd \
        --output_dir outputs/run_01 \
        --alpha 0.10 \
        --score_type quantile \
        [--estimate_shift]
"""

import argparse
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from conformal_depth.models import ConformalDepthModel
from conformal_depth.data import build_c3vd_dataloaders
from conformal_depth.conformal.scores import (
    QuantileResidualScore,
    ScaleAlignedScore,
    collect_model_outputs,
    compute_calibration_scores,
)
from conformal_depth.conformal.calibration import run_calibration
from conformal_depth.conformal.coverage import (
    extract_cls_tokens,
    estimate_tv_via_classifier,
    ShiftAwareCoverageAdjuster,
)


def parse_args():
    p = argparse.ArgumentParser(description="Conformal calibration for ConformalDepth")
    p.add_argument("--checkpoint",      required=True)
    p.add_argument("--c3vd_root",       required=True)
    p.add_argument("--output_dir",      required=True)
    p.add_argument("--alpha",           type=float, default=0.10)
    p.add_argument("--delta",           type=float, default=0.05,
                   help="RCPS confidence level")
    p.add_argument("--score_type",      default="quantile",
                   choices=["quantile", "scale_aligned"],
                   help="Nonconformity score family")
    p.add_argument("--pixel_quantile",  type=float, default=0.90,
                   help="Within-frame aggregation quantile")
    p.add_argument("--estimate_shift",  action="store_true",
                   help="Estimate TV slack between calib and test distributions")
    p.add_argument("--batch_size",      type=int, default=4)
    p.add_argument("--num_workers",     type=int, default=4)
    p.add_argument("--model",           default="depth-anything/Depth-Anything-V2-Large-hf")
    p.add_argument("--device",          default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    print(f"[calibrate] Loading model from {args.checkpoint}")
    model = ConformalDepthModel.from_pretrained(model_name=args.model)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()

    # ------------------------------------------------------------------
    # Calibration data
    # ------------------------------------------------------------------
    print(f"[calibrate] Loading C3VD calibration split from {args.c3vd_root}")
    _, _, calib_loader = build_c3vd_dataloaders(
        root=args.c3vd_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print(f"[calibrate] Calibration set size: {len(calib_loader.dataset)} frames")

    # ------------------------------------------------------------------
    # Collect model outputs
    # ------------------------------------------------------------------
    print("[calibrate] Running model on calibration set…")
    outputs = collect_model_outputs(model, calib_loader, device)

    # ------------------------------------------------------------------
    # Compute nonconformity scores
    # ------------------------------------------------------------------
    if args.score_type == "quantile":
        score_fn = QuantileResidualScore(pixel_quantile=args.pixel_quantile)
    else:
        score_fn = ScaleAlignedScore(pixel_quantile=args.pixel_quantile)

    print(f"[calibrate] Computing {args.score_type} nonconformity scores…")
    scores = compute_calibration_scores(outputs, score_fn)
    print(f"[calibrate] Score stats: mean={scores.mean():.4f}, "
          f"std={scores.std():.4f}, max={scores.max():.4f}")

    # ------------------------------------------------------------------
    # Conformal calibration
    # ------------------------------------------------------------------
    result = run_calibration(scores, alpha=args.alpha, delta=args.delta, use_rcps=True)
    print(f"[calibrate] {result}")

    # ------------------------------------------------------------------
    # Optional: shift-aware adjustment
    # ------------------------------------------------------------------
    q_adjusted = result.threshold
    eps_tv = None

    if args.estimate_shift:
        print("[calibrate] Estimating distribution shift (calib vs test)…")
        _, val_loader, test_loader = build_c3vd_dataloaders(
            root=args.c3vd_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        calib_tokens = extract_cls_tokens(model, calib_loader, device)
        test_tokens  = extract_cls_tokens(model, test_loader,  device)

        eps_tv = estimate_tv_via_classifier(calib_tokens, test_tokens)
        print(f"[calibrate] TV slack ε_TV = {eps_tv:.4f}")

        adjuster = ShiftAwareCoverageAdjuster(method="tv")
        q_adjusted = adjuster.adjust(result.threshold, eps_tv, scores)
        print(f"[calibrate] Shift-adjusted threshold: {q_adjusted:.4f} mm")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    calib_out = {
        "alpha":        args.alpha,
        "delta":        args.delta,
        "score_type":   args.score_type,
        "n_calib":      result.n_calib,
        "q_hat":        result.q_hat,
        "lambda_star":  result.lambda_star,
        "threshold":    result.threshold,
        "q_adjusted":   q_adjusted,
        "eps_tv":       eps_tv,
        "score_mean":   float(scores.mean()),
        "score_std":    float(scores.std()),
        "score_max":    float(scores.max()),
    }

    out_path = os.path.join(args.output_dir, "conformal_calibration.json")
    with open(out_path, "w") as f:
        json.dump(calib_out, f, indent=2)
    print(f"[calibrate] Saved calibration result to {out_path}")

    np.save(
        os.path.join(args.output_dir, "calibration_scores.npy"),
        scores,
    )
    print("[calibrate] Done.")


if __name__ == "__main__":
    main()
