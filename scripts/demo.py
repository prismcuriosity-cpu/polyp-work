#!/usr/bin/env python3
"""
Interactive Demo — run ConformalDepth on a single colonoscopy frame.

Given a JPEG/PNG image, camera intrinsics, a polyp mask (optional), and a
calibration JSON, produces:
  - Depth interval maps (lo, median, hi)
  - Conformal coverage map
  - Guaranteed polyp diameter interval in mm

Usage:
    python scripts/demo.py \
        --image       frame.jpg \
        --checkpoint  outputs/run_01/checkpoint_best.pt \
        --calib_json  outputs/run_01/conformal_calibration.json \
        --intrinsics  "896,896,512,384" \
        [--mask       polyp_mask.png] \
        [--output_dir demo_out/]
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

from conformal_depth.models import ConformalDepthModel
from conformal_depth.data.transforms import build_transforms, apply_transforms
from conformal_depth.conformal.size_interval import polyp_diameter_interval, classify_size
from conformal_depth.evaluation.visualize import plot_depth_panel


def load_image(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Returns (rgb_uint8, normalized_tensor) for the model."""
    img_bgr = cv2.imread(path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb


def parse_intrinsics(s: str) -> np.ndarray:
    """Parse 'fx,fy,cx,cy' string → float32 array."""
    return np.array([float(x) for x in s.split(",")], dtype=np.float32)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image",       required=True)
    p.add_argument("--checkpoint",  required=True)
    p.add_argument("--calib_json",  required=True)
    p.add_argument("--intrinsics",  default="896,896,512,384",
                   help="fx,fy,cx,cy in pixels")
    p.add_argument("--mask",        default=None, help="Binary polyp mask PNG")
    p.add_argument("--output_dir",  default="demo_out")
    p.add_argument("--model",       default="depth-anything/Depth-Anything-V2-Large-hf")
    p.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    # Load calibration
    with open(args.calib_json) as f:
        calib = json.load(f)
    q_hat = calib["q_adjusted"]
    print(f"[demo] Conformal threshold q̂ = {q_hat:.4f} mm")

    # Load model
    print(f"[demo] Loading model…")
    model = ConformalDepthModel.from_pretrained(model_name=args.model)
    ckpt  = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()

    # Load image
    img_rgb = load_image(args.image)
    H, W    = img_rgb.shape[:2]
    print(f"[demo] Image: {W}×{H}")

    # Transform
    transform = build_transforms(split="test")
    dummy_depth = np.zeros((H, W), dtype=np.float32)
    result = apply_transforms(transform, img_rgb, dummy_depth)
    img_tensor = result["image"].unsqueeze(0).to(device)    # (1, 3, H', W')

    # Inference
    with torch.no_grad():
        raw = model(pixel_values=img_tensor, output_size=(H, W))  # (1, 3, H, W)

    depth_lo  = raw[0, 0].cpu().numpy()   # (H, W)
    depth_med = raw[0, 1].cpu().numpy()
    depth_hi  = raw[0, 2].cpu().numpy()

    # Conformal padding
    depth_lo_q = depth_lo - q_hat
    depth_hi_q = depth_hi + q_hat

    print(f"[demo] Depth range: [{depth_med.min():.1f}, {depth_med.max():.1f}] mm")
    print(f"[demo] Mean interval width: {(depth_hi_q - depth_lo_q).mean():.2f} mm")

    # Polyp mask
    polyp_mask = None
    if args.mask:
        msk_raw = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)
        polyp_mask = (msk_raw > 127).astype(bool)
        print(f"[demo] Polyp mask: {polyp_mask.sum()} px ({polyp_mask.mean()*100:.1f}% of frame)")

        # Compute diameter interval
        intrinsics = parse_intrinsics(args.intrinsics)
        D_lo, D_hi = polyp_diameter_interval(
            mask=polyp_mask,
            depth_lo=depth_lo_q,
            depth_hi=depth_hi_q,
            intrinsics=intrinsics,
            q_hat=0.0,          # already padded above
        )
        D_mid = (D_lo + D_hi) / 2
        size_class = classify_size(D_mid)

        print(f"\n{'='*50}")
        print(f"  Polyp diameter interval: [{D_lo:.1f}, {D_hi:.1f}] mm")
        print(f"  Point estimate (midpoint): {D_mid:.1f} mm")
        print(f"  Paris classification: {size_class.upper()}")
        print(f"{'='*50}\n")

        # Save sizing result
        sizing = {
            "D_lo_mm": D_lo, "D_hi_mm": D_hi, "D_mid_mm": D_mid,
            "size_class": size_class, "q_hat_mm": q_hat,
        }
        with open(os.path.join(args.output_dir, "sizing_result.json"), "w") as f:
            json.dump(sizing, f, indent=2)

    # Visualization
    dummy_gt = depth_med.copy()   # use median as pseudo-GT for display
    fig = plot_depth_panel(
        image=img_rgb,
        gt_depth=dummy_gt,
        pred_med=depth_med,
        pred_lo=depth_lo_q,
        pred_hi=depth_hi_q,
        polyp_mask=polyp_mask,
        save_path=os.path.join(args.output_dir, "depth_panel.png"),
        q_hat=0.0,    # already padded
    )
    plt.close(fig)

    print(f"[demo] Results saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
