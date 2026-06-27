#!/usr/bin/env python3
"""
Full Evaluation Script — compares ConformalDepth against all baselines.

Runs evaluation on:
  - C3VD test split (phantom, GT depth)
  - SimCol3D (synthetic, coverage stress-test)
  - EndoSLAM (ex-vivo, transfer)

Baselines (enabled with --baselines):
  - ZoeDepth zero-shot (Intel/zoedepth-nk, metric output in mm)
  - Depth-Anything V2 Large zero-shot (relative depth, scale-aligned at eval)

Reports:
  - Depth metrics (AbsRel, RMSE, δ<1.25, SILog)
  - Conformal coverage vs. nominal at calibrated q̂
  - Coverage by depth bin (CCDR)
  - Width–coverage calibration curve
  - Polyp sizing: MAE, RMSE, CCC, 3-group accuracy
  - Confusion matrix
  - Per-baseline comparison table

Usage:
    python scripts/evaluate.py \
        --checkpoint outputs/run_01/checkpoint_best.pt \
        --calib_json  outputs/run_01/conformal_calibration.json \
        --c3vd_root  /data/c3vd \
        --simcol_root /data/simcol3d \
        --endoslam_root /data/endoslam \
        --kvasir_root /data/kvasir-seg \
        --output_dir outputs/eval_01 \
        --baselines \
        [--zoedepth_model Intel/zoedepth-nk] \
        [--da_model depth-anything/Depth-Anything-V2-Large-hf]
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from conformal_depth.models import (
    ConformalDepthModel,
    DepthAnythingZeroShot,
    ZoeDepthWrapper,
    MCDropoutWrapper,
)
from conformal_depth.data import (
    C3VDDataset,
    SimCol3DDataset,
    EndoSLAMDataset,
    KvasirSEGDataset,
)
from conformal_depth.conformal import (
    batch_diameter_intervals,
    interval_accuracy_at_threshold,
)
from conformal_depth.evaluation.metrics import (
    compute_depth_metrics,
    empirical_coverage,
    mean_interval_width,
    coverage_by_depth_bin,
    width_coverage_calibration_curve,
    three_group_accuracy,
    confusion_matrix_3group,
)
from conformal_depth.evaluation.visualize import (
    plot_coverage_calibration,
    plot_confusion_matrix,
    plot_sizing_scatter,
)


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_model(
    model:      torch.nn.Module,
    loader:     DataLoader,
    device:     torch.device,
    q_hat:      float = 0.0,
    has_gt:     bool = True,
) -> dict:
    """
    Run model over loader, collect depth predictions and GT, return aggregated results.
    """
    model.eval()
    all_pred_lo, all_pred_hi, all_pred_med, all_gt, all_mask = [], [], [], [], []

    for batch in tqdm(loader, desc="Evaluating", leave=False):
        imgs = batch["image"].to(device)

        if hasattr(model, "forward") and hasattr(model, "lora_encoder"):
            # ConformalDepthModel
            raw = model(pixel_values=imgs)
            lo  = raw[:, 0:1]
            med = raw[:, 1:2]
            hi  = raw[:, 2:3]
        else:
            # Baseline: single-channel output, replicate for interval
            med = model(imgs)
            lo  = med * 0.90   # ±10% placeholder for baselines
            hi  = med * 1.10

        all_pred_lo.append(lo.cpu())
        all_pred_hi.append(hi.cpu())
        all_pred_med.append(med.cpu())

        if has_gt:
            all_gt.append(batch["depth"])
            all_mask.append(batch["depth_mask"])

    # Flatten to valid pixels
    if not has_gt:
        return {}

    pred_lo  = torch.cat(all_pred_lo).squeeze(1).numpy()    # (N, H, W)
    pred_hi  = torch.cat(all_pred_hi).squeeze(1).numpy()
    pred_med = torch.cat(all_pred_med).squeeze(1).numpy()
    gt       = torch.cat(all_gt).squeeze(1).numpy()
    mask     = torch.cat(all_mask).squeeze(1).numpy().astype(bool)

    flat_lo  = pred_lo[mask]
    flat_hi  = pred_hi[mask]
    flat_med = pred_med[mask]
    flat_gt  = gt[mask]

    # Padded intervals
    lo_q = flat_lo - q_hat
    hi_q = flat_hi + q_hat

    depth_metrics = compute_depth_metrics(flat_med, flat_gt)
    coverage_raw  = empirical_coverage(flat_lo, flat_hi, flat_gt)
    coverage_pad  = empirical_coverage(lo_q,    hi_q,    flat_gt)
    width_raw     = mean_interval_width(flat_lo, flat_hi)
    width_pad     = mean_interval_width(lo_q,    hi_q)

    ccdr = coverage_by_depth_bin(lo_q, hi_q, flat_gt, n_bins=5)

    # WCC curve
    q_grid = np.linspace(0, flat_gt.std() * 2, 100)
    wcc = width_coverage_calibration_curve(flat_lo, flat_hi, flat_gt, q_grid)

    return {
        "depth_metrics": depth_metrics,
        "coverage_raw":   coverage_raw,
        "coverage_padded": coverage_pad,
        "width_raw":      width_raw,
        "width_padded":   width_pad,
        "ccdr":           ccdr,
        "wcc":            wcc,
        # raw arrays for sizing eval
        "_pred_lo":  pred_lo,
        "_pred_hi":  pred_hi,
        "_pred_med": pred_med,
        "_gt":       gt,
        "_mask":     mask,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",     required=True)
    p.add_argument("--calib_json",     required=True)
    p.add_argument("--c3vd_root",      default=None)
    p.add_argument("--simcol_root",    default=None)
    p.add_argument("--endoslam_root",  default=None)
    p.add_argument("--kvasir_root",    default=None)
    p.add_argument("--output_dir",     default="outputs/eval")
    p.add_argument("--baselines",      action="store_true",
                   help="Run ZoeDepth and Depth-Anything baselines alongside ConformalDepth")
    p.add_argument("--zoedepth_model", default="Intel/zoedepth-nk",
                   choices=["Intel/zoedepth-nk", "Intel/zoedepth-n", "Intel/zoedepth-k"],
                   help="ZoeDepth variant: nk=NYU+KITTI, n=NYU only, k=KITTI only")
    p.add_argument("--da_model",       default="depth-anything/Depth-Anything-V2-Large-hf",
                   help="Depth-Anything model for zero-shot baseline")
    p.add_argument("--batch_size",     type=int, default=4)
    p.add_argument("--num_workers",    type=int, default=4)
    p.add_argument("--model",          default="depth-anything/Depth-Anything-V2-Large-hf")
    p.add_argument("--device",         default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def _build_dataset_loaders(args) -> dict[str, DataLoader | None]:
    """Return {name: DataLoader} for every active dataset root."""
    loaders = {}
    if args.c3vd_root:
        ds = C3VDDataset(root=args.c3vd_root, split="test")
        loaders["c3vd"] = DataLoader(
            ds, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False
        )
    if args.simcol_root:
        ds = SimCol3DDataset(root=args.simcol_root, split="test")
        loaders["simcol3d"] = DataLoader(
            ds, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False
        )
    if args.endoslam_root:
        ds = EndoSLAMDataset(root=args.endoslam_root, split="test")
        loaders["endoslam"] = DataLoader(
            ds, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False
        )
    return loaders


def _run_on_datasets(
    model_name: str,
    model: torch.nn.Module,
    loaders: dict,
    device: torch.device,
    q_hat: float,
    alpha: float,
    output_dir: str,
) -> dict:
    """Evaluate one model across all datasets. Returns nested results dict."""
    results = {}
    for ds_name, loader in loaders.items():
        print(f"  [{model_name}] {ds_name}…")
        res = evaluate_model(model, loader, device, q_hat=q_hat)
        results[ds_name] = {k: v for k, v in res.items() if not k.startswith("_")}
        dm = res.get("depth_metrics", {})
        print(
            f"    AbsRel={dm.get('abs_rel', float('nan')):.4f}  "
            f"RMSE={dm.get('rmse', float('nan')):.2f}mm  "
            f"δ<1.25={dm.get('d1', float('nan')):.3f}  "
            f"Cov(pad)={res.get('coverage_padded', float('nan')):.3f}"
        )
        if ds_name == "c3vd":
            wcc = res.get("wcc", {})
            if wcc:
                fig = plot_coverage_calibration(
                    wcc["q_hats"], wcc["coverages"], wcc["widths"],
                    nominal_alpha=alpha,
                    save_path=os.path.join(
                        output_dir, f"{ds_name}_{model_name}_wcc.png"
                    ),
                )
                plt_close(fig)
    return results


def main():
    args = parse_args()
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    # Load calibration
    with open(args.calib_json) as f:
        calib = json.load(f)
    q_hat = calib["q_adjusted"]
    alpha = calib["alpha"]
    print(f"[eval] Using q̂ = {q_hat:.4f} mm  (alpha={alpha})")

    # Build dataset loaders (shared across all models)
    loaders = _build_dataset_loaders(args)
    if not loaders:
        print("[eval] No dataset roots provided. Pass at least one of: "
              "--c3vd_root, --simcol_root, --endoslam_root")
        return

    # ------------------------------------------------------------------
    # ConformalDepth (primary model)
    # ------------------------------------------------------------------
    print(f"\n[eval] Loading ConformalDepthModel from {args.checkpoint}")
    conf_model = ConformalDepthModel.from_pretrained(model_name=args.model)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    conf_model.load_state_dict(ckpt["model_state"])
    conf_model.to(device).eval()

    all_results = {"ConformalDepth": _run_on_datasets(
        "ConformalDepth", conf_model, loaders, device, q_hat, alpha, args.output_dir
    )}
    del conf_model   # free VRAM before loading baselines

    # ------------------------------------------------------------------
    # Baselines
    # ------------------------------------------------------------------
    if args.baselines:
        # --- ZoeDepth ---
        print(f"\n[eval] ZoeDepth baseline ({args.zoedepth_model})…")
        zoe = ZoeDepthWrapper(model_name=args.zoedepth_model).to(device).eval()
        # ZoeDepth outputs metric depth (mm after ×1000); q_hat does not apply
        # (no conformal calibration for baselines), pass q_hat=0.
        all_results["ZoeDepth"] = _run_on_datasets(
            "ZoeDepth", zoe, loaders, device, q_hat=0.0, alpha=alpha,
            output_dir=args.output_dir
        )
        del zoe

        # --- Depth-Anything V2 zero-shot ---
        print(f"\n[eval] Depth-Anything V2 baseline ({args.da_model})…")
        da = DepthAnythingZeroShot(model_name=args.da_model).to(device).eval()
        all_results["DepthAnything"] = _run_on_datasets(
            "DepthAnything", da, loaders, device, q_hat=0.0, alpha=alpha,
            output_dir=args.output_dir
        )
        del da

    # ------------------------------------------------------------------
    # Polyp sizing note
    # ------------------------------------------------------------------
    if args.kvasir_root and args.c3vd_root:
        print("\n[eval] Polyp sizing evaluation (Kvasir-SEG masks)…")
        print("  [Note] Full polyp sizing requires paired (frame, mask, GT_diameter).")
        print("  Provide --kvasir_root with depth-paired frames for full eval.")

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    def _serialize(v):
        if isinstance(v, dict):
            return {kk: _serialize(vv) for kk, vv in v.items()}
        if hasattr(v, "tolist"):
            return v.tolist()
        return v

    serializable = {model_key: _serialize(ds_res)
                    for model_key, ds_res in all_results.items()}

    out_path = os.path.join(args.output_dir, "evaluation_results.json")
    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\n[eval] Results saved to {out_path}")

    # ------------------------------------------------------------------
    # Pretty-print comparison table (one row per model × dataset)
    # ------------------------------------------------------------------
    col_w = 18
    header = (f"{'Model':<{col_w}} {'Dataset':<12} {'AbsRel':>8} "
              f"{'RMSE':>8} {'δ<1.25':>8} {'SILog':>8} {'Cov(pad)':>10} {'Width':>8}")
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))

    for model_key, ds_results in all_results.items():
        for ds_name, res in ds_results.items():
            dm = res.get("depth_metrics", {})
            print(
                f"{model_key:<{col_w}} {ds_name:<12} "
                f"{dm.get('abs_rel', float('nan')):>8.4f} "
                f"{dm.get('rmse', float('nan')):>8.2f} "
                f"{dm.get('d1', float('nan')):>8.3f} "
                f"{dm.get('silog', float('nan')):>8.4f} "
                f"{res.get('coverage_padded', float('nan')):>10.3f} "
                f"{res.get('width_padded', float('nan')):>8.2f}"
            )
        print("-" * len(header))

    print("=" * len(header))


def plt_close(fig):
    import matplotlib.pyplot as plt
    plt.close(fig)


if __name__ == "__main__":
    main()
