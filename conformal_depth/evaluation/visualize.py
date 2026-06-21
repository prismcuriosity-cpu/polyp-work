"""
Visualization utilities for ConformalDepth.

Generates:
  - Side-by-side depth map panels (GT, predicted median, interval width)
  - Coverage calibration plots
  - Sequential drift curves
  - Polyp sizing scatter plots with error bars
  - Paris classification confusion matrices
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path


def colorize_depth(depth: np.ndarray, vmin: float | None = None, vmax: float | None = None) -> np.ndarray:
    """Convert (H, W) depth array to (H, W, 3) uint8 RGB using viridis colormap."""
    from matplotlib import colormaps
    cmap = colormaps["viridis"]
    if vmin is None:
        vmin = depth[depth > 0].min() if (depth > 0).any() else 0.0
    if vmax is None:
        vmax = depth.max()
    normalized = np.clip((depth - vmin) / (vmax - vmin + 1e-8), 0.0, 1.0)
    rgba = (cmap(normalized) * 255).astype(np.uint8)
    return rgba[..., :3]


def plot_depth_panel(
    image:      np.ndarray,    # (H, W, 3) uint8 RGB
    gt_depth:   np.ndarray,    # (H, W)
    pred_med:   np.ndarray,    # (H, W)
    pred_lo:    np.ndarray,    # (H, W)
    pred_hi:    np.ndarray,    # (H, W)
    polyp_mask: np.ndarray | None = None,  # (H, W) bool
    save_path:  str | None = None,
    q_hat:      float = 0.0,
) -> plt.Figure:
    """
    5-panel figure: RGB | GT depth | Pred median | Interval width | Coverage mask
    """
    width_map = (pred_hi + q_hat) - (pred_lo - q_hat)
    covered   = (gt_depth >= pred_lo - q_hat) & (gt_depth <= pred_hi + q_hat) & (gt_depth > 0)

    vmin = max(0.0, float(gt_depth[gt_depth > 0].min()) if (gt_depth > 0).any() else 0.0)
    vmax = float(gt_depth.max())

    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    titles = ["Input RGB", "GT Depth (mm)", "Pred Median (mm)",
              "Interval Width (mm)", "Coverage"]

    axes[0].imshow(image)
    axes[1].imshow(gt_depth, cmap="viridis", vmin=vmin, vmax=vmax)
    axes[2].imshow(pred_med, cmap="viridis", vmin=vmin, vmax=vmax)
    axes[3].imshow(width_map, cmap="plasma")
    axes[4].imshow(covered.astype(float), cmap="RdYlGn", vmin=0, vmax=1)

    if polyp_mask is not None:
        for ax in axes[1:]:
            contour = ax.contour(polyp_mask.astype(float), levels=[0.5],
                                 colors=["yellow"], linewidths=1.5)

    for ax, title in zip(axes, titles):
        ax.set_title(title, fontsize=11)
        ax.axis("off")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_coverage_calibration(
    q_hats:     np.ndarray,
    coverages:  np.ndarray,
    widths:     np.ndarray,
    nominal_alpha: float = 0.10,
    save_path:  str | None = None,
) -> plt.Figure:
    """
    Plot the width–coverage calibration curve.

    Left axis: coverage vs λ  (should cross 1-α at the calibrated q̂).
    Right axis: mean width vs λ.
    """
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()

    ax1.plot(q_hats, coverages, "b-o", markersize=3, label="Empirical coverage")
    ax1.axhline(1.0 - nominal_alpha, color="b", linestyle="--",
                label=f"Nominal {100*(1-nominal_alpha):.0f}%")
    ax2.plot(q_hats, widths, "r-s", markersize=3, label="Mean interval width")

    ax1.set_xlabel("Conformal slack λ (mm)")
    ax1.set_ylabel("Empirical coverage", color="b")
    ax2.set_ylabel("Mean interval width (mm)", color="r")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    plt.title("Coverage–Efficiency Trade-off")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_sequential_drift(
    drift_curve:  np.ndarray,     # (T,) drift values
    frame_times:  np.ndarray | None = None,
    calib_end_idx: int | None = None,
    save_path:    str | None = None,
) -> plt.Figure:
    """Plot inter-frame drift curve with calibration/test split marker."""
    T = len(drift_curve)
    x = frame_times if frame_times is not None else np.arange(T)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(x, drift_curve, lw=1.0, color="steelblue")
    ax.fill_between(x, 0, drift_curve, alpha=0.25, color="steelblue")

    if calib_end_idx is not None:
        ax.axvline(x[calib_end_idx], color="red", linestyle="--",
                   label="Calib / Test split")
        ax.legend()

    ax.set_xlabel("Frame index")
    ax.set_ylabel("Feature drift (W₁ proxy)")
    ax.set_title("Inter-frame Distribution Drift in Colonoscopy Sequence")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_sizing_scatter(
    pred_diameters:    np.ndarray,
    gt_diameters:      np.ndarray,
    pred_lo_diameters: np.ndarray | None = None,
    pred_hi_diameters: np.ndarray | None = None,
    ccc: float | None = None,
    save_path:         str | None = None,
) -> plt.Figure:
    """
    Bland–Altman + scatter plot for polyp diameter estimation.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Scatter
    ax = axes[0]
    if pred_lo_diameters is not None and pred_hi_diameters is not None:
        err_lo = pred_diameters - pred_lo_diameters
        err_hi = pred_hi_diameters - pred_diameters
        ax.errorbar(gt_diameters, pred_diameters,
                    yerr=[err_lo, err_hi], fmt="o", alpha=0.5,
                    color="steelblue", ecolor="lightblue", capsize=3)
    else:
        ax.scatter(gt_diameters, pred_diameters, alpha=0.5, color="steelblue")

    lim_max = max(gt_diameters.max(), pred_diameters.max()) * 1.05
    ax.plot([0, lim_max], [0, lim_max], "k--", label="Identity")
    for t in [5.0, 10.0]:
        ax.axvline(t, color="gray", linestyle=":", alpha=0.5)
        ax.axhline(t, color="gray", linestyle=":", alpha=0.5)
    title = "Predicted vs GT Diameter (mm)"
    if ccc is not None:
        title += f"\nCCC = {ccc:.3f}"
    ax.set_title(title)
    ax.set_xlabel("GT Diameter (mm)")
    ax.set_ylabel("Predicted Diameter (mm)")
    ax.legend()

    # Bland–Altman
    ax = axes[1]
    mean_d = (pred_diameters + gt_diameters) / 2
    diff_d = pred_diameters - gt_diameters
    bias = diff_d.mean()
    loa  = 1.96 * diff_d.std()
    ax.scatter(mean_d, diff_d, alpha=0.5, color="darkorange")
    ax.axhline(bias,      color="red",   linestyle="-",  label=f"Bias={bias:.2f}mm")
    ax.axhline(bias + loa, color="red", linestyle="--", label=f"LoA={bias+loa:.2f}mm")
    ax.axhline(bias - loa, color="red", linestyle="--", label=f"LoA={bias-loa:.2f}mm")
    ax.axhline(0, color="gray", linestyle=":")
    ax.set_title("Bland–Altman Plot")
    ax.set_xlabel("Mean of GT and Pred (mm)")
    ax.set_ylabel("Pred − GT (mm)")
    ax.legend(fontsize=8)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_confusion_matrix(
    cm:         np.ndarray,        # (3, 3) confusion matrix
    save_path:  str | None = None,
) -> plt.Figure:
    """Plot 3-group Paris classification confusion matrix."""
    labels = ["Diminutive\n(<5mm)", "Small\n(5–9mm)", "Large\n(≥10mm)"]
    cm_pct = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, data, fmt, title in [
        (axes[0], cm,     "d",    "Confusion Matrix (counts)"),
        (axes[1], cm_pct, ".2f", "Confusion Matrix (recall)"),
    ]:
        sns.heatmap(data, annot=True, fmt=fmt, cmap="Blues",
                    xticklabels=labels, yticklabels=labels, ax=ax,
                    vmin=0, vmax=data.max())
        ax.set_title(title)
        ax.set_ylabel("Ground Truth")
        ax.set_xlabel("Predicted")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
