"""
Evaluation metrics for ConformalDepth.

Depth metrics:
  - AbsRel, SqRel, RMSE, RMSElog, δ<1.25/1.25²/1.25³  (standard benchmarks)
  - SILog (scale-invariant log)
  - Spearman ρ (for zero-shot relative-depth models)

Conformal / uncertainty metrics:
  - Empirical coverage at nominal level α
  - Mean interval width (efficiency)
  - Width-coverage calibration curve (WCC)
  - Coverage Conditional on Depth Range (CCDR) — are coverage guarantees
    maintained across different depth bins?

Clinical sizing metrics:
  - Diameter RMSE and MAE (mm)
  - Concordance Correlation Coefficient (CCC)
  - 3-group Paris classification accuracy (diminutive / small / large)
  - Confusion matrix for the 3-group classification
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.stats import spearmanr, pearsonr


# ---------------------------------------------------------------------------
# Standard depth metrics
# ---------------------------------------------------------------------------

def compute_depth_metrics(
    pred: np.ndarray,    # (N,) predicted depth values, valid pixels
    gt:   np.ndarray,    # (N,) GT depth values, valid pixels
    eps:  float = 1e-6,
) -> dict[str, float]:
    """
    Compute standard depth evaluation metrics on flattened valid-pixel arrays.

    Args:
        pred: Predicted metric depth in mm.
        gt:   Ground-truth metric depth in mm.
        eps:  Numerical stability floor.

    Returns:
        Dictionary with keys: abs_rel, sq_rel, rmse, rmse_log, silog,
        d1, d2, d3, spearman_rho.
    """
    assert pred.shape == gt.shape, "pred and gt must have the same shape"
    pred = pred.astype(np.float64).clip(min=eps)
    gt   = gt.astype(np.float64).clip(min=eps)

    thresh = np.maximum(gt / pred, pred / gt)
    d1 = (thresh < 1.25  ).mean()
    d2 = (thresh < 1.25**2).mean()
    d3 = (thresh < 1.25**3).mean()

    abs_rel  = (np.abs(pred - gt) / gt).mean()
    sq_rel   = ((pred - gt) ** 2 / gt).mean()
    rmse     = np.sqrt(((pred - gt) ** 2).mean())
    rmse_log = np.sqrt(((np.log(pred) - np.log(gt)) ** 2).mean())

    log_d  = np.log(pred) - np.log(gt)
    silog  = np.sqrt((log_d ** 2).mean() - 0.5 * (log_d.mean() ** 2))

    rho, _ = spearmanr(pred, gt)

    return {
        "abs_rel":     float(abs_rel),
        "sq_rel":      float(sq_rel),
        "rmse":        float(rmse),
        "rmse_log":    float(rmse_log),
        "silog":       float(silog),
        "d1":          float(d1),
        "d2":          float(d2),
        "d3":          float(d3),
        "spearman_rho": float(rho),
    }


# ---------------------------------------------------------------------------
# Conformal coverage and efficiency
# ---------------------------------------------------------------------------

def empirical_coverage(
    pred_lo:  np.ndarray,    # (N,) lower depth bounds
    pred_hi:  np.ndarray,    # (N,) upper depth bounds
    gt:       np.ndarray,    # (N,) GT depths
) -> float:
    """Fraction of GT depths that fall within the predicted interval."""
    covered = (gt >= pred_lo) & (gt <= pred_hi)
    return float(covered.mean())


def mean_interval_width(pred_lo: np.ndarray, pred_hi: np.ndarray) -> float:
    return float((pred_hi - pred_lo).mean())


def coverage_by_depth_bin(
    pred_lo:  np.ndarray,
    pred_hi:  np.ndarray,
    gt:       np.ndarray,
    n_bins:   int = 5,
) -> dict[str, list]:
    """
    Compute empirical coverage and mean interval width per depth bin.

    Returns a dict with:
        "bin_centers": list of depth centers
        "coverage":    list of per-bin coverage values
        "width":       list of per-bin mean widths
    """
    bins = np.percentile(gt, np.linspace(0, 100, n_bins + 1))
    centers, coverages, widths = [], [], []

    for i in range(n_bins):
        lo_b, hi_b = bins[i], bins[i + 1]
        sel = (gt >= lo_b) & (gt < hi_b)
        if sel.sum() == 0:
            continue
        centers.append(float((lo_b + hi_b) / 2))
        coverages.append(empirical_coverage(pred_lo[sel], pred_hi[sel], gt[sel]))
        widths.append(mean_interval_width(pred_lo[sel], pred_hi[sel]))

    return {"bin_centers": centers, "coverage": coverages, "width": widths}


def width_coverage_calibration_curve(
    pred_lo:    np.ndarray,
    pred_hi:    np.ndarray,
    gt:         np.ndarray,
    q_hats:     np.ndarray,   # grid of λ values
) -> dict[str, np.ndarray]:
    """
    Sweep λ over q_hats and record (coverage, mean_width).

    Useful for plotting the efficiency–coverage trade-off curve.
    """
    coverages = []
    widths    = []
    for q in q_hats:
        lo_q = pred_lo - q
        hi_q = pred_hi + q
        coverages.append(empirical_coverage(lo_q, hi_q, gt))
        widths.append(mean_interval_width(lo_q, hi_q))

    return {
        "q_hats":    q_hats,
        "coverages": np.array(coverages),
        "widths":    np.array(widths),
    }


# ---------------------------------------------------------------------------
# Clinical sizing metrics
# ---------------------------------------------------------------------------

def concordance_correlation_coefficient(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """
    Lin's Concordance Correlation Coefficient (CCC).

    CCC = 2 * ρ * σ_pred * σ_true / (σ_pred² + σ_true² + (μ_pred - μ_true)²)

    Clinically relevant: CCC 0.89–0.93 (AI) vs 0.41 (endoscopist).
    """
    mu_p, mu_t = y_pred.mean(), y_true.mean()
    s_p  = y_pred.std(ddof=0)
    s_t  = y_true.std(ddof=0)
    rho, _ = pearsonr(y_pred, y_true)
    ccc = (2 * rho * s_p * s_t) / (s_p**2 + s_t**2 + (mu_p - mu_t)**2 + 1e-12)
    return float(ccc)


PARIS_THRESHOLDS = [5.0, 10.0]    # mm

def diameter_to_group(d: float) -> int:
    for i, t in enumerate(PARIS_THRESHOLDS):
        if d < t:
            return i
    return len(PARIS_THRESHOLDS)


def three_group_accuracy(
    pred_diameters: np.ndarray,   # (N,) predicted diameters in mm
    gt_diameters:   np.ndarray,   # (N,) GT diameters in mm
) -> dict[str, float]:
    """
    Three-group Paris classification accuracy.
    Reference: AI 89.9%, endoscopist 54.7% (Clinical Endoscopy review).
    """
    pred_groups = np.array([diameter_to_group(d) for d in pred_diameters])
    gt_groups   = np.array([diameter_to_group(d) for d in gt_diameters])
    acc = float((pred_groups == gt_groups).mean())

    # Per-class recall
    recalls = {}
    for g, name in enumerate(["diminutive", "small", "large"]):
        in_class = gt_groups == g
        if in_class.sum() > 0:
            recalls[f"recall_{name}"] = float((pred_groups[in_class] == g).mean())

    return {
        "accuracy": acc,
        "rmse_mm":  float(np.sqrt(((pred_diameters - gt_diameters) ** 2).mean())),
        "mae_mm":   float(np.abs(pred_diameters - gt_diameters).mean()),
        "ccc":      concordance_correlation_coefficient(pred_diameters, gt_diameters),
        **recalls,
    }


def confusion_matrix_3group(
    pred_diameters: np.ndarray,
    gt_diameters:   np.ndarray,
) -> np.ndarray:
    """Returns 3×3 confusion matrix for Paris classification."""
    from sklearn.metrics import confusion_matrix
    pred_g = np.array([diameter_to_group(d) for d in pred_diameters])
    gt_g   = np.array([diameter_to_group(d) for d in gt_diameters])
    return confusion_matrix(gt_g, pred_g, labels=[0, 1, 2])


# ---------------------------------------------------------------------------
# Full evaluation report
# ---------------------------------------------------------------------------

def full_evaluation_report(
    pred_lo:        np.ndarray,   # (N,) per-pixel lower bounds
    pred_hi:        np.ndarray,   # (N,) per-pixel upper bounds
    pred_med:       np.ndarray,   # (N,) median predictions
    gt_depth:       np.ndarray,   # (N,) GT depths (mm)
    pred_diameters: np.ndarray,   # (M,) predicted polyp diameters
    gt_diameters:   np.ndarray,   # (M,) GT polyp diameters
    q_hat: float = 0.0,
    alpha: float = 0.10,
) -> dict:
    """Assemble a complete evaluation report."""
    # Padded intervals
    lo_q = pred_lo - q_hat
    hi_q = pred_hi + q_hat

    depth_metrics = compute_depth_metrics(pred_med, gt_depth)
    conformal = {
        "nominal_coverage": 1.0 - alpha,
        "empirical_coverage_raw":    empirical_coverage(pred_lo, pred_hi, gt_depth),
        "empirical_coverage_padded": empirical_coverage(lo_q,    hi_q,    gt_depth),
        "mean_width_raw":            mean_interval_width(pred_lo, pred_hi),
        "mean_width_padded":         mean_interval_width(lo_q,    hi_q),
    }
    sizing = three_group_accuracy(pred_diameters, gt_diameters)

    return {
        "depth_metrics": depth_metrics,
        "conformal":     conformal,
        "sizing":        sizing,
    }
