"""
Nonconformity scores for dense depth conformal prediction.

Three families:
1. QuantileResidual   — signed residual on the predicted quantile interval.
2. ScaleAlignedScore  — residual after affine scale alignment (shift+scale).
3. PixelWiseMax       — takes pixelwise max residual over the polyp mask.

All scores operate on CPU numpy arrays for efficiency during calibration.
"""

from __future__ import annotations

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class NonconformityScore:
    """
    Base protocol: score(pred_lo, pred_hi, gt_depth, mask) → scalar or array.

    Lower score = more conforming.  Calibration threshold q̂ is the
    (1-α)(1 + 1/n)-quantile of scores on the calibration set.
    """

    def score(
        self,
        pred_lo:  np.ndarray,   # (H, W) lower depth prediction
        pred_hi:  np.ndarray,   # (H, W) upper depth prediction
        gt_depth: np.ndarray,   # (H, W) GT depth in mm
        mask:     np.ndarray,   # (H, W) bool, valid pixels
    ) -> float:
        raise NotImplementedError

    def __call__(self, *args, **kwargs) -> float:
        return self.score(*args, **kwargs)


# ---------------------------------------------------------------------------
# 1. Quantile residual  (standard conformalized-quantile-regression score)
# ---------------------------------------------------------------------------

class QuantileResidualScore(NonconformityScore):
    """
    For a pixel with GT depth y, predicted interval [ŷ_lo, ŷ_hi]:
        s = max(ŷ_lo − y,  y − ŷ_hi)   (positive when outside interval)

    The per-frame nonconformity score is the (1-β)-quantile over valid mask
    pixels (β controls how many pixels need coverage within a frame).

    Aggregating over frames as max gives worst-case pixel guarantee;
    mean gives average-pixel guarantee.

    After conformal calibration with threshold q̂:
        Padded interval: [ŷ_lo − q̂,  ŷ_hi + q̂]
    achieves marginal coverage ≥ 1-α on future frames.
    """

    def __init__(self, pixel_quantile: float = 0.9, reduction: str = "max"):
        """
        Args:
            pixel_quantile: Within-frame aggregation quantile (β above).
            reduction: "max" | "mean" | "median" across pixels.
        """
        assert 0 < pixel_quantile <= 1.0
        self.pixel_quantile = pixel_quantile
        self.reduction = reduction

    def score(
        self,
        pred_lo:  np.ndarray,
        pred_hi:  np.ndarray,
        gt_depth: np.ndarray,
        mask:     np.ndarray,
    ) -> float:
        valid_lo = pred_lo[mask]
        valid_hi = pred_hi[mask]
        valid_gt = gt_depth[mask]

        if valid_gt.size == 0:
            return 0.0

        pixel_scores = np.maximum(valid_lo - valid_gt, valid_gt - valid_hi)

        if self.reduction == "max":
            return float(np.percentile(pixel_scores, self.pixel_quantile * 100))
        elif self.reduction == "mean":
            return float(pixel_scores.mean())
        else:
            return float(np.median(pixel_scores))


# ---------------------------------------------------------------------------
# 2. Scale-aligned score
# ---------------------------------------------------------------------------

class ScaleAlignedScore(NonconformityScore):
    """
    First aligns prediction to GT via least-squares affine fit (scale+shift),
    then computes the residual.  This measures how well the *shape* of the
    depth map conforms, decoupled from scale drift — important when the metric
    anchor from illumination self-supervision is imperfect.

    Alignment:  [a, b] = argmin Σ(a*ŷ + b - y)² over valid pixels.
    Score:      max over valid pixels of |a*ŷ_med - y| normalized by ŷ_hi - ŷ_lo.
    """

    def __init__(self, pixel_quantile: float = 0.9):
        self.pixel_quantile = pixel_quantile

    def _align(self, pred: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
        """Least-squares scale and shift: returns (scale, shift)."""
        A = np.stack([pred, np.ones_like(pred)], axis=1)
        result = np.linalg.lstsq(A, gt, rcond=None)
        a, b = result[0]
        return float(a), float(b)

    def score(
        self,
        pred_lo:  np.ndarray,
        pred_hi:  np.ndarray,
        gt_depth: np.ndarray,
        mask:     np.ndarray,
    ) -> float:
        pred_med = (pred_lo + pred_hi) / 2.0
        valid_med = pred_med[mask]
        valid_gt  = gt_depth[mask]
        valid_lo  = pred_lo[mask]
        valid_hi  = pred_hi[mask]

        if valid_gt.size < 2:
            return 0.0

        a, b = self._align(valid_med, valid_gt)
        aligned_med = a * valid_med + b

        width = np.maximum(valid_hi - valid_lo, 1e-6)
        normalized_residual = np.abs(aligned_med - valid_gt) / width

        return float(np.percentile(normalized_residual, self.pixel_quantile * 100))


# ---------------------------------------------------------------------------
# 3. Mask-restricted pixel-wise max  (for polyp-specific sizing)
# ---------------------------------------------------------------------------

class MaskRestrictedScore(NonconformityScore):
    """
    Like QuantileResidualScore but restricted only to the polyp mask region.
    Used during conformal calibration when GT polyp segmentation is available.
    """

    def __init__(self, quantile_within_mask: float = 0.95):
        self.q = quantile_within_mask

    def score(
        self,
        pred_lo:   np.ndarray,
        pred_hi:   np.ndarray,
        gt_depth:  np.ndarray,
        mask:      np.ndarray,
        polyp_mask: np.ndarray | None = None,
    ) -> float:
        active = mask if polyp_mask is None else (mask & polyp_mask)
        if active.sum() == 0:
            return 0.0

        pixel_scores = np.maximum(
            pred_lo[active] - gt_depth[active],
            gt_depth[active] - pred_hi[active],
        )
        return float(np.percentile(pixel_scores, self.q * 100))


# ---------------------------------------------------------------------------
# Score batching utilities
# ---------------------------------------------------------------------------

def compute_calibration_scores(
    model_outputs: list[dict],
    score_fn: NonconformityScore,
) -> np.ndarray:
    """
    Iterate over a list of calibration-set outputs and compute per-frame scores.

    Each element of model_outputs must have keys:
        pred_lo, pred_hi, gt_depth, mask  — all (H, W) numpy float32/bool

    Returns np.ndarray of shape (N,) — one score per frame.
    """
    scores = []
    for out in model_outputs:
        s = score_fn.score(
            pred_lo=out["pred_lo"],
            pred_hi=out["pred_hi"],
            gt_depth=out["gt_depth"],
            mask=out["mask"],
        )
        scores.append(s)
    return np.array(scores, dtype=np.float64)


@torch.no_grad()
def collect_model_outputs(
    model: torch.nn.Module,
    loader,
    device: torch.device,
) -> list[dict]:
    """
    Run model over *loader* and collect per-frame prediction dictionaries.

    Returns a list of dicts with numpy arrays ready for score computation.
    """
    model.eval()
    all_outputs = []

    for batch in loader:
        imgs  = batch["image"].to(device)
        gt    = batch["depth"]          # (B, 1, H, W)
        masks = batch["depth_mask"]     # (B, 1, H, W)

        raw = model(pixel_values=imgs)  # (B, 3, H, W)
        lo  = raw[:, 0:1].cpu()
        hi  = raw[:, 2:3].cpu()

        B = imgs.shape[0]
        for i in range(B):
            all_outputs.append({
                "pred_lo":  lo[i, 0].numpy(),
                "pred_hi":  hi[i, 0].numpy(),
                "gt_depth": gt[i, 0].numpy(),
                "mask":     masks[i, 0].numpy().astype(bool),
            })

    return all_outputs
