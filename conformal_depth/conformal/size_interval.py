"""
Back-projection: depth intervals → metric polyp diameter intervals.

Given:
  - A binary polyp mask M ⊆ {1..H} × {1..W}
  - Per-pixel depth interval [d_lo(u,v), d_hi(u,v)]  (in mm)
  - Camera intrinsics K = (fx, fy, cx, cy)
  - A conformal slack q̂ that pads the depth interval

We compute a guaranteed interval on the polyp's principal-axis diameter D:
  D_lo = diameter from back-projecting (mask, d_hi) × deflation factor
  D_hi = diameter from back-projecting (mask, d_lo) × inflation factor

The principal axis diameter is the length of the major axis of the 2D
ellipse fitted to the mask, scaled by the mean depth at that axis.

Formally, for a pixel (u, v) with depth d:
    X = (u - cx) * d / fx
    Y = (v - cy) * d / fy
    Z = d

The 3D polyp point cloud is {(X_i, Y_i, Z_i)} for i ∈ M.
Diameter = distance between the two extremal 3D points along the principal
component of (X_i, Y_i) (ignoring depth variation within the lesion).

We then propagate depth interval through PCA via interval arithmetic:
  - For D_lo: use d_lo (smallest depth → smallest angular spread in 3D)
  - For D_hi: use d_hi (largest depth → largest spread)
  - Additional ±q̂ padding is applied symmetrically to both lo and hi.
"""

from __future__ import annotations

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Core back-projection
# ---------------------------------------------------------------------------

def backproject_mask(
    mask: np.ndarray,          # (H, W) bool
    depth: np.ndarray,         # (H, W) float32, mm
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> np.ndarray:
    """
    Back-project masked pixels to 3D camera-frame coordinates.

    Returns:
        (N, 3) float64 array of 3D points [X, Y, Z] in mm.
    """
    vs, us = np.where(mask)               # pixel row, col
    d = depth[vs, us].astype(np.float64)

    X = (us - cx) * d / fx
    Y = (vs - cy) * d / fy
    Z = d

    return np.stack([X, Y, Z], axis=1)   # (N, 3)


def principal_axis_diameter(points_3d: np.ndarray) -> float:
    """
    Compute the major-axis diameter of a 3D point cloud.

    Projects onto the PCA plane (XY in camera frame, ignoring depth), fits
    the principal component, and returns the span along that component in mm.

    Args:
        points_3d: (N, 3) array, N ≥ 2.

    Returns:
        Diameter in mm (scalar float).
    """
    if len(points_3d) < 2:
        return 0.0

    # Work in the XY plane (transverse plane perpendicular to the optical axis)
    pts_xy = points_3d[:, :2]   # (N, 2)
    centroid = pts_xy.mean(axis=0)
    pts_centered = pts_xy - centroid

    # PCA via SVD
    _, _, Vt = np.linalg.svd(pts_centered, full_matrices=False)
    principal = Vt[0]                      # (2,) unit vector

    projections = pts_centered @ principal
    diameter = float(projections.max() - projections.min())
    return diameter


# ---------------------------------------------------------------------------
# Interval arithmetic over depth → diameter interval
# ---------------------------------------------------------------------------

def polyp_diameter_interval(
    mask: np.ndarray,          # (H, W) bool — polyp segmentation mask
    depth_lo: np.ndarray,      # (H, W) float32, conformal lower depth, mm
    depth_hi: np.ndarray,      # (H, W) float32, conformal upper depth, mm
    intrinsics: np.ndarray,    # (4,) [fx, fy, cx, cy]
    q_hat: float = 0.0,        # additional conformal depth slack (mm)
    min_mask_pixels: int = 50,
) -> tuple[float, float]:
    """
    Compute a guaranteed [D_lo, D_hi] interval on polyp diameter in mm.

    Depth monotonically scales the 3D extent: larger depth → larger diameter
    for the same pixel footprint.  Therefore:
        D_lo = diameter(backproject(mask, depth_lo − q̂))
        D_hi = diameter(backproject(mask, depth_hi + q̂))

    Args:
        mask:       Binary polyp mask.
        depth_lo:   Per-pixel lower depth bound (from quantile head).
        depth_hi:   Per-pixel upper depth bound (from quantile head).
        intrinsics: Camera intrinsics [fx, fy, cx, cy] in pixels.
        q_hat:      Conformal depth slack to pad the interval.
        min_mask_pixels: Minimum number of mask pixels required.

    Returns:
        (D_lo, D_hi) in mm.  Returns (0, 0) if mask is too small.
    """
    fx, fy, cx, cy = intrinsics

    if mask.sum() < min_mask_pixels:
        return (0.0, 0.0)

    # Pad depth interval with conformal slack
    d_lo_padded = np.maximum(depth_lo - q_hat, 1e-3)     # keep depth positive
    d_hi_padded = depth_hi + q_hat

    # Back-project with lower bound depth → smallest plausible 3D size
    pts_lo = backproject_mask(mask, d_lo_padded, fx, fy, cx, cy)
    D_lo   = principal_axis_diameter(pts_lo)

    # Back-project with upper bound depth → largest plausible 3D size
    pts_hi = backproject_mask(mask, d_hi_padded, fx, fy, cx, cy)
    D_hi   = principal_axis_diameter(pts_hi)

    # Ensure D_lo ≤ D_hi (can flip if depth very uncertain)
    if D_lo > D_hi:
        D_lo, D_hi = D_hi, D_lo

    return (D_lo, D_hi)


# ---------------------------------------------------------------------------
# Batch version (tensor input from model)
# ---------------------------------------------------------------------------

@torch.no_grad()
def batch_diameter_intervals(
    masks:       torch.Tensor,    # (B, 1, H, W) bool
    depth_lo:    torch.Tensor,    # (B, 1, H, W) float
    depth_hi:    torch.Tensor,    # (B, 1, H, W) float
    intrinsics:  torch.Tensor,    # (B, 4)        [fx, fy, cx, cy]
    q_hat: float = 0.0,
) -> list[tuple[float, float]]:
    """
    Compute diameter intervals for a batch of frames.

    Returns list of (D_lo, D_hi) tuples, one per sample in the batch.
    """
    B = masks.shape[0]
    intervals = []

    mask_np  = masks.squeeze(1).cpu().numpy().astype(bool)    # (B, H, W)
    lo_np    = depth_lo.squeeze(1).cpu().numpy()              # (B, H, W)
    hi_np    = depth_hi.squeeze(1).cpu().numpy()              # (B, H, W)
    K_np     = intrinsics.cpu().numpy()                        # (B, 4)

    for i in range(B):
        d_lo, d_hi = polyp_diameter_interval(
            mask=mask_np[i],
            depth_lo=lo_np[i],
            depth_hi=hi_np[i],
            intrinsics=K_np[i],
            q_hat=q_hat,
        )
        intervals.append((d_lo, d_hi))

    return intervals


# ---------------------------------------------------------------------------
# Clinical size classification  (Paris classification thresholds)
# ---------------------------------------------------------------------------

# Paris classification: <5mm hyperplastic, 5–9mm small adenoma, ≥10mm large
PARIS_THRESHOLDS_MM = [5.0, 10.0]

def classify_size(diameter_mm: float) -> str:
    """Return Paris classification group for a given diameter."""
    if diameter_mm < PARIS_THRESHOLDS_MM[0]:
        return "diminutive"       # < 5 mm
    elif diameter_mm < PARIS_THRESHOLDS_MM[1]:
        return "small"            # 5–9 mm
    else:
        return "large"            # ≥ 10 mm


def classify_interval(D_lo: float, D_hi: float) -> str:
    """
    Classify the size interval as the worst-case Paris group that the
    interval *could* contain.

    If the interval spans a threshold we report the larger class (conservative
    for clinical over-detection).
    """
    return classify_size(D_hi)


def interval_accuracy_at_threshold(
    pred_intervals: list[tuple[float, float]],
    gt_diameters:   list[float],
    n_groups: int = 3,
) -> dict[str, float]:
    """
    Compute 3-group size classification accuracy as reported in Clinical
    Endoscopy (AI accuracy 89.9%, endoscopist 54.7%).

    Args:
        pred_intervals: List of (D_lo, D_hi) tuples.
        gt_diameters:   List of GT diameters in mm.
        n_groups:       2 (5mm threshold) or 3 (5mm + 10mm).

    Returns:
        dict with "accuracy", "correct", "total".
    """
    assert len(pred_intervals) == len(gt_diameters)

    thresholds = PARIS_THRESHOLDS_MM[:n_groups - 1]

    def group(d: float) -> int:
        for i, t in enumerate(thresholds):
            if d < t:
                return i
        return len(thresholds)

    correct = 0
    for (lo, hi), gt in zip(pred_intervals, gt_diameters):
        pred_d = (lo + hi) / 2.0    # point estimate = midpoint
        if group(pred_d) == group(gt):
            correct += 1

    total = len(gt_diameters)
    return {
        "accuracy": correct / total if total > 0 else 0.0,
        "correct":  correct,
        "total":    total,
    }
