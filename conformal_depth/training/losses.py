"""
Loss functions for ConformalDepth training.

1. PinballLoss      — quantile regression loss for the three-head output.
2. SILogLoss        — scale-invariant log loss for the median head.
3. IlluminationDeclineLoss — LightDepth-style illumination self-supervision
                             providing the metric scale anchor on unlabeled video.
4. TotalLoss        — weighted combination.

Illumination-Decline Self-Supervision (metric anchor)
------------------------------------------------------
In endoscopy, light intensity I(d) follows an inverse-square law:
    I(d) ≈ I₀ / d²

Therefore log(I(d)) = log(I₀) - 2·log(d).
For two pixels (u₁, d₁) and (u₂, d₂) with the same surface albedo:
    log(I₁) - log(I₂) = -2·(log(d₁) - log(d₂))
    → d₁/d₂ = exp(-0.5·(log(I₁) - log(I₂)))

This gives a self-supervised scale signal: the ratio of depths between two
pixels is constrained by the ratio of their luminances.  We implement this
as a contrastive loss on (pixel-pair, depth-ratio) samples drawn from each
frame.

Reference: LightDepth (Rodríguez-Puigvert et al., 2023).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. Pinball (quantile) loss
# ---------------------------------------------------------------------------

class PinballLoss(nn.Module):
    """
    Pinball loss for quantile regression.

    For quantile τ ∈ (0, 1) and residual r = y − ŷ:
        ρ_τ(r) = τ · r         if r ≥ 0  (under-prediction penalty)
               = (τ − 1) · r   if r < 0  (over-prediction penalty)

    The model outputs three channels [lo, median, hi] with quantile levels
    [α/2, 0.5, 1−α/2].

    Args:
        alpha: Target miscoverage level.  Quantile levels become
               [alpha/2, 0.5, 1 - alpha/2].
    """

    def __init__(self, alpha: float = 0.10):
        super().__init__()
        self.taus = torch.tensor([alpha / 2, 0.5, 1.0 - alpha / 2])

    def forward(
        self,
        pred: torch.Tensor,       # (B, 3, H, W)
        target: torch.Tensor,     # (B, 1, H, W)  GT depth
        mask: torch.Tensor,       # (B, 1, H, W)  valid pixel mask (bool)
    ) -> torch.Tensor:
        taus = self.taus.to(pred.device).view(1, 3, 1, 1)
        target_exp = target.expand_as(pred)     # (B, 3, H, W)
        mask_exp   = mask.expand_as(pred)

        residual = target_exp - pred             # r = y - ŷ
        loss_pos = taus       * residual.clamp(min=0)
        loss_neg = (taus - 1) * residual.clamp(max=0)
        loss_per_pixel = loss_pos + loss_neg     # (B, 3, H, W)

        return loss_per_pixel[mask_exp].mean()


# ---------------------------------------------------------------------------
# 2. Scale-invariant log loss (SILog)
# ---------------------------------------------------------------------------

class SILogLoss(nn.Module):
    """
    Scale-invariant log depth loss (Eigen et al., NIPS 2014).

    SILog(ŷ, y) = mean((log ŷ − log y)²) − λ · mean(log ŷ − log y)²

    Applied to the median head only.
    """

    def __init__(self, lam: float = 0.5, eps: float = 1e-6):
        super().__init__()
        self.lam = lam
        self.eps = eps

    def forward(
        self,
        pred_median: torch.Tensor,   # (B, 1, H, W)
        target:      torch.Tensor,   # (B, 1, H, W)
        mask:        torch.Tensor,   # (B, 1, H, W)
    ) -> torch.Tensor:
        pred   = pred_median[mask].clamp(min=self.eps)
        gt     = target[mask].clamp(min=self.eps)
        log_d  = torch.log(pred) - torch.log(gt)
        return (log_d ** 2).mean() - self.lam * (log_d.mean() ** 2)


# ---------------------------------------------------------------------------
# 3. Illumination-decline self-supervised loss (metric anchor)
# ---------------------------------------------------------------------------

class IlluminationDeclineLoss(nn.Module):
    """
    Self-supervised metric scale anchor via inverse-square illumination model.

    For a random pair of pixels (i, j) sampled from the same frame:
        L_illum = (log(I_j/I_i) + 2·log(d_j/d_i))²

    where I_k is the luminance of pixel k and d_k is its predicted depth.

    In expectation this is zero only when the absolute depth scale is correct.
    The loss gradient pulls the global depth scale toward the metric value
    implied by the scene illumination.

    Args:
        n_pairs: Number of random pixel pairs sampled per frame.
        min_luminance: Pixels darker than this are excluded (specular noise).
    """

    def __init__(self, n_pairs: int = 512, min_luminance: float = 0.05):
        super().__init__()
        self.n_pairs = n_pairs
        self.min_luminance = min_luminance

    def _luminance(self, images: torch.Tensor) -> torch.Tensor:
        """Convert (B, 3, H, W) RGB to (B, 1, H, W) luminance."""
        weights = images.new_tensor([0.2126, 0.7152, 0.0722]).view(1, 3, 1, 1)
        return (images * weights).sum(dim=1, keepdim=True)

    def forward(
        self,
        pred_depth: torch.Tensor,    # (B, 1, H, W) median depth
        images:     torch.Tensor,    # (B, 3, H, W) input frames
    ) -> torch.Tensor:
        B, _, H, W = images.shape
        N = self.n_pairs
        eps = 1e-6

        luminance = self._luminance(images)   # (B, 1, H, W)
        lum_flat  = luminance.view(B, -1)     # (B, H*W)
        dep_flat  = pred_depth.view(B, -1)    # (B, H*W)

        # Sample random pixel indices
        valid = (lum_flat > self.min_luminance)  # (B, H*W)
        loss_total = torch.tensor(0.0, device=images.device)
        count = 0

        for b in range(B):
            valid_idx = valid[b].nonzero(as_tuple=False).squeeze(1)
            n_valid = len(valid_idx)
            if n_valid < 2:
                continue

            n_sample = min(N, n_valid)
            perm = torch.randperm(n_valid, device=images.device)[:n_sample]
            idx = valid_idx[perm]

            # Split into pairs
            n_pair = n_sample // 2
            idx_i = idx[:n_pair]
            idx_j = idx[n_pair:2 * n_pair]

            lum_i = lum_flat[b][idx_i].clamp(min=eps)
            lum_j = lum_flat[b][idx_j].clamp(min=eps)
            dep_i = dep_flat[b][idx_i].clamp(min=eps)
            dep_j = dep_flat[b][idx_j].clamp(min=eps)

            log_lum_ratio   = torch.log(lum_j / lum_i)
            log_depth_ratio = torch.log(dep_j / dep_i)

            # Residual: log(I_j/I_i) + 2·log(d_j/d_i) should be 0
            residual = log_lum_ratio + 2.0 * log_depth_ratio
            loss_total = loss_total + (residual ** 2).mean()
            count += 1

        return loss_total / max(count, 1)


# ---------------------------------------------------------------------------
# 4. Interval width regularization
# ---------------------------------------------------------------------------

class WidthRegularizationLoss(nn.Module):
    """
    Penalize unnecessarily wide intervals.

    Encourages the model to learn tight intervals: penalizes (d_hi - d_lo)
    above a floor proportional to the local depth.

    L_width = mean(max(0, d_hi - d_lo - β·d_median)²)

    where β is the minimum relative width target.
    """

    def __init__(self, beta: float = 0.05):
        super().__init__()
        self.beta = beta

    def forward(
        self,
        pred: torch.Tensor,   # (B, 3, H, W)
        mask: torch.Tensor,   # (B, 1, H, W)
    ) -> torch.Tensor:
        lo     = pred[:, 0:1]
        median = pred[:, 1:2]
        hi     = pred[:, 2:3]
        width  = hi - lo
        target_width = self.beta * median.detach()
        excess = F.relu(width - target_width)
        return (excess[mask.expand_as(excess)] ** 2).mean()


# ---------------------------------------------------------------------------
# 5. Monotonicity constraint: d_lo ≤ d_med ≤ d_hi
# ---------------------------------------------------------------------------

class MonotonicityLoss(nn.Module):
    """
    Penalize quantile crossing: enforce d_lo ≤ d_med ≤ d_hi.

    L_cross = mean(ReLU(lo - med)²) + mean(ReLU(med - hi)²)
    """

    def forward(self, pred: torch.Tensor) -> torch.Tensor:
        lo     = pred[:, 0]
        median = pred[:, 1]
        hi     = pred[:, 2]
        loss_lo = F.relu(lo - median).pow(2).mean()
        loss_hi = F.relu(median - hi).pow(2).mean()
        return loss_lo + loss_hi


# ---------------------------------------------------------------------------
# 6. Combined total loss
# ---------------------------------------------------------------------------

class TotalLoss(nn.Module):
    """
    Weighted sum of all loss components.

    Weights:
        w_pinball:  Pinball loss on all three quantile heads.
        w_silog:    SILog on median head.
        w_illum:    Illumination decline self-supervision.
        w_width:    Interval width regularization.
        w_mono:     Quantile monotonicity constraint.
    """

    def __init__(
        self,
        alpha: float = 0.10,
        w_pinball: float = 1.0,
        w_silog:   float = 0.5,
        w_illum:   float = 0.1,
        w_width:   float = 0.01,
        w_mono:    float = 0.1,
    ):
        super().__init__()
        self.w_pinball = w_pinball
        self.w_silog   = w_silog
        self.w_illum   = w_illum
        self.w_width   = w_width
        self.w_mono    = w_mono

        self.pinball = PinballLoss(alpha=alpha)
        self.silog   = SILogLoss()
        self.illum   = IlluminationDeclineLoss()
        self.width   = WidthRegularizationLoss()
        self.mono    = MonotonicityLoss()

    def forward(
        self,
        pred:   torch.Tensor,    # (B, 3, H, W)
        target: torch.Tensor,    # (B, 1, H, W)
        mask:   torch.Tensor,    # (B, 1, H, W) bool
        images: torch.Tensor,    # (B, 3, H, W) for illumination loss
    ) -> dict[str, torch.Tensor]:
        losses = {}

        losses["pinball"] = self.w_pinball * self.pinball(pred, target, mask)
        losses["silog"]   = self.w_silog   * self.silog(pred[:, 1:2], target, mask)
        losses["illum"]   = self.w_illum   * self.illum(pred[:, 1:2], images)
        losses["width"]   = self.w_width   * self.width(pred, mask)
        losses["mono"]    = self.w_mono    * self.mono(pred)
        losses["total"]   = sum(losses.values())

        return losses
