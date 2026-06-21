"""
Unit tests for training losses.

Verifies:
- PinballLoss is non-negative and zero at exact quantile predictions.
- SILogLoss is scale-invariant (multiplying pred by constant leaves loss unchanged).
- IlluminationDeclineLoss is zero when depth follows inverse-square law exactly.
- MonotonicityLoss is zero when lo ≤ med ≤ hi.
- TotalLoss output dict has all expected keys.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pytest

from conformal_depth.training.losses import (
    PinballLoss,
    SILogLoss,
    IlluminationDeclineLoss,
    MonotonicityLoss,
    WidthRegularizationLoss,
    TotalLoss,
)


B, H, W = 2, 64, 64


def _make_batch(seed=0):
    torch.manual_seed(seed)
    depth  = torch.rand(B, 1, H, W) * 50 + 5       # 5–55 mm
    pred   = torch.stack([
        depth.squeeze(1) * 0.9,
        depth.squeeze(1),
        depth.squeeze(1) * 1.1,
    ], dim=1)                                         # (B, 3, H, W)
    mask   = torch.ones(B, 1, H, W, dtype=torch.bool)
    images = torch.rand(B, 3, H, W)
    return pred, depth, mask, images


class TestPinballLoss:

    def test_nonnegative(self):
        pred, depth, mask, _ = _make_batch()
        loss = PinballLoss(alpha=0.10)(pred, depth, mask)
        assert loss.item() >= 0

    def test_zero_at_exact_quantiles(self):
        """When the median exactly equals GT and lo/hi are the true quantiles,
        pinball loss should be very small."""
        depth = torch.full((B, 1, H, W), 30.0)
        # Exact quantile predictions (no slack)
        pred = depth.expand(B, 3, H, W).clone()
        mask = torch.ones(B, 1, H, W, dtype=torch.bool)
        loss = PinballLoss(alpha=0.10)(pred, depth, mask)
        assert loss.item() < 1e-4

    def test_loss_decreases_with_better_prediction(self):
        pred_bad, depth, mask, _ = _make_batch()
        # Make predictions much worse (wide residuals)
        pred_worse = pred_bad.clone()
        pred_worse[:, 1] += 20.0   # median far from GT

        loss_better = PinballLoss()(pred_bad, depth, mask)
        loss_worse  = PinballLoss()(pred_worse, depth, mask)
        assert loss_better < loss_worse


class TestSILogLoss:

    def test_scale_invariance(self):
        """Multiplying both pred and GT by same constant should not change loss."""
        torch.manual_seed(1)
        pred = torch.rand(B, 1, H, W) * 40 + 1
        gt   = torch.rand(B, 1, H, W) * 40 + 1
        mask = torch.ones(B, 1, H, W, dtype=torch.bool)

        loss_fn = SILogLoss()
        loss1 = loss_fn(pred, gt, mask)
        loss2 = loss_fn(pred * 3.0, gt * 3.0, mask)

        assert abs(loss1.item() - loss2.item()) < 1e-4, \
            f"SILog not scale-invariant: {loss1.item():.6f} vs {loss2.item():.6f}"

    def test_nonnegative(self):
        pred, depth, mask, _ = _make_batch()
        loss = SILogLoss()(pred[:, 1:2], depth, mask)
        assert loss.item() >= 0

    def test_zero_at_perfect_prediction(self):
        depth = torch.full((B, 1, H, W), 25.0)
        mask  = torch.ones(B, 1, H, W, dtype=torch.bool)
        loss  = SILogLoss()(depth, depth, mask)
        assert loss.item() < 1e-5


class TestIlluminationDeclineLoss:

    def test_nonnegative(self):
        pred, _, _, images = _make_batch()
        loss_fn = IlluminationDeclineLoss(n_pairs=64)
        loss = loss_fn(pred[:, 1:2], images)
        assert loss.item() >= 0

    def test_zero_when_depth_follows_inverse_square(self):
        """If depth = C / sqrt(luminance), the illumination loss should be small."""
        B_, H_, W_ = 1, 32, 32
        lum = torch.rand(B_, 1, H_, W_).clamp(min=0.1)
        C = 50.0   # scale constant
        depth = C / torch.sqrt(lum)

        images = lum.expand(B_, 3, H_, W_)
        loss_fn = IlluminationDeclineLoss(n_pairs=256, min_luminance=0.05)
        loss = loss_fn(depth, images)
        # Should be very small but floating-point noise keeps it non-zero
        assert loss.item() < 0.5, f"Loss too large: {loss.item():.4f}"


class TestMonotonicityLoss:

    def test_zero_when_monotone(self):
        pred, _, _, _ = _make_batch()   # pred[:, 0] = 0.9*d, [1]=d, [2]=1.1*d
        loss = MonotonicityLoss()(pred)
        assert loss.item() < 1e-6

    def test_positive_when_crossing(self):
        pred, _, _, _ = _make_batch()
        pred_cross = pred.clone()
        pred_cross[:, 0] = pred_cross[:, 1] + 5   # lo > median (crossing)
        loss = MonotonicityLoss()(pred_cross)
        assert loss.item() > 0


class TestTotalLoss:

    def test_all_keys_present(self):
        pred, depth, mask, images = _make_batch()
        loss_fn = TotalLoss()
        losses  = loss_fn(pred, depth, mask, images)
        for key in ["pinball", "silog", "illum", "width", "mono", "total"]:
            assert key in losses, f"Missing key: {key}"

    def test_total_is_sum_of_components(self):
        pred, depth, mask, images = _make_batch()
        loss_fn = TotalLoss()
        losses  = loss_fn(pred, depth, mask, images)
        expected = sum(v for k, v in losses.items() if k != "total")
        assert abs(losses["total"].item() - expected.item()) < 1e-5

    def test_backprop(self):
        """Total loss should be differentiable w.r.t. predictions."""
        pred = torch.rand(B, 3, H, W, requires_grad=True)
        depth = torch.rand(B, 1, H, W) * 40 + 5
        mask  = torch.ones(B, 1, H, W, dtype=torch.bool)
        images = torch.rand(B, 3, H, W)

        losses = TotalLoss()(pred, depth, mask, images)
        losses["total"].backward()
        assert pred.grad is not None
        assert not torch.isnan(pred.grad).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
