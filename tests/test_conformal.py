"""
Unit tests for conformal prediction components.

Tests the mathematical guarantees:
1. Conformal quantile satisfies coverage guarantee on hold-out data.
2. RCPS achieves risk control at level α with probability ≥ 1-δ.
3. Interval arithmetic for back-projection is monotone in depth.
4. Nonconformity scores return valid scalars.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from conformal_depth.conformal.calibration import (
    conformal_quantile,
    RCPSCalibrator,
    run_calibration,
)
from conformal_depth.conformal.scores import (
    QuantileResidualScore,
    ScaleAlignedScore,
    MaskRestrictedScore,
)
from conformal_depth.conformal.size_interval import (
    backproject_mask,
    principal_axis_diameter,
    polyp_diameter_interval,
    classify_size,
)


# ---------------------------------------------------------------------------
# Calibration tests
# ---------------------------------------------------------------------------

class TestConformalQuantile:
    """The conformal quantile must guarantee coverage on exchangeable data."""

    def _simulate_calibration(self, n=1000, seed=42):
        rng = np.random.default_rng(seed)
        # Generate (pred_lo, pred_hi, gt) tuples with known coverage
        gt   = rng.uniform(0, 100, n)
        lo   = gt - rng.exponential(5, n)
        hi   = gt + rng.exponential(5, n)
        scores = np.maximum(lo - gt, gt - hi)
        return scores, gt, lo, hi

    def test_coverage_guarantee(self):
        """Standard conformal quantile achieves >= 1-alpha coverage."""
        n_calib = 500
        n_test  = 1000
        alpha   = 0.10
        n_trials = 50
        coverages = []

        rng = np.random.default_rng(0)
        for trial in range(n_trials):
            # Scores from some fixed distribution
            all_scores = rng.exponential(10, n_calib + n_test)
            calib_scores = all_scores[:n_calib]
            test_scores  = all_scores[n_calib:]

            q_hat = conformal_quantile(calib_scores, alpha)
            coverage = (test_scores <= q_hat).mean()
            coverages.append(coverage)

        mean_cov = np.mean(coverages)
        # Should be >= 1-alpha (with tiny slack for finite samples)
        assert mean_cov >= (1 - alpha - 0.02), \
            f"Mean coverage {mean_cov:.3f} below nominal {1-alpha}"

    def test_quantile_monotone_in_alpha(self):
        """Larger alpha → smaller (tighter) threshold."""
        scores = np.random.default_rng(42).exponential(10, 500)
        q10 = conformal_quantile(scores, alpha=0.10)
        q20 = conformal_quantile(scores, alpha=0.20)
        assert q10 >= q20, f"Expected q(0.10) >= q(0.20), got {q10:.3f} < {q20:.3f}"

    def test_full_calibration_pipeline(self):
        rng = np.random.default_rng(7)
        scores = rng.exponential(5, 300)
        result = run_calibration(scores, alpha=0.10, delta=0.05, use_rcps=True)
        assert result.q_hat > 0
        assert result.lambda_star is not None
        assert result.lambda_star >= result.q_hat   # RCPS is more conservative


class TestRCPS:
    """RCPS should control risk at alpha with probability >= 1-delta."""

    def test_rcps_achieves_risk_control(self):
        rng = np.random.default_rng(99)
        alpha = 0.10
        delta = 0.05
        n     = 1000
        scores = rng.exponential(8, n)
        grid   = np.linspace(0, scores.max() * 1.5, 200)

        calibrator = RCPSCalibrator(alpha=alpha, delta=delta)
        lam_star = calibrator.fit(scores, grid)

        # Empirical risk at lambda_star should be <= alpha
        r_hat = (scores > lam_star).mean()
        assert r_hat <= alpha + 1e-6, \
            f"RCPS risk {r_hat:.4f} exceeds alpha {alpha}"

    def test_rcps_lambda_nonnegative(self):
        scores = np.array([0.1, 0.5, 1.0, 2.0, 5.0])
        grid   = np.linspace(0, 10, 100)
        lam    = RCPSCalibrator(alpha=0.5, delta=0.05).fit(scores, grid)
        assert lam >= 0


# ---------------------------------------------------------------------------
# Nonconformity score tests
# ---------------------------------------------------------------------------

class TestNonconformityScores:

    def _make_frame(self, H=64, W=64):
        rng = np.random.default_rng(0)
        gt   = rng.uniform(5, 50, (H, W)).astype(np.float32)
        lo   = gt - 3.0
        hi   = gt + 3.0
        mask = (gt > 10).astype(bool)
        return lo, hi, gt, mask

    def test_quantile_residual_score(self):
        lo, hi, gt, mask = self._make_frame()
        score_fn = QuantileResidualScore(pixel_quantile=0.9)
        s = score_fn.score(lo, hi, gt, mask)
        # Score is max(lo-y, y-hi): negative means GT is inside interval (good)
        assert isinstance(s, float)
        # GT is inside [lo=gt-3, hi=gt+3], so score should be negative
        assert s < 0, f"Expected negative score (GT inside interval), got {s}"

    def test_quantile_score_positive_when_outside(self):
        lo = np.full((10, 10), 20.0, dtype=np.float32)
        hi = np.full((10, 10), 30.0, dtype=np.float32)
        gt = np.full((10, 10), 35.0, dtype=np.float32)   # outside interval
        mask = np.ones((10, 10), dtype=bool)

        score_fn = QuantileResidualScore()
        s = score_fn.score(lo, hi, gt, mask)
        assert s > 0, "Score must be positive when GT is outside interval"

    def test_scale_aligned_score(self):
        lo, hi, gt, mask = self._make_frame()
        score_fn = ScaleAlignedScore()
        s = score_fn.score(lo, hi, gt, mask)
        assert isinstance(s, float)
        assert s >= 0

    def test_mask_restricted_score_with_empty_mask(self):
        lo = np.zeros((10, 10), dtype=np.float32)
        hi = np.zeros((10, 10), dtype=np.float32)
        gt = np.zeros((10, 10), dtype=np.float32)
        mask = np.zeros((10, 10), dtype=bool)   # all invalid

        score_fn = MaskRestrictedScore()
        s = score_fn.score(lo, hi, gt, mask)
        assert s == 0.0, "Empty mask should return 0 score"


# ---------------------------------------------------------------------------
# Size interval tests
# ---------------------------------------------------------------------------

class TestSizeInterval:

    def test_backproject_flat_disk(self):
        """A circular mask at constant depth should give a measurable 3D cloud."""
        H, W = 100, 100
        mask = np.zeros((H, W), dtype=bool)
        cv_center = (50, 50)
        radius = 20
        for r in range(H):
            for c in range(W):
                if (r - 50)**2 + (c - 50)**2 < radius**2:
                    mask[r, c] = True

        depth = np.full((H, W), 50.0, dtype=np.float32)   # 50mm constant
        pts = backproject_mask(mask, depth, fx=475, fy=475, cx=50, cy=50)
        assert pts.shape[1] == 3
        assert len(pts) == mask.sum()

    def test_principal_axis_diameter_known_case(self):
        """A line of points along X should have diameter = span."""
        pts = np.array([[0.0, 0.0, 50.0],
                        [10.0, 0.0, 50.0],
                        [20.0, 0.0, 50.0]])
        d = principal_axis_diameter(pts)
        assert abs(d - 20.0) < 1e-6, f"Expected 20mm, got {d}"

    def test_diameter_interval_monotone_in_depth(self):
        """Deeper depth → larger back-projected diameter for same mask."""
        H, W = 50, 50
        mask = np.zeros((H, W), dtype=bool)
        mask[20:30, 20:30] = True

        depth_near = np.full((H, W), 20.0, dtype=np.float32)
        depth_far  = np.full((H, W), 80.0, dtype=np.float32)
        K = np.array([475.0, 475.0, 25.0, 25.0])

        D_lo_near, D_hi_near = polyp_diameter_interval(mask, depth_near, depth_near, K)
        D_lo_far,  D_hi_far  = polyp_diameter_interval(mask, depth_far,  depth_far,  K)

        assert D_hi_far > D_hi_near, \
            f"Far depth should give larger diameter: {D_hi_far:.2f} <= {D_hi_near:.2f}"

    def test_classify_size(self):
        assert classify_size(3.0)  == "diminutive"
        assert classify_size(7.0)  == "small"
        assert classify_size(12.0) == "large"

    def test_diameter_interval_includes_gt(self):
        """With q_hat=0, the interval [D_lo, D_hi] derived from exact depth
        bounds should be well-defined."""
        H, W = 80, 80
        mask = np.zeros((H, W), dtype=bool)
        mask[30:50, 30:50] = True

        base_depth = np.full((H, W), 40.0, dtype=np.float32)
        lo  = base_depth * 0.9
        hi  = base_depth * 1.1
        K = np.array([475.0, 475.0, 40.0, 40.0])

        D_lo, D_hi = polyp_diameter_interval(mask, lo, hi, K, q_hat=0.0)
        D_gt, _ = polyp_diameter_interval(mask, base_depth, base_depth, K, q_hat=0.0)

        assert D_lo <= D_gt <= D_hi, \
            f"GT diameter {D_gt:.2f} not in [{D_lo:.2f}, {D_hi:.2f}]"


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
