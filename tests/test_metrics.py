"""
Unit tests for evaluation metrics.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from conformal_depth.evaluation.metrics import (
    compute_depth_metrics,
    empirical_coverage,
    mean_interval_width,
    three_group_accuracy,
    concordance_correlation_coefficient,
    confusion_matrix_3group,
)


class TestDepthMetrics:

    def test_perfect_prediction(self):
        gt = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        m = compute_depth_metrics(gt, gt)
        assert m["abs_rel"] < 1e-5
        assert m["rmse"] < 1e-5
        assert abs(m["d1"] - 1.0) < 1e-5
        assert abs(m["d2"] - 1.0) < 1e-5
        assert abs(m["d3"] - 1.0) < 1e-5

    def test_scale_error(self):
        gt   = np.array([10.0, 20.0, 30.0])
        pred = gt * 2.0
        m = compute_depth_metrics(pred, gt)
        assert m["abs_rel"] > 0
        assert m["rmse"] > 0

    def test_all_keys_present(self):
        gt = np.random.default_rng(0).uniform(5, 50, 100)
        pred = gt + np.random.default_rng(1).normal(0, 2, 100)
        pred = np.clip(pred, 0.1, None)
        m = compute_depth_metrics(pred, gt)
        for key in ["abs_rel", "sq_rel", "rmse", "rmse_log", "silog", "d1", "d2", "d3"]:
            assert key in m


class TestCoverage:

    def test_full_coverage(self):
        gt = np.array([10.0, 20.0, 30.0])
        lo = gt - 5.0
        hi = gt + 5.0
        assert empirical_coverage(lo, hi, gt) == 1.0

    def test_zero_coverage(self):
        gt = np.array([10.0, 20.0, 30.0])
        lo = gt + 100
        hi = gt + 200
        assert empirical_coverage(lo, hi, gt) == 0.0

    def test_partial_coverage(self):
        gt = np.array([10.0, 20.0, 30.0, 40.0])
        lo = np.array([ 5.0, 15.0, 50.0, 50.0])
        hi = np.array([15.0, 25.0, 60.0, 60.0])
        cov = empirical_coverage(lo, hi, gt)
        assert abs(cov - 0.5) < 1e-6

    def test_interval_width(self):
        lo = np.array([0.0, 0.0])
        hi = np.array([4.0, 6.0])
        assert abs(mean_interval_width(lo, hi) - 5.0) < 1e-6


class TestSizingMetrics:

    def test_perfect_accuracy(self):
        gt   = np.array([3.0, 7.0, 12.0])    # diminutive, small, large
        pred = np.array([3.5, 6.5, 11.0])
        m = three_group_accuracy(pred, gt)
        assert m["accuracy"] == 1.0

    def test_ccc_perfect(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert abs(concordance_correlation_coefficient(x, x) - 1.0) < 1e-6

    def test_ccc_range(self):
        rng = np.random.default_rng(42)
        x = rng.uniform(0, 20, 100)
        y = rng.uniform(0, 20, 100)
        ccc = concordance_correlation_coefficient(x, y)
        assert -1.0 <= ccc <= 1.0

    def test_confusion_matrix_shape(self):
        gt   = np.array([3.0, 3.0, 7.0, 12.0])
        pred = np.array([3.0, 7.0, 7.0, 12.0])
        cm = confusion_matrix_3group(pred, gt)
        assert cm.shape == (3, 3)
        assert cm.sum() == len(gt)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
