"""
Smoke tests for baseline model wrappers.

These tests use mock/patched models to verify the wrapper logic
(shape handling, scale conversion, clipping) without requiring
network access or large checkpoint downloads.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import pytest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Minimal stub that mimics HuggingFace depth model output
# ---------------------------------------------------------------------------

class _FakeHFOutput:
    def __init__(self, depth: torch.Tensor):
        self.predicted_depth = depth   # (B, H, W) in metres


class _FakeHFModel(nn.Module):
    """Returns a constant depth map in metres for any input."""
    def __init__(self, depth_m: float = 0.05):   # 50mm = 0.05m
        super().__init__()
        self.depth_m = depth_m
        # No real parameters — nn.Module.parameters() will return empty iter

    def forward(self, pixel_values, **kwargs):
        B, _, H, W = pixel_values.shape
        # Match whatever spatial size was passed in (after wrapper's resize)
        depth = torch.full((B, H, W), self.depth_m)
        return _FakeHFOutput(depth)


# ---------------------------------------------------------------------------
# ZoeDepthWrapper tests
# ---------------------------------------------------------------------------

class TestZoeDepthWrapper:

    def _make_wrapper(self, depth_m: float = 0.04, max_depth_mm: float = 150.0):
        """Return a ZoeDepthWrapper with a fake inner model (no HF download)."""
        from conformal_depth.models.baselines import ZoeDepthWrapper

        # Patch the lazy loader (_load_hf_depth_model) — no module-level HF attr
        with patch(
            "conformal_depth.models.baselines._load_hf_depth_model",
            return_value=_FakeHFModel(depth_m=depth_m),
        ):
            wrapper = ZoeDepthWrapper(
                model_name="Intel/zoedepth-nk",
                depth_scale=1000.0,
                max_depth_mm=max_depth_mm,
            )
        return wrapper

    def test_output_shape(self):
        """Output must be (B, 1, H, W)."""
        wrapper = self._make_wrapper()
        imgs = torch.rand(2, 3, 256, 320)
        with torch.no_grad():
            out = wrapper(imgs)
        assert out.shape == (2, 1, 256, 320), f"Unexpected shape: {out.shape}"

    def test_metres_to_mm_conversion(self):
        """depth_scale=1000 should multiply metre output by 1000 → mm."""
        depth_m = 0.04          # 40 mm
        wrapper  = self._make_wrapper(depth_m=depth_m)
        imgs = torch.rand(1, 3, 128, 128)
        with torch.no_grad():
            out = wrapper(imgs)   # (1, 1, 128, 128)
        expected_mm = depth_m * 1000.0   # 40.0
        assert abs(out.mean().item() - expected_mm) < 0.5, (
            f"Expected ~{expected_mm:.1f}mm, got {out.mean().item():.3f}mm"
        )

    def test_clipping_to_max_depth(self):
        """Values above max_depth_mm must be clamped."""
        depth_m = 0.5           # 500 mm — well above endoscopy range
        max_mm  = 150.0
        wrapper = self._make_wrapper(depth_m=depth_m, max_depth_mm=max_mm)
        imgs = torch.rand(1, 3, 64, 64)
        with torch.no_grad():
            out = wrapper(imgs)
        assert out.max().item() <= max_mm + 1e-4, (
            f"Depth not clipped: max={out.max().item():.1f}mm, limit={max_mm}mm"
        )

    def test_no_negative_depth(self):
        """Output must be non-negative."""
        wrapper = self._make_wrapper(depth_m=0.02)
        imgs = torch.rand(2, 3, 64, 64)
        with torch.no_grad():
            out = wrapper(imgs)
        assert out.min().item() >= 0.0

    def test_spatial_upsample_to_input_size(self):
        """Output spatial dimensions must match the *input* image, not the
        ZoeDepth canonical resolution."""
        wrapper = self._make_wrapper()
        H, W = 480, 640
        imgs = torch.rand(1, 3, H, W)
        with torch.no_grad():
            out = wrapper(imgs)
        assert out.shape[-2:] == (H, W), (
            f"Output {out.shape[-2:]} does not match input ({H}, {W})"
        )

    def test_no_grad_through_model(self):
        """Model parameters must not accumulate gradients."""
        wrapper = self._make_wrapper()
        imgs = torch.rand(1, 3, 64, 64, requires_grad=True)
        with torch.no_grad():
            out = wrapper(imgs)
        assert not out.requires_grad


# ---------------------------------------------------------------------------
# DepthAnythingZeroShot sanity (also mocked)
# ---------------------------------------------------------------------------

class TestDepthAnythingZeroShot:

    def _make_wrapper(self, depth_val: float = 5.0):
        from conformal_depth.models.baselines import DepthAnythingZeroShot

        with patch(
            "conformal_depth.models.baselines._load_hf_depth_model",
            return_value=_FakeHFModel(depth_m=depth_val),
        ):
            wrapper = DepthAnythingZeroShot(
                model_name="depth-anything/Depth-Anything-V2-Large-hf"
            )
        return wrapper

    def test_output_shape(self):
        wrapper = self._make_wrapper()
        imgs = torch.rand(2, 3, 256, 256)
        with torch.no_grad():
            out = wrapper(imgs)
        assert out.shape == (2, 1, 256, 256)

    def test_output_nonnegative(self):
        wrapper = self._make_wrapper(depth_val=2.0)
        imgs = torch.rand(1, 3, 64, 64)
        with torch.no_grad():
            out = wrapper(imgs)
        assert out.min().item() >= 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
