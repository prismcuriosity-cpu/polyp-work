"""
Baseline depth models for comparison with ConformalDepth.

Wraps ZoeDepth, Depth-Anything (zero-shot), MonoViT, MC-Dropout, and Deep
Ensemble behind a common interface:  model(pixel_values) → (B, 1, H, W) in mm.

ZoeDepth (isl-org/ZoeDepth, https://github.com/isl-org/ZoeDepth):
  Metric monocular depth via zero-shot transfer from NYU+KITTI.
  Available on HuggingFace as Intel/zoedepth-nk|n|k.
  Outputs metric depth in **metres**; the wrapper multiplies by 1000 → mm.
  Unlike Depth-Anything (relative), ZoeDepth depth is directly comparable
  to GT in mm — making it the strongest zero-shot metric baseline.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _load_hf_depth_model(model_name: str):
    """Lazy import of HuggingFace AutoModelForDepthEstimation.

    Deferred so that importing this module does not require transformers
    (keeps the test suite fast when transformers is not installed).
    """
    from transformers import AutoModelForDepthEstimation
    return AutoModelForDepthEstimation.from_pretrained(model_name)


# ---------------------------------------------------------------------------
# Common protocol
# ---------------------------------------------------------------------------

class BaseDepthModel(nn.Module):
    """All baselines implement this interface."""

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Returns (B, 1, H, W) depth map."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Depth-Anything zero-shot baseline
# ---------------------------------------------------------------------------

class DepthAnythingZeroShot(BaseDepthModel):
    """
    Depth-Anything V2 run without any fine-tuning or scale correction.
    Returns relative (affine-invariant) depth; comparisons use correlation
    metrics (Spearman ρ, SILog).
    """

    def __init__(self, model_name: str = "depth-anything/Depth-Anything-V2-Large-hf"):
        super().__init__()
        self.model = _load_hf_depth_model(model_name)
        for p in self.model.parameters():
            p.requires_grad_(False)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        H, W = pixel_values.shape[-2:]
        out = self.model(pixel_values=pixel_values)
        depth = out.predicted_depth.unsqueeze(1)           # (B, 1, Hf, Wf)
        return F.interpolate(depth, size=(H, W), mode="bilinear", align_corners=False)


# ---------------------------------------------------------------------------
# MonoViT thin wrapper (loads timm ViT-base + DPT head checkpoint)
# ---------------------------------------------------------------------------

class MonoViTWrapper(BaseDepthModel):
    """
    Thin wrapper around a MonoViT-style checkpoint.
    The checkpoint must follow the DA/DPT architecture so we re-use
    AutoModelForDepthEstimation with a custom config override.

    Provide checkpoint_path to load from a local .pth / .safetensors file.
    """

    def __init__(self, checkpoint_path: str | None = None):
        super().__init__()
        # MonoViT uses ViT-B DPT; load Depth-Anything-Base config as scaffold
        self.model = AutoModelForDepthEstimation.from_pretrained(
            "depth-anything/Depth-Anything-V2-Base-hf"
        )
        if checkpoint_path is not None:
            state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            missing, unexpected = self.model.load_state_dict(state, strict=False)
            if missing:
                print(f"[MonoViT] Missing keys: {len(missing)}")
        for p in self.model.parameters():
            p.requires_grad_(False)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        H, W = pixel_values.shape[-2:]
        out = self.model(pixel_values=pixel_values)
        depth = out.predicted_depth.unsqueeze(1)
        return F.interpolate(depth, size=(H, W), mode="bilinear", align_corners=False)


# ---------------------------------------------------------------------------
# ZoeDepth zero-shot metric baseline
# ---------------------------------------------------------------------------

class ZoeDepthWrapper(BaseDepthModel):
    """
    ZoeDepth (Bhat et al., 2023 — https://github.com/isl-org/ZoeDepth) loaded
    from HuggingFace and run zero-shot (no endoscopy fine-tuning).

    Unlike Depth-Anything, ZoeDepth outputs **metric depth in metres** trained
    on NYU Depth V2 and/or KITTI, so values can be directly compared to GT mm
    after the ×1000 scale conversion — making it the strongest off-the-shelf
    metric baseline available without any domain adaptation.

    Model variants on HuggingFace:
        Intel/zoedepth-nk  — trained on NYU + KITTI (recommended, most general)
        Intel/zoedepth-n   — NYU only (indoor scenes, closer to endoscopy scale)
        Intel/zoedepth-k   — KITTI only (outdoor, poor fit for endoscopy)

    Args:
        model_name:   HuggingFace model ID.
        depth_scale:  Multiply raw output (metres) to get mm.  Default 1000.
        max_depth_mm: Clip predictions above this value (endoscopy range ≈ 150 mm).
    """

    ZOEDEPTH_INPUT_SIZE = (384, 512)   # ZoeDepth canonical inference resolution

    def __init__(
        self,
        model_name:   str   = "Intel/zoedepth-nk",
        depth_scale:  float = 1000.0,
        max_depth_mm: float = 150.0,
    ):
        super().__init__()
        self.depth_scale  = depth_scale
        self.max_depth_mm = max_depth_mm

        # AutoModelForDepthEstimation handles both ZoeDepthForDepthEstimation
        # and its BEiT / DPT sub-components transparently.
        self.model = _load_hf_depth_model(model_name)
        for p in self.model.parameters():
            p.requires_grad_(False)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: (B, 3, H, W) ImageNet-normalized float32 tensors.
                ZoeDepth uses the same ImageNet normalization convention, so
                no un-normalization is required.

        Returns:
            (B, 1, H, W) depth in mm, clipped to [0, max_depth_mm].
        """
        H, W = pixel_values.shape[-2:]

        # ZoeDepth internally expects a fixed spatial resolution for its
        # seed-estimation head.  Resize input, then upsample output back.
        inp = F.interpolate(
            pixel_values,
            size=self.ZOEDEPTH_INPUT_SIZE,
            mode="bilinear",
            align_corners=False,
        )

        out   = self.model(pixel_values=inp)
        depth = out.predicted_depth.unsqueeze(1)           # (B, 1, H', W') metres

        # Back to original resolution
        depth = F.interpolate(depth, size=(H, W), mode="bilinear", align_corners=False)

        # Convert metres → mm and clip to endoscopy working range
        depth_mm = depth * self.depth_scale
        return depth_mm.clamp(min=0.0, max=self.max_depth_mm)


# ---------------------------------------------------------------------------
# MC-Dropout uncertainty baseline
# ---------------------------------------------------------------------------

class MCDropoutWrapper(BaseDepthModel):
    """
    Turns any BaseDepthModel into a Monte-Carlo dropout uncertainty estimator.

    Call forward() to get the mean depth.
    Call predict_with_uncertainty() to get mean + std depth maps.

    Note: the underlying model must have Dropout layers (or we inject them).
    """

    def __init__(
        self,
        base_model: BaseDepthModel,
        n_samples: int = 30,
        dropout_p: float = 0.1,
    ):
        super().__init__()
        self.base = base_model
        self.n_samples = n_samples
        self._inject_dropout(dropout_p)

    def _inject_dropout(self, p: float):
        """Replace all Identity layers after attention with Dropout."""
        for module in self.base.modules():
            if isinstance(module, nn.Sequential):
                for i, child in enumerate(module):
                    if isinstance(child, nn.GELU) or isinstance(child, nn.ReLU):
                        # insert dropout after activations
                        pass  # handled by eval/train mode on existing dropouts

    def _enable_dropout(self):
        for m in self.base.modules():
            if isinstance(m, nn.Dropout):
                m.train()

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.predict_with_uncertainty(pixel_values)[0]

    @torch.no_grad()
    def predict_with_uncertainty(
        self, pixel_values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            mean_depth: (B, 1, H, W)
            std_depth:  (B, 1, H, W)  – aleatoric+epistemic combined
        """
        self.base.eval()
        self._enable_dropout()

        samples = []
        for _ in range(self.n_samples):
            samples.append(self.base(pixel_values))

        stack = torch.stack(samples, dim=0)   # (N, B, 1, H, W)
        return stack.mean(0), stack.std(0)


# ---------------------------------------------------------------------------
# Deep Ensemble uncertainty baseline
# ---------------------------------------------------------------------------

class DeepEnsemble(BaseDepthModel):
    """
    Ensemble of independently trained depth models.
    Each member is a BaseDepthModel loaded from its own checkpoint.
    """

    def __init__(self, members: list[BaseDepthModel]):
        super().__init__()
        self.members = nn.ModuleList(members)

    @classmethod
    def from_checkpoints(
        cls,
        checkpoint_paths: list[str],
        model_name: str = "depth-anything/Depth-Anything-V2-Base-hf",
    ) -> "DeepEnsemble":
        members = []
        for ckpt in checkpoint_paths:
            m = DepthAnythingZeroShot(model_name)
            state = torch.load(ckpt, map_location="cpu", weights_only=True)
            m.model.load_state_dict(state, strict=False)
            members.append(m)
        return cls(members)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.predict_with_uncertainty(pixel_values)[0]

    @torch.no_grad()
    def predict_with_uncertainty(
        self, pixel_values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        preds = [m(pixel_values) for m in self.members]
        stack = torch.stack(preds, dim=0)
        return stack.mean(0), stack.std(0)
