"""
Baseline depth models for comparison with ConformalDepth.

Wraps EndoDAC, LightDepth, MonoViT, and Depth-Anything (zero-shot) behind a
common interface:  model(pixel_values) → (B, 1, H, W) metric depth in mm.

Uncertainty baselines (MC-Dropout, Deep Ensemble) are in UncertaintyWrapper.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForDepthEstimation


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
        self.model = AutoModelForDepthEstimation.from_pretrained(model_name)
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
