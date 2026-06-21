"""
ConformalDepth model: Depth-Anything ViT + DV-LoRA + two-quantile dense head.

Architecture
------------
1. Frozen Depth-Anything ViT-B/L encoder (loaded from HuggingFace).
2. DV-LoRA adapter injected into Q/V projections of every transformer block.
3. DPT-style decoder that produces three outputs at the same spatial resolution
   as the input:
       - depth_median  (τ = 0.5)  – point estimate
       - depth_lo      (τ = α/2)  – lower quantile
       - depth_hi      (τ = 1-α/2) – upper quantile

Scale alignment (metric anchor) is done externally via LightDepth-style
illumination-decline self-supervision (see training/losses.py).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForDepthEstimation, AutoConfig

from .lora import DVLoRAAdapter


# ---------------------------------------------------------------------------
# DPT-style dense head (produces H×W output per quantile channel)
# ---------------------------------------------------------------------------

class DPTQuantileHead(nn.Module):
    """
    Lightweight feature-fusion head adapted from DPT that outputs *n_quantiles*
    depth maps at the input image resolution.

    We re-use the neck features from the Depth-Anything backbone (4 stage
    embeddings are concatenated in a feature pyramid) and project them to the
    desired number of output channels.
    """

    def __init__(
        self,
        in_channels: int = 256,          # DPT feature-fusion output width
        hidden_channels: int = 128,
        n_quantiles: int = 3,            # lo, median, hi
    ) -> None:
        super().__init__()

        self.n_quantiles = n_quantiles

        self.head = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels // 2, n_quantiles, kernel_size=1),
            nn.Softplus(),               # depth > 0
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, in_channels, H, W) fused DPT features.
        Returns:
            (B, n_quantiles, H, W) depth maps in arbitrary (scale-relative) units.
        """
        return self.head(features)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class ConformalDepthModel(nn.Module):
    """
    Full ConformalDepth model.

    Usage::

        model = ConformalDepthModel.from_pretrained("depth-anything/Depth-Anything-V2-Large-hf")
        # outputs: (B, 3, H, W) — channels are [lo, median, hi]
        out = model(pixel_values=imgs)
        depth_lo, depth_med, depth_hi = out[:, 0], out[:, 1], out[:, 2]
    """

    DA_FEATURE_DIM = {
        "depth-anything/Depth-Anything-V2-Small-hf": 128,
        "depth-anything/Depth-Anything-V2-Base-hf":  256,
        "depth-anything/Depth-Anything-V2-Large-hf": 256,
    }

    def __init__(
        self,
        backbone: nn.Module,
        feature_dim: int = 256,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        n_quantiles: int = 3,
        target_modules: list[str] | None = None,
    ) -> None:
        super().__init__()

        # Wrap the backbone's ViT encoder with LoRA; leave the DPT neck frozen
        encoder = backbone.backbone        # the ViT part
        self.lora_encoder = DVLoRAAdapter(
            encoder,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
        )

        # DPT neck (feature-fusion pyramid) — frozen
        self.neck = backbone.neck
        for p in self.neck.parameters():
            p.requires_grad_(False)

        # Replace the single-channel DA head with our quantile head
        self.quantile_head = DPTQuantileHead(
            in_channels=feature_dim,
            n_quantiles=n_quantiles,
        )

        # Scale factor (learned affine per-sample, initialized to identity)
        # Multiplied after softplus so final depth = scale * raw_depth + shift
        self.register_parameter("log_scale", nn.Parameter(torch.zeros(1)))
        self.register_parameter("shift", nn.Parameter(torch.zeros(1)))

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        model_name: str = "depth-anything/Depth-Anything-V2-Large-hf",
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        n_quantiles: int = 3,
        **kwargs,
    ) -> "ConformalDepthModel":
        """Load Depth-Anything from HuggingFace and wrap with LoRA + quantile head."""
        backbone = AutoModelForDepthEstimation.from_pretrained(model_name)
        feature_dim = cls.DA_FEATURE_DIM.get(model_name, 256)
        return cls(
            backbone=backbone,
            feature_dim=feature_dim,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            n_quantiles=n_quantiles,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        pixel_values: torch.Tensor,
        output_size: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        """
        Args:
            pixel_values: (B, 3, H, W) normalized images.
            output_size: (H_out, W_out) to upsample to; defaults to input size.

        Returns:
            (B, 3, H_out, W_out)  channels = [depth_lo, depth_median, depth_hi]
        """
        B, _, H, W = pixel_values.shape
        if output_size is None:
            output_size = (H, W)

        # LoRA-adapted encoder
        enc_out = self.lora_encoder(
            pixel_values=pixel_values,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = enc_out.hidden_states  # tuple of (B, T, C)

        # DPT neck: expects hidden states as list of (B, C, h, w) patch grids
        patch_size = 14  # DINOv2 patch size
        h_patch = H // patch_size
        w_patch = W // patch_size

        # Reshape tokens → spatial (drop [CLS] token at index 0)
        spatial = [
            hs[:, 1:].reshape(B, h_patch, w_patch, -1).permute(0, 3, 1, 2)
            for hs in hidden_states[1:]   # skip embedding layer output
        ]

        # DPT neck fuses 4 stages
        neck_features = self.neck(spatial[-4:])  # (B, feature_dim, H/4, W/4)

        # Quantile head → (B, 3, H/4, W/4)
        raw = self.quantile_head(neck_features)

        # Affine scale alignment (metric anchor)
        scale = self.log_scale.exp()
        depth_maps = scale * raw + self.shift

        # Upsample to original resolution
        depth_maps = F.interpolate(
            depth_maps, size=output_size, mode="bilinear", align_corners=False
        )

        return depth_maps   # [lo, median, hi]

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def point_estimate(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Returns only the median depth map (B, 1, H, W)."""
        return self.forward(pixel_values)[:, 1:2]

    def interval(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (depth_lo, depth_hi) each (B, 1, H, W)."""
        out = self.forward(pixel_values)
        return out[:, 0:1], out[:, 2:3]

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]
