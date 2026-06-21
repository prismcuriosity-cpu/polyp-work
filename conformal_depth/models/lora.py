"""
DV-LoRA: Depth-Video LoRA adapter for frozen DINOv2/Depth-Anything encoder.

Injects low-rank updates into the query/value projections of every attention
layer.  The adapter is applied to the transformer blocks of the ViT backbone
via PEFT's LoraConfig, which keeps the encoder frozen and trains only ~0.5%
of parameters.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model, TaskType


# Keys in DINOv2/ViT attention that we adapt
_DINO_TARGET_MODULES = ["query", "value"]
_DA_TARGET_MODULES = ["q_proj", "v_proj"]


def wrap_encoder_with_lora(
    encoder: nn.Module,
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: list[str] | None = None,
) -> nn.Module:
    """
    Freeze *encoder* and inject LoRA adapters into attention projections.

    Args:
        encoder: A HuggingFace-style ViT (e.g. Depth-Anything ViT-B/L backbone).
        r: LoRA rank.
        lora_alpha: Scaling factor (effective LR multiplier = lora_alpha / r).
        lora_dropout: Dropout on the low-rank path.
        target_modules: Which linear layer name-fragments to inject into.
            Defaults to DINOv2 convention ("query", "value").

    Returns:
        PEFT-wrapped encoder with all base weights frozen.
    """
    if target_modules is None:
        target_modules = _DINO_TARGET_MODULES

    config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
    )
    return get_peft_model(encoder, config)


class DVLoRAAdapter(nn.Module):
    """
    Thin wrapper that stores LoRA hyperparameters and exposes merge/unmerge
    for fast inference (avoids the residual add at every layer).
    """

    def __init__(
        self,
        encoder: nn.Module,
        r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        target_modules: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.encoder = wrap_encoder_with_lora(
            encoder, r=r, lora_alpha=lora_alpha,
            lora_dropout=lora_dropout, target_modules=target_modules,
        )
        self.r = r
        self.lora_alpha = lora_alpha

    def forward(self, pixel_values: torch.Tensor, **kwargs):
        return self.encoder(pixel_values=pixel_values, **kwargs)

    def merge_weights(self):
        """Merge LoRA weights into base weights for fast inference."""
        self.encoder.merge_adapter()

    def unmerge_weights(self):
        self.encoder.unmerge_adapter()

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]
