"""
Shared image/depth transforms for all datasets.

Depth-Anything was trained on 518×518 images with ImageNet normalization.
We follow the same convention and expose a `build_transforms` factory.
"""

from __future__ import annotations

import numpy as np
import torch
import torchvision.transforms.functional as TF
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

DA_SIZE = 518   # Depth-Anything canonical training size


def build_transforms(
    split: str = "train",
    size: int = DA_SIZE,
    color_jitter: bool = True,
) -> A.Compose:
    """
    Args:
        split: "train" | "val" | "test"
        size: Resize shorter edge to this value; then centre-crop to square.
        color_jitter: Apply photometric distortion (recommended for colonoscopy).

    Returns:
        albumentations.Compose that accepts image + depth arrays and returns
        {"image": Tensor(3,H,W), "depth": Tensor(1,H,W)} both float32.
    """
    if split == "train":
        aug = [
            A.SmallestMaxSize(max_size=size + 32),
            A.RandomCrop(height=size, width=size),
            A.HorizontalFlip(p=0.5),
        ]
        if color_jitter:
            aug += [
                A.ColorJitter(
                    brightness=0.3, contrast=0.3,
                    saturation=0.2, hue=0.05, p=0.8,
                ),
                A.GaussianBlur(blur_limit=(3, 7), p=0.3),
                A.GaussNoise(p=0.2),
            ]
    else:
        aug = [
            A.SmallestMaxSize(max_size=size),
            A.CenterCrop(height=size, width=size),
        ]

    aug += [
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ]

    # albumentations needs additional_targets to co-transform depth
    return A.Compose(aug, additional_targets={"depth": "mask"})


def depth_to_tensor(depth_np: np.ndarray) -> torch.Tensor:
    """Convert (H, W) float32 numpy depth array to (1, H, W) tensor."""
    return torch.from_numpy(depth_np).unsqueeze(0).float()


def apply_transforms(
    transform: A.Compose,
    image: np.ndarray,   # (H, W, 3) uint8
    depth: np.ndarray,   # (H, W) float32
) -> dict:
    out = transform(image=image, depth=depth)
    img_tensor = out["image"]                       # (3, H, W) float32
    dep_tensor = out["depth"].unsqueeze(0).float()  # (1, H, W) float32
    return {"image": img_tensor, "depth": dep_tensor}
