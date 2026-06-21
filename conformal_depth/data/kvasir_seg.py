"""
Kvasir-SEG / SUN-SEG Dataset Loader — polyp segmentation masks.

These datasets provide RGB frames + binary polyp masks.  They are used to
extract the polyp region (principal-axis bounding ellipse) for the
back-projected diameter interval computation.

Kvasir-SEG structure:
    <root>/
    ├── images/   *.jpg
    └── masks/    *.jpg (binary, same filename stem)

SUN-SEG structure (video clips):
    <root>/
    ├── <split>/
    │   ├── <case_id>/
    │   │   ├── Frame/     *.jpg
    │   │   └── GT/        *.png (binary mask)
    │   └── ...
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .transforms import build_transforms, apply_transforms


class KvasirSEGDataset(Dataset):
    """
    Kvasir-SEG with binary polyp masks.  Split is random 80/10/10.
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        transform_size: int = 518,
        seed: int = 42,
    ) -> None:
        self.root = Path(root)
        self.transform = build_transforms(split=split, size=transform_size)
        self.samples = self._index(seed, split)

    def _index(self, seed: int, split: str) -> list[dict]:
        img_dir = self.root / "images"
        msk_dir = self.root / "masks"
        all_imgs = sorted(img_dir.glob("*.jpg"))

        rng = np.random.default_rng(seed)
        all_imgs = list(rng.permutation(all_imgs))
        n = len(all_imgs)
        n_train, n_val = int(0.80 * n), int(0.10 * n)
        split_map = {
            "train": all_imgs[:n_train],
            "val":   all_imgs[n_train:n_train + n_val],
            "test":  all_imgs[n_train + n_val:],
        }

        samples = []
        for img_path in split_map.get(split, all_imgs):
            msk_path = msk_dir / img_path.name
            if msk_path.exists():
                samples.append({"img": str(img_path), "mask": str(msk_path)})
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]
        img = cv2.cvtColor(cv2.imread(s["img"]), cv2.COLOR_BGR2RGB)
        msk = cv2.imread(s["mask"], cv2.IMREAD_GRAYSCALE)

        # co-transform: pass mask as pseudo-depth
        H, W = img.shape[:2]
        mask_f = msk.astype(np.float32) / 255.0
        result = apply_transforms(self.transform, img, mask_f)

        # binarize after interpolation
        polyp_mask = (result["depth"] > 0.5).bool()

        return {
            "image":      result["image"],
            "polyp_mask": polyp_mask,          # (1, H, W) bool
            "frame_path": s["img"],
        }


class SUNSEGDataset(Dataset):
    """
    SUN-SEG video polyp dataset with per-frame GT masks.
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",   # "train" | "TestEasyDataset" | "TestHardDataset"
        transform_size: int = 518,
    ) -> None:
        self.root = Path(root)
        self.transform = build_transforms(split="val" if "Test" in split else split,
                                          size=transform_size)
        self.samples = self._index(split)

    def _index(self, split: str) -> list[dict]:
        split_dir = self.root / split
        if not split_dir.exists():
            raise FileNotFoundError(f"SUN-SEG split dir not found: {split_dir}")

        samples = []
        for case_dir in sorted(split_dir.iterdir()):
            if not case_dir.is_dir():
                continue
            frame_dir = case_dir / "Frame"
            gt_dir    = case_dir / "GT"
            if not frame_dir.exists():
                continue
            for fp in sorted(frame_dir.glob("*.jpg")):
                gp = gt_dir / fp.with_suffix(".png").name
                if gp.exists():
                    samples.append({
                        "img": str(fp), "mask": str(gp),
                        "case": case_dir.name,
                    })
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]
        img = cv2.cvtColor(cv2.imread(s["img"]), cv2.COLOR_BGR2RGB)
        msk = cv2.imread(s["mask"], cv2.IMREAD_GRAYSCALE)
        mask_f = (msk > 127).astype(np.float32)

        result = apply_transforms(self.transform, img, mask_f)
        polyp_mask = (result["depth"] > 0.5).bool()

        return {
            "image":      result["image"],
            "polyp_mask": polyp_mask,
            "frame_path": s["img"],
            "case":       s["case"],
        }
