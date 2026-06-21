"""
SimCol3D Dataset Loader — Rau et al., MICCAI 2023.

Synthetic colonoscopy dataset with perfect GT depth for coverage stress-tests.

Directory structure:

    <root>/
    ├── SyntheticColon_I/
    │   ├── Frames_S1/
    │   │   ├── FrameBuffer_0000.png   (RGB)
    │   │   ├── Depth_0000.png         (16-bit, depth in cm, scale=10)
    │   │   └── ...
    │   └── ...
    ├── SyntheticColon_II/
    │   └── ...
    └── ...

SimCol3D stores depth as 16-bit PNG where pixel value / 10 = depth in mm.
"""

from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .transforms import build_transforms, apply_transforms


# SimCol3D default intrinsics (from paper: 475×475 px FOV, square sensor)
SIMCOL_INTRINSICS = np.array([475.0, 475.0, 256.0, 256.0], dtype=np.float32)


class SimCol3DDataset(Dataset):
    """
    Args:
        root: Path to SimCol3D root.
        split: "train" | "val" | "test"  (random 70/15/15 split by sequence).
        depth_scale: Raw pixel → mm.  Default 0.1 (pixel_val / 10 = mm).
        max_depth_mm: Clip depth beyond this (endoscopy range).
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        depth_scale: float = 0.1,
        max_depth_mm: float = 150.0,
        transform_size: int = 518,
        seed: int = 42,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.depth_scale = depth_scale
        self.max_depth_mm = max_depth_mm
        self.transform = build_transforms(split=split, size=transform_size)
        self.samples = self._index(seed)

    def _index(self, seed: int) -> list[dict]:
        rng = np.random.default_rng(seed)
        all_seqs: list[Path] = []
        for colon_dir in sorted(self.root.iterdir()):
            if not colon_dir.is_dir() or not colon_dir.name.startswith("Synthetic"):
                continue
            for seq_dir in sorted(colon_dir.iterdir()):
                if seq_dir.is_dir():
                    all_seqs.append(seq_dir)

        rng.shuffle(all_seqs)
        n = len(all_seqs)
        n_train = int(0.70 * n)
        n_val   = int(0.15 * n)
        splits = {
            "train": all_seqs[:n_train],
            "val":   all_seqs[n_train:n_train + n_val],
            "test":  all_seqs[n_train + n_val:],
        }
        seqs = splits[self.split]

        samples = []
        for seq_dir in seqs:
            rgb_files = sorted(seq_dir.glob("FrameBuffer_*.png"))
            for rf in rgb_files:
                frame_num = re.search(r"(\d+)", rf.stem).group(1)
                df = seq_dir / f"Depth_{frame_num}.png"
                if df.exists():
                    samples.append({
                        "rgb_path":   str(rf),
                        "depth_path": str(df),
                        "seq_name":   seq_dir.name,
                    })
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]

        img = cv2.cvtColor(cv2.imread(s["rgb_path"]), cv2.COLOR_BGR2RGB)
        depth_raw = cv2.imread(s["depth_path"], cv2.IMREAD_ANYDEPTH).astype(np.float32)
        depth_mm = depth_raw * self.depth_scale
        valid = (depth_mm > 0) & (depth_mm < self.max_depth_mm)
        depth_mm[~valid] = 0.0

        result = apply_transforms(self.transform, img, depth_mm)
        dep = result["depth"]

        return {
            "image":      result["image"],
            "depth":      dep,
            "depth_mask": (dep > 0).bool(),
            "intrinsics": torch.from_numpy(SIMCOL_INTRINSICS),
            "seq_name":   s["seq_name"],
            "frame_idx":  idx,
        }


def build_simcol_dataloaders(
    root: str,
    batch_size: int = 8,
    num_workers: int = 4,
):
    loaders = {}
    for split in ("train", "val", "test"):
        ds = SimCol3DDataset(root=root, split=split)
        loaders[split] = DataLoader(
            ds, batch_size=batch_size, shuffle=(split == "train"),
            num_workers=num_workers, pin_memory=True, drop_last=(split == "train"),
        )
    return loaders["train"], loaders["val"], loaders["test"]
