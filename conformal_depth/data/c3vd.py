"""
C3VD Dataset Loader — Bobrow et al., Med. Image Anal. 90:102956, 2023.
C3VDv2 extension — Golhar et al., arXiv:2506.24074, Sci. Data 2026.

Directory structure expected on disk (set C3VD_ROOT env var or pass root=):

    <root>/
    ├── sequences/
    │   ├── <seq_name>/
    │   │   ├── color/          *.png   (1080×1350 RGB)
    │   │   ├── depth/          *.exr   (metric depth in mm, float32)
    │   │   ├── normals/        *.exr
    │   │   ├── occlusion/      *.exr
    │   │   ├── opticalflow/    *.flo
    │   │   ├── pose.txt        (6-DoF per-frame: tx ty tz qx qy qz qw)
    │   │   └── intrinsics.txt  (fx fy cx cy)
    │   └── ...
    ├── splits/
    │   ├── train.txt
    │   ├── val.txt
    │   └── test.txt            <- calibration split used for conformal
    └── c3vdv2/                 <- C3VDv2 sequences (same structure, optional)

C3VDv2 adds 169,371 frames across 60 phantom segments; enable with use_v2=True.

Returns per-frame dict:
    image         : Tensor (3, H, W)  float32, ImageNet-normalized
    depth         : Tensor (1, H, W)  float32, metric mm
    depth_mask    : Tensor (1, H, W)  bool, valid depth pixels
    intrinsics    : Tensor (4,)        [fx, fy, cx, cy] in pixels
    pose          : Tensor (4, 4)      camera-to-world SE3 matrix
    seq_name      : str
    frame_idx     : int
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .transforms import build_transforms, apply_transforms


def _load_exr_depth(path: str | Path) -> np.ndarray:
    """Read a 32-bit EXR depth image → (H, W) float32 array in metres."""
    img = cv2.imread(str(path), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read depth EXR: {path}")
    if img.ndim == 3:
        img = img[..., 0]
    return img.astype(np.float32)


def _load_intrinsics(path: str | Path) -> np.ndarray:
    """Parse intrinsics.txt → [fx, fy, cx, cy]."""
    vals = np.loadtxt(str(path), dtype=np.float32)
    if vals.size >= 4:
        return vals[:4]
    raise ValueError(f"Unexpected intrinsics format in {path}")


def _load_pose(path: str | Path, frame_idx: int) -> np.ndarray:
    """
    Parse pose.txt (one line per frame: tx ty tz qx qy qz qw) and return
    the 4×4 camera-to-world transform for *frame_idx*.
    """
    data = np.loadtxt(str(path), dtype=np.float64)
    if data.ndim == 1:
        data = data[None, :]   # single-frame sequence
    row = data[frame_idx]
    t = row[:3]
    q = row[3:]   # (qx, qy, qz, qw)
    from scipy.spatial.transform import Rotation
    R = Rotation.from_quat(q).as_matrix()
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R
    T[:3, 3]  = t
    return T


class C3VDDataset(Dataset):
    """
    Loads individual frames from C3VD (+ optionally C3VDv2) with GT depth.

    Args:
        root: Dataset root directory.
        split: "train" | "val" | "test"
        use_v2: Include C3VDv2 sequences.
        depth_scale: Multiply raw EXR values by this to get mm.
            C3VD stores depth in metres; set depth_scale=1000 to get mm.
        max_depth_mm: Clip depths beyond this value (mark invalid).
        transform_size: Resize to this square resolution before returning.
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        use_v2: bool = True,
        depth_scale: float = 1000.0,     # metres → mm
        max_depth_mm: float = 150.0,     # endoscope working range
        transform_size: int = 518,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.depth_scale = depth_scale
        self.max_depth_mm = max_depth_mm
        self.transform = build_transforms(split=split, size=transform_size)

        self.samples = self._index_samples(use_v2)

    def _index_samples(self, use_v2: bool) -> list[dict]:
        samples = []

        split_dirs = [self.root / "sequences"]
        if use_v2 and (self.root / "c3vdv2").exists():
            split_dirs.append(self.root / "c3vdv2")

        split_file = self.root / "splits" / f"{self.split}.txt"
        if split_file.exists():
            allowed_seqs = set(split_file.read_text().split())
        else:
            allowed_seqs = None    # use all sequences

        for seq_dir_root in split_dirs:
            if not seq_dir_root.exists():
                continue
            for seq_path in sorted(seq_dir_root.iterdir()):
                if not seq_path.is_dir():
                    continue
                if allowed_seqs and seq_path.name not in allowed_seqs:
                    continue

                color_dir = seq_path / "color"
                depth_dir = seq_path / "depth"
                if not color_dir.exists() or not depth_dir.exists():
                    continue

                color_files = sorted(color_dir.glob("*.png"))
                for frame_idx, cf in enumerate(color_files):
                    stem = cf.stem
                    df = depth_dir / f"{stem}.exr"
                    if not df.exists():
                        continue
                    samples.append({
                        "color_path": str(cf),
                        "depth_path": str(df),
                        "intrinsics_path": str(seq_path / "intrinsics.txt"),
                        "pose_path": str(seq_path / "pose.txt"),
                        "seq_name": seq_path.name,
                        "frame_idx": frame_idx,
                    })

        if not samples:
            raise RuntimeError(
                f"C3VD: no samples found in {self.root} for split='{self.split}'. "
                "Check your directory structure and split files."
            )
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]

        # --- image ---
        img_bgr = cv2.imread(s["color_path"], cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise FileNotFoundError(s["color_path"])
        image_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # --- depth ---
        depth_m = _load_exr_depth(s["depth_path"])
        depth_mm = depth_m * self.depth_scale
        valid = (depth_mm > 0) & (depth_mm < self.max_depth_mm)
        depth_mm[~valid] = 0.0

        # --- apply transforms ---
        result = apply_transforms(self.transform, image_rgb, depth_mm)

        # --- intrinsics ---
        K = _load_intrinsics(s["intrinsics_path"])  # [fx, fy, cx, cy]

        # --- pose ---
        pose = _load_pose(s["pose_path"], s["frame_idx"])

        # depth_mask mirrors the valid map after spatial transforms
        dep = result["depth"]
        depth_mask = (dep > 0).bool()

        return {
            "image":       result["image"],               # (3, H, W)
            "depth":       dep,                           # (1, H, W) mm
            "depth_mask":  depth_mask,                    # (1, H, W) bool
            "intrinsics":  torch.from_numpy(K),           # (4,)
            "pose":        torch.from_numpy(pose),        # (4, 4)
            "seq_name":    s["seq_name"],
            "frame_idx":   s["frame_idx"],
        }


def build_c3vd_dataloaders(
    root: str,
    batch_size: int = 8,
    num_workers: int = 4,
    use_v2: bool = True,
    pin_memory: bool = True,
):
    """
    Convenience factory — returns (train_loader, val_loader, calib_loader).

    The calibration split (test.txt) is used for conformal calibration.
    """
    from torch.utils.data import DataLoader

    loaders = {}
    for split in ("train", "val", "test"):
        ds = C3VDDataset(root=root, split=split, use_v2=use_v2)
        shuffle = (split == "train")
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=(split == "train"),
        )
    return loaders["train"], loaders["val"], loaders["test"]
