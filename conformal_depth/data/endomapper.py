"""
EndoMapper Dataset Loader — Azagra et al., Sci. Data 2023.

Real in-vivo colonoscopy with calibrated camera (known intrinsics per patient).
Used as transfer evaluation (no GT depth — evaluation via scale-aligned metrics).

Directory structure:

    <root>/
    ├── seq_<id>/
    │   ├── color/        *.jpg
    │   ├── calib.yaml    (fx fy cx cy k1 k2 p1 p2)
    │   └── timestamps.txt
    └── ...
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from torch.utils.data import Dataset, DataLoader

from .transforms import build_transforms, apply_transforms


def _parse_calib(calib_path: str | Path) -> np.ndarray:
    """Parse calib.yaml → [fx, fy, cx, cy]."""
    with open(calib_path) as f:
        data = yaml.safe_load(f)
    # Handle both flat and nested YAML conventions used by EndoMapper
    if "camera_matrix" in data:
        K = np.array(data["camera_matrix"]["data"], dtype=np.float32).reshape(3, 3)
        return np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2]], dtype=np.float32)
    # Flat: fx fy cx cy
    return np.array([data["fx"], data["fy"], data["cx"], data["cy"]], dtype=np.float32)


class EndoMapperDataset(Dataset):
    """
    Unlabeled real-colonoscopy dataset — no GT depth.
    Used for out-of-distribution (transfer) evaluation.

    Returns images + intrinsics only; depth is None.
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "test",
        transform_size: int = 518,
        max_frames_per_seq: int | None = None,
        seed: int = 42,
    ) -> None:
        self.root = Path(root)
        self.transform = build_transforms(split=split, size=transform_size)
        self.max_per_seq = max_frames_per_seq
        self.samples = self._index(seed, split)

    def _index(self, seed: int, split: str) -> list[dict]:
        rng = np.random.default_rng(seed)
        seqs = sorted([d for d in self.root.iterdir() if d.is_dir()])
        rng.shuffle(seqs)
        n = len(seqs)
        n_train, n_val = int(0.6 * n), int(0.2 * n)
        split_map = {
            "train": seqs[:n_train],
            "val":   seqs[n_train:n_train + n_val],
            "test":  seqs[n_train + n_val:],
        }
        seqs = split_map.get(split, seqs)

        samples = []
        for seq in seqs:
            color_dir = seq / "color"
            if not color_dir.exists():
                continue
            calib_path = seq / "calib.yaml"
            if not calib_path.exists():
                continue

            intrinsics = _parse_calib(calib_path)
            frames = sorted(color_dir.glob("*.jpg"))
            if self.max_per_seq:
                frames = frames[:self.max_per_seq]

            for f in frames:
                samples.append({
                    "frame_path": str(f),
                    "intrinsics": intrinsics,
                    "seq_name": seq.name,
                })
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]
        img = cv2.cvtColor(cv2.imread(s["frame_path"]), cv2.COLOR_BGR2RGB)
        H, W = img.shape[:2]

        # Dummy depth for transform compatibility
        dummy_depth = np.zeros((H, W), dtype=np.float32)
        result = apply_transforms(self.transform, img, dummy_depth)

        return {
            "image":      result["image"],
            "depth":      None,
            "intrinsics": torch.from_numpy(s["intrinsics"]),
            "seq_name":   s["seq_name"],
            "frame_idx":  idx,
        }
