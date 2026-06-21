"""
EndoSLAM Dataset Loader — Ozyoruk et al., MedIA 2021.

Ex-vivo dataset with porcine colon/small intestine/stomach + polyp-mimicking
elevations.  Provides RGB frames and GT depth from structured-light scanner.

Directory structure:

    <root>/
    ├── Colon/
    │   ├── Frames/
    │   │   ├── 0.png, 1.png, ...
    │   ├── Depth/
    │   │   ├── 0.png, 1.png, ...     (16-bit, mm)
    │   └── Poses/
    │       └── groundtruth.txt        (tx ty tz qx qy qz qw)
    ├── SmallIntestine/
    └── Stomach/

Intrinsics are from the published calibration (see paper Table 1).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .transforms import build_transforms, apply_transforms


# Published calibration per organ type (approximate — override per sequence)
INTRINSICS_BY_ORGAN = {
    "Colon":          np.array([896.0, 896.0, 512.0, 384.0], dtype=np.float32),
    "SmallIntestine": np.array([796.0, 796.0, 512.0, 384.0], dtype=np.float32),
    "Stomach":        np.array([912.0, 912.0, 512.0, 384.0], dtype=np.float32),
}

DEFAULT_INTRINSICS = np.array([896.0, 896.0, 512.0, 384.0], dtype=np.float32)


class EndoSLAMDataset(Dataset):
    """
    Args:
        root: EndoSLAM root directory.
        organs: List of organ subdirectories to include.
        split: "train" | "val" | "test" (split by sequence index).
    """

    def __init__(
        self,
        root: str | Path,
        organs: list[str] | None = None,
        split: str = "train",
        max_depth_mm: float = 150.0,
        transform_size: int = 518,
        seed: int = 42,
    ) -> None:
        self.root = Path(root)
        self.organs = organs or ["Colon", "SmallIntestine", "Stomach"]
        self.split = split
        self.max_depth_mm = max_depth_mm
        self.transform = build_transforms(split=split, size=transform_size)
        self.samples = self._index(seed)

    def _index(self, seed: int) -> list[dict]:
        rng = np.random.default_rng(seed)
        all_seqs = []

        for organ in self.organs:
            organ_dir = self.root / organ
            if not organ_dir.exists():
                continue

            # Each organ may have multiple sub-sequences (subdirs or flat)
            frame_dir = organ_dir / "Frames"
            depth_dir = organ_dir / "Depth"
            if frame_dir.exists() and depth_dir.exists():
                all_seqs.append({
                    "frame_dir": frame_dir,
                    "depth_dir": depth_dir,
                    "organ": organ,
                    "pose_path": str(organ_dir / "Poses" / "groundtruth.txt"),
                    "intrinsics": INTRINSICS_BY_ORGAN.get(organ, DEFAULT_INTRINSICS),
                })
            else:
                # Multiple sub-sequences
                for sub in sorted(organ_dir.iterdir()):
                    if sub.is_dir():
                        fd = sub / "Frames"
                        dd = sub / "Depth"
                        if fd.exists() and dd.exists():
                            all_seqs.append({
                                "frame_dir": fd,
                                "depth_dir": dd,
                                "organ": organ,
                                "pose_path": str(sub / "Poses" / "groundtruth.txt"),
                                "intrinsics": INTRINSICS_BY_ORGAN.get(organ, DEFAULT_INTRINSICS),
                            })

        rng.shuffle(all_seqs)
        n = len(all_seqs)
        n_train, n_val = int(0.70 * n), int(0.15 * n)
        splits = {
            "train": all_seqs[:n_train],
            "val":   all_seqs[n_train:n_train + n_val],
            "test":  all_seqs[n_train + n_val:],
        }
        seqs = splits.get(self.split, all_seqs)

        samples = []
        for seq in seqs:
            frames = sorted(Path(seq["frame_dir"]).glob("*.png"),
                            key=lambda p: int(p.stem))
            for frame_path in frames:
                dep_path = Path(seq["depth_dir"]) / frame_path.name
                if dep_path.exists():
                    samples.append({
                        **seq,
                        "frame_path": str(frame_path),
                        "depth_path": str(dep_path),
                    })
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]

        img = cv2.cvtColor(cv2.imread(s["frame_path"]), cv2.COLOR_BGR2RGB)
        depth_raw = cv2.imread(s["depth_path"], cv2.IMREAD_ANYDEPTH).astype(np.float32)
        # EndoSLAM depth is stored in mm directly as 16-bit PNG
        depth_mm = depth_raw.astype(np.float32)
        valid = (depth_mm > 0) & (depth_mm < self.max_depth_mm)
        depth_mm[~valid] = 0.0

        result = apply_transforms(self.transform, img, depth_mm)
        dep = result["depth"]

        return {
            "image":      result["image"],
            "depth":      dep,
            "depth_mask": (dep > 0).bool(),
            "intrinsics": torch.from_numpy(s["intrinsics"]),
            "seq_name":   f"{s['organ']}_{Path(s['frame_path']).parent.parent.name}",
            "frame_idx":  idx,
        }


def build_endoslam_loader(
    root: str,
    split: str = "test",
    batch_size: int = 8,
    num_workers: int = 4,
) -> DataLoader:
    ds = EndoSLAMDataset(root=root, split=split)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
