
from __future__ import annotations

import json
import random
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from monai.transforms import (
    Compose,
    RandFlipd,
    RandRotate90d,
    RandRotated,
    RandAffined,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandAdjustContrastd,
)


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

class ACDCSliceDataset(Dataset):
    """
    Loads preprocessed 2-D slices (.npy) and optionally applies augmentation.

    Returned dict
        image    : FloatTensor (1, H, W)
        label    : LongTensor  (H, W)
        slice_id : str
    """

    def __init__(
        self,
        slice_ids:  List[str],
        images_dir: str | Path,
        labels_dir: str | Path,
        augment:    bool = False,
    ) -> None:
        self.slice_ids  = slice_ids
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.augment    = augment

        # ── Spatial transforms (applied to image + label together) ────────────
        self._spatial = Compose([
            RandFlipd(keys=["image", "label"], spatial_axis=0, prob=0.5),
            RandFlipd(keys=["image", "label"], spatial_axis=1, prob=0.5),
            RandRotate90d(keys=["image", "label"], prob=0.5),
            RandAffined(
                keys=["image", "label"],
                mode=["bilinear", "nearest"],
                prob=0.7,
                rotate_range=(0.26,),          # ±15°
                scale_range=(0.15,),           # ±15% zoom
                shear_range=(0.1,),
                translate_range=(8, 8),        # ±8 px
                padding_mode="zeros",
            ),
        ])

        # ── Intensity transforms (image only) ─────────────────────────────────
        self._intensity = Compose([
            RandGaussianNoised(keys=["image"], std=0.1,  prob=0.3),
            RandGaussianSmoothd(
                keys=["image"],
                sigma_x=(0.5, 1.5), sigma_y=(0.5, 1.5),
                prob=0.2,
            ),
            RandScaleIntensityd(keys=["image"],  factors=0.25, prob=0.4),
            RandShiftIntensityd(keys=["image"],  offsets=0.20, prob=0.4),
            RandAdjustContrastd(keys=["image"],  gamma=(0.7, 1.5), prob=0.3),
        ])

    def __len__(self) -> int:
        return len(self.slice_ids)

    def __getitem__(self, idx: int) -> Dict:
        sid = self.slice_ids[idx]

        image = np.load(self.images_dir / f"{sid}.npy")   # (H,W) float32
        label = np.load(self.labels_dir / f"{sid}.npy")   # (H,W) int64

        image = torch.from_numpy(image).unsqueeze(0)   # (1,H,W)
        label = torch.from_numpy(label).long()         # (H,W)

        sample = {"image": image, "label": label, "slice_id": sid}

        if self.augment:
            # Give label a temporary channel dim for spatial transforms
            sample["label"] = sample["label"].unsqueeze(0).float()  # (1,H,W)
            sample = self._spatial(sample)
            sample["label"] = sample["label"].squeeze(0).long()     # (H,W)

            # Intensity on image only (label unchanged)
            sample = self._intensity(sample)

        return sample


# ──────────────────────────────────────────────────────────────────────────────
# Pool manager
# ──────────────────────────────────────────────────────────────────────────────

class PoolManager:
    """Manages labeled/unlabeled slice-ID pools persisted as JSON."""

    def __init__(self, labeled_path: str | Path, unlabeled_path: str | Path) -> None:
        self.labeled_path   = Path(labeled_path)
        self.unlabeled_path = Path(unlabeled_path)
        self._labeled:   List[str] = []
        self._unlabeled: List[str] = []
        self._load()

    def _load(self) -> None:
        if self.labeled_path.is_file():
            with open(self.labeled_path) as f:
                self._labeled = json.load(f)
        if self.unlabeled_path.is_file():
            with open(self.unlabeled_path) as f:
                self._unlabeled = json.load(f)

    def save(self) -> None:
        import shutil
        self.labeled_path.parent.mkdir(parents=True, exist_ok=True)
        for p in (self.labeled_path, self.unlabeled_path):
            if p.is_dir():
                shutil.rmtree(p)
        with open(self.labeled_path,   "w") as f:
            json.dump(self._labeled, f, indent=2)
        with open(self.unlabeled_path, "w") as f:
            json.dump(self._unlabeled, f, indent=2)

    @staticmethod
    def _pid(sid: str) -> str:
        return sid.split("_frame")[0]

    @property
    def labeled(self)   -> List[str]: return list(self._labeled)
    @property
    def unlabeled(self) -> List[str]: return list(self._unlabeled)

    @property
    def num_labeled_patients(self) -> int:
        return len({self._pid(s) for s in self._labeled})

    def move_to_labeled(self, slice_ids: List[str]) -> None:
        id_set          = set(slice_ids)
        self._unlabeled = [s for s in self._unlabeled if s not in id_set]
        self._labeled   = list(dict.fromkeys(self._labeled + slice_ids))
        self.save()


# ──────────────────────────────────────────────────────────────────────────────
# DataLoader factories
# ──────────────────────────────────────────────────────────────────────────────

def make_dataloaders(
    labeled_ids:  List[str],
    images_dir:   str | Path,
    labels_dir:   str | Path,
    batch_size:   int   = 8,
    num_workers:  int   = 4,
    val_fraction: float = 0.2,
    seed:         int   = 42,
) -> Tuple[DataLoader, DataLoader]:
    """Patient-level train/val split → (train_loader, val_loader)."""
    rng = random.Random(seed)

    p2s: Dict[str, List[str]] = {}
    for sid in labeled_ids:
        p2s.setdefault(sid.split("_frame")[0], []).append(sid)

    patients = sorted(p2s.keys())
    rng.shuffle(patients)

    n_val      = max(1, int(len(patients) * val_fraction))
    val_pats   = patients[:n_val]
    train_pats = patients[n_val:]

    train_ids = [s for p in train_pats for s in p2s[p]]
    val_ids   = [s for p in val_pats   for s in p2s[p]]

    train_ds = ACDCSliceDataset(train_ids, images_dir, labels_dir, augment=True)
    val_ds   = ACDCSliceDataset(val_ids,   images_dir, labels_dir, augment=False)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader


def make_inference_loader(
    slice_ids:   List[str],
    images_dir:  str | Path,
    labels_dir:  str | Path,
    batch_size:  int = 8,
    num_workers: int = 4,
) -> DataLoader:
    """DataLoader for MC-Dropout inference (no augmentation)."""
    ds = ACDCSliceDataset(slice_ids, images_dir, labels_dir, augment=False)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
