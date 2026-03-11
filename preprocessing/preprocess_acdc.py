from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np
from monai.transforms import ResizeWithPadOrCrop

from visualization.plots import save_preprocessing_figure


# ──────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_h5_slice(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load image and label arrays from an ACDC .h5 slice file.

    Returns
    -------
    image : float32 (H, W)
    label : int64   (H, W)
    """
    with h5py.File(path, "r") as f:
        keys = list(f.keys())
        img_key = "image" if "image" in f else keys[0]
        lbl_key = "label" if "label" in f else keys[1]
        image = f[img_key][()].astype(np.float32)
        label = f[lbl_key][()].astype(np.int64)
    return image, label


# ──────────────────────────────────────────────────────────────────────────────
# Transforms
# ──────────────────────────────────────────────────────────────────────────────

def normalize_intensity(image: np.ndarray) -> np.ndarray:
    """Z-score normalise non-zero voxels; leave zero background unchanged."""
    mask = image > 0
    if mask.sum() == 0:
        return image
    mean = float(image[mask].mean())
    std  = float(image[mask].std()) + 1e-8
    out  = image.copy()
    out[mask] = (image[mask] - mean) / std
    return out


def resize_slice(
    image: np.ndarray,
    label: np.ndarray,
    target: Tuple[int, int] = (160, 160),
) -> Tuple[np.ndarray, np.ndarray]:
    """Centre-crop or zero-pad a 2-D slice to *target* using MONAI."""
    import torch
    pad_crop = ResizeWithPadOrCrop(spatial_size=target)

    img_t = torch.from_numpy(image[np.newaxis])                      # (1, H, W)
    lbl_t = torch.from_numpy(label[np.newaxis].astype(np.float32))   # (1, H, W)

    img_r = pad_crop(img_t).numpy()[0]
    lbl_r = pad_crop(lbl_t).numpy()[0]
    return img_r.astype(np.float32), lbl_r.astype(np.int64)


# ──────────────────────────────────────────────────────────────────────────────
# Slice-level processing
# ──────────────────────────────────────────────────────────────────────────────

def process_slice_file(
    h5_path:      Path,
    out_images:   Path,
    out_labels:   Path,
    spatial_size: Tuple[int, int] = (160, 160),
) -> str:
    """
    Load, normalise, resize and save one .h5 slice.

    Returns the slice ID (stem of h5 filename) or "" if the slice is
    all-background and should be skipped.
    """
    image_raw, label_raw = load_h5_slice(h5_path)

    if label_raw.max() == 0:
        return ""

    image_norm       = normalize_intensity(image_raw)
    image_r, label_r = resize_slice(image_norm, label_raw, spatial_size)

    sid = h5_path.stem   # e.g. "patient001_frame01_slice_0"
    np.save(out_images / f"{sid}.npy", image_r)
    np.save(out_labels / f"{sid}.npy", label_r)
    return sid


# ──────────────────────────────────────────────────────────────────────────────
# Dataset-level discovery
# ──────────────────────────────────────────────────────────────────────────────

def find_slice_dir(raw_data_root: Path) -> Path:
    """Locate ACDC_training_slices under raw_data_root."""
    candidates = [
        raw_data_root / "ACDC_preprocessed" / "ACDC_training_slices",
        raw_data_root / "ACDC_training_slices",
        raw_data_root,
    ]
    for c in candidates:
        if c.is_dir() and any(c.glob("patient*.h5")):
            return c

    matches = list(raw_data_root.rglob("ACDC_training_slices"))
    if matches:
        return matches[0]

    raise FileNotFoundError(
        f"Could not find ACDC_training_slices under {raw_data_root}.\n"
        "Expected files like: patient001_frame01_slice_0.h5"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Splits
# ──────────────────────────────────────────────────────────────────────────────

def create_splits(
    all_slice_ids:   List[str],
    initial_labeled: int,
    splits_dir:      Path,
    seed:            int = 42,
) -> None:
    """
    Write labeled_pool.json and unlabeled_pool.json.

    Patient ID is extracted as the part before the first '_frame':
        "patient001_frame01_slice_0"  →  "patient001"

    All 97 training patients are partitioned; the first *initial_labeled*
    go into the labeled seed set.
    """
    random.seed(seed)

    patient_to_slices: Dict[str, List[str]] = {}
    for sid in all_slice_ids:
        pid = sid.split("_frame")[0]
        patient_to_slices.setdefault(pid, []).append(sid)

    patients = sorted(patient_to_slices.keys())
    random.shuffle(patients)

    labeled_patients   = patients[:initial_labeled]
    unlabeled_patients = patients[initial_labeled:]

    labeled_slices   = [s for p in labeled_patients   for s in patient_to_slices[p]]
    unlabeled_slices = [s for p in unlabeled_patients for s in patient_to_slices[p]]

    import shutil
    splits_dir.mkdir(parents=True, exist_ok=True)

    # Guard: if a previous run accidentally created these paths as directories
    # (due to the initial mkdir -p in the setup script), remove them first.
    for fname in ("labeled_pool.json", "unlabeled_pool.json"):
        p = splits_dir / fname
        if p.is_dir():
            shutil.rmtree(p)
            print(f"  [Fix] Removed stale directory {p}")

    with open(splits_dir / "labeled_pool.json",   "w") as f:
        json.dump(labeled_slices, f, indent=2)
    with open(splits_dir / "unlabeled_pool.json", "w") as f:
        json.dump(unlabeled_slices, f, indent=2)

    print(
        f"[Splits] Total patients: {len(patients)}  |  "
        f"Seed labeled: {len(labeled_patients)} patients ({len(labeled_slices)} slices)  |  "
        f"Unlabeled pool: {len(unlabeled_patients)} patients ({len(unlabeled_slices)} slices)"
    )


def reset_splits(cfg: dict) -> None:
    """
    Rebuild labeled_pool.json and unlabeled_pool.json from scratch using
    the already-processed .npy files.  Called when re-starting AL from
    cycle 0 without re-running the full preprocessing step.
    """
    images_dir   = Path(cfg["paths"]["processed_images"])
    splits_dir   = Path(cfg["paths"]["splits_dir"])
    initial_lbl  = cfg["active_learning"]["initial_labeled"]
    seed         = cfg["training"]["seed"]

    all_ids = sorted(p.stem for p in images_dir.glob("*.npy"))
    if not all_ids:
        raise FileNotFoundError(
            f"No processed slices found in {images_dir}. "
            "Run preprocessing first."
        )

    create_splits(
        all_slice_ids   = all_ids,
        initial_labeled = initial_lbl,
        splits_dir      = splits_dir,
        seed            = seed,
    )
    print(f"[Reset] Pools rebuilt from {len(all_ids)} slices in {images_dir}")


# ──────────────────────────────────────────────────────────────────────────────
# Main entry-point
# ──────────────────────────────────────────────────────────────────────────────

def run_preprocessing(cfg: dict, raw_data_root: Path) -> None:
    """
    Process all .h5 training slices and create data-split JSONs.

    Parameters
    ----------
    cfg           : full config dict
    raw_data_root : path to data/raw/acdc
    """
    out_images   = Path(cfg["paths"]["processed_images"])
    out_labels   = Path(cfg["paths"]["processed_labels"])
    splits_dir   = Path(cfg["paths"]["splits_dir"])
    plot_dir     = Path(cfg["paths"]["output_plots"]) / "preprocessing"
    spatial_size = tuple(cfg["preprocessing"]["spatial_size"])

    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    # ── Find slice directory ──────────────────────────────────────────────────
    slice_dir = find_slice_dir(raw_data_root)
    h5_files  = sorted(slice_dir.glob("patient*.h5"))

    # Filter out duplicate files with "(N)" suffixes in filename
    h5_files = [f for f in h5_files if "(" not in f.name]

    print(f"[Preprocessing] Found {len(h5_files)} slice files in {slice_dir.name}/")

    all_slice_ids: List[str] = []
    vis_done:      set       = set()

    for i, h5_path in enumerate(h5_files):
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(h5_files)} slices processed ...", flush=True)

        try:
            sid = process_slice_file(h5_path, out_images, out_labels, spatial_size)
        except Exception as exc:
            print(f"  [ERROR] {h5_path.name}: {exc}")
            continue

        if not sid:
            continue

        all_slice_ids.append(sid)

        # Visualise first valid slice per patient
        pid = sid.split("_frame")[0]
        if pid not in vis_done:
            vis_done.add(pid)
            try:
                image_raw, _ = load_h5_slice(h5_path)
                prep_slice   = np.load(out_images / f"{sid}.npy")
                save_preprocessing_figure(
                    raw_slice  = image_raw,
                    prep_slice = prep_slice,
                    patient_id = pid,
                    save_path  = plot_dir / f"{pid}_preprocess.png",
                )
            except Exception:
                pass

    print(f"[Preprocessing] Saved {len(all_slice_ids)} valid slices "
          f"across {len(vis_done)} patients.")

    if all_slice_ids:
        create_splits(
            all_slice_ids   = all_slice_ids,
            initial_labeled = cfg["active_learning"]["initial_labeled"],
            splits_dir      = splits_dir,
            seed            = cfg["training"]["seed"],
        )
    else:
        print("[WARNING] No valid slices found. Check dataset path.")