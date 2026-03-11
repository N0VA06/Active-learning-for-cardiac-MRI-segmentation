from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Colour map for ACDC classes
# ──────────────────────────────────────────────────────────────────────────────

#  0 = Background  →  transparent / black
#  1 = RV          →  blue
#  2 = Myocardium  →  green
#  3 = LV          →  red
ACDC_COLOURS = {
    0: (0.0,  0.0,  0.0),    # Background
    1: (0.27, 0.51, 0.71),   # RV       – steel blue
    2: (0.40, 0.74, 0.40),   # Myo      – medium green
    3: (0.84, 0.15, 0.16),   # LV       – brick red
}

CLASS_NAMES = {0: "Background", 1: "RV", 2: "Myocardium", 3: "LV"}


def colorise_mask(mask: np.ndarray, alpha: float = 0.7) -> np.ndarray:
    """
    Convert a (H, W) integer class mask to an (H, W, 4) RGBA image.

    Parameters
    ----------
    mask  : integer array with class indices
    alpha : opacity for non-background classes

    Returns
    -------
    rgba : (H, W, 4) float32 array in [0, 1]
    """
    H, W = mask.shape
    rgba = np.zeros((H, W, 4), dtype=np.float32)

    for cls_idx, colour in ACDC_COLOURS.items():
        region = mask == cls_idx
        rgba[region, 0] = colour[0]
        rgba[region, 1] = colour[1]
        rgba[region, 2] = colour[2]
        rgba[region, 3] = 0.0 if cls_idx == 0 else alpha

    return rgba


# ──────────────────────────────────────────────────────────────────────────────
# Comparison figure
# ──────────────────────────────────────────────────────────────────────────────

def save_segmentation_comparison(
    image:     np.ndarray,
    gt_mask:   np.ndarray,
    pred_mask: np.ndarray,
    slice_id:  str,
    save_path: str | Path,
) -> None:
    """
    Save a three-panel segmentation comparison figure.

    Panels:
        1. Input MRI slice
        2. Ground-truth mask overlaid on MRI
        3. Predicted mask overlaid on MRI

    Parameters
    ----------
    image     : (H, W) float – preprocessed MRI slice
    gt_mask   : (H, W) int   – ground-truth class indices
    pred_mask : (H, W) int   – predicted class indices
    slice_id  : identifier string used in title
    save_path : output PNG path
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    gt_rgba   = colorise_mask(gt_mask)
    pred_rgba = colorise_mask(pred_mask)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Segmentation: {slice_id}", fontsize=13, fontweight="bold")

    # Panel 1: MRI input
    axes[0].imshow(image, cmap="gray", interpolation="nearest")
    axes[0].set_title("Input MRI Slice")
    axes[0].axis("off")

    # Panel 2: Ground truth
    axes[1].imshow(image, cmap="gray", interpolation="nearest")
    axes[1].imshow(gt_rgba, interpolation="nearest")
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")

    # Panel 3: Prediction
    axes[2].imshow(image, cmap="gray", interpolation="nearest")
    axes[2].imshow(pred_rgba, interpolation="nearest")
    axes[2].set_title("Prediction")
    axes[2].axis("off")

    # Legend
    patches = [
        mpatches.Patch(color=ACDC_COLOURS[cls], label=name)
        for cls, name in CLASS_NAMES.items()
        if cls != 0
    ]
    fig.legend(handles=patches, loc="lower center", ncol=3, fontsize=10,
               bbox_to_anchor=(0.5, -0.04))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Uncertainty map figure
# ──────────────────────────────────────────────────────────────────────────────

def save_uncertainty_map(
    image:        np.ndarray,
    pred_mask:    np.ndarray,
    unc_map:      np.ndarray,
    slice_id:     str,
    save_path:    str | Path,
) -> None:
    """
    Save a three-panel figure: input | prediction | uncertainty heatmap.

    Parameters
    ----------
    image     : (H, W) float – MRI slice
    pred_mask : (H, W) int   – predicted class mask
    unc_map   : (H, W) float – per-pixel entropy / uncertainty
    slice_id  : identifier string
    save_path : output PNG path
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    pred_rgba = colorise_mask(pred_mask)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Uncertainty: {slice_id}", fontsize=13, fontweight="bold")

    axes[0].imshow(image,    cmap="gray", interpolation="nearest")
    axes[0].set_title("Input MRI")
    axes[0].axis("off")

    axes[1].imshow(image,    cmap="gray", interpolation="nearest")
    axes[1].imshow(pred_rgba, interpolation="nearest")
    axes[1].set_title("Prediction")
    axes[1].axis("off")

    im = axes[2].imshow(unc_map, cmap="hot", interpolation="nearest")
    axes[2].set_title("Predictive Entropy")
    axes[2].axis("off")
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
