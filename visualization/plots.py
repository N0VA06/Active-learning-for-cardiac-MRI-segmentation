from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")                   # headless backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Preprocessing visualisation
# ──────────────────────────────────────────────────────────────────────────────

def save_preprocessing_figure(
    raw_slice:  np.ndarray,
    prep_slice: np.ndarray,
    patient_id: str,
    save_path:  str | Path,
) -> None:
    """
    Save a side-by-side comparison of raw and preprocessed MRI slices.

    Parameters
    ----------
    raw_slice  : 2-D numpy array (H, W) – original intensity
    prep_slice : 2-D numpy array (H, W) – after preprocessing
    patient_id : string label shown in the figure title
    save_path  : full path of the output PNG
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle(f"Preprocessing: {patient_id}", fontsize=14, fontweight="bold")

    axes[0].imshow(raw_slice, cmap="gray", interpolation="nearest")
    axes[0].set_title("Raw MRI slice")
    axes[0].axis("off")
    axes[0].text(
        0.02, 0.02,
        f"shape: {raw_slice.shape}\nmin: {raw_slice.min():.1f}  max: {raw_slice.max():.1f}",
        transform=axes[0].transAxes,
        color="yellow", fontsize=8, va="bottom",
        bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.5),
    )

    axes[1].imshow(prep_slice, cmap="gray", interpolation="nearest")
    axes[1].set_title("Preprocessed slice (160×160, z-score)")
    axes[1].axis("off")
    axes[1].text(
        0.02, 0.02,
        f"shape: {prep_slice.shape}\nmin: {prep_slice.min():.2f}  max: {prep_slice.max():.2f}",
        transform=axes[1].transAxes,
        color="yellow", fontsize=8, va="bottom",
        bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.5),
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Training curves
# ──────────────────────────────────────────────────────────────────────────────

def save_training_curves(
    history:   Dict[str, List[float]],
    al_cycle:  int,
    save_path: str | Path,
) -> None:
    """
    Save loss and Dice score training curves for one AL cycle.

    Parameters
    ----------
    history   : dict with keys train_loss, val_loss, train_dice, val_dice
    al_cycle  : AL cycle index (used in title)
    save_path : output PNG path
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Training Curves – AL Cycle {al_cycle}", fontsize=13, fontweight="bold")

    # Loss
    ax1.plot(epochs, history["train_loss"], label="Train Loss",  color="#2196F3", lw=2)
    ax1.plot(epochs, history["val_loss"],   label="Val Loss",    color="#F44336", lw=2, ls="--")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Dice + CE Loss")
    ax1.set_title("Loss vs Epoch")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Dice
    ax2.plot(epochs, history["train_dice"], label="Train Dice", color="#4CAF50", lw=2)
    ax2.plot(epochs, history["val_dice"],   label="Val Dice",   color="#FF9800", lw=2, ls="--")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Dice Score")
    ax2.set_title("Dice vs Epoch")
    ax2.set_ylim(0, 1)
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Active Learning progress
# ──────────────────────────────────────────────────────────────────────────────

def save_al_progress_plot(
    cycles:         List[int],
    labeled_counts: List[int],
    dice_scores:    List[float],
    save_path:      str | Path,
) -> None:
    """
    Save an AL progress plot: Dice score and labeled sample count vs cycle.

    Parameters
    ----------
    cycles         : list of cycle indices [0, 1, 2, ...]
    labeled_counts : number of labeled slices at each cycle
    dice_scores    : validation/test Dice at each cycle
    save_path      : output PNG path
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 9), sharex=True)
    fig.suptitle("Active Learning Progress", fontsize=14, fontweight="bold")

    # Dice vs cycle
    ax1.plot(cycles, dice_scores, "o-", color="#4CAF50", lw=2.5, ms=8, label="Dice Score")
    ax1.fill_between(cycles, dice_scores, alpha=0.15, color="#4CAF50")
    ax1.set_ylabel("Mean Dice Score")
    ax1.set_title("Segmentation Performance vs AL Cycle")
    ax1.set_ylim(0, 1)
    ax1.grid(alpha=0.3)
    ax1.legend()

    # Labeled count vs cycle
    ax2.bar(cycles, labeled_counts, color="#2196F3", alpha=0.75, edgecolor="white")
    ax2.set_xlabel("AL Cycle")
    ax2.set_ylabel("# Labeled Slices")
    ax2.set_title("Labeled Pool Size vs AL Cycle")
    ax2.grid(axis="y", alpha=0.3)
    ax2.set_xticks(cycles)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Summary metrics table
# ──────────────────────────────────────────────────────────────────────────────

def save_metrics_table_plot(
    metrics_per_cycle: List[Dict],
    save_path:         str | Path,
) -> None:
    """
    Render a table of per-cycle metrics as a matplotlib figure.

    Parameters
    ----------
    metrics_per_cycle : list of dicts with keys: cycle, mean_dice, mean_hd95, ...
    save_path         : output PNG path
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    col_labels = ["Cycle", "Labeled\nSlices", "Dice", "HD95", "IoU", "Precision", "Recall"]
    rows = []
    for m in metrics_per_cycle:
        rows.append([
            m.get("cycle", "–"),
            m.get("labeled_slices", "–"),
            f"{m.get('mean_dice', 0):.4f}",
            f"{m.get('mean_hd95', 0):.2f}",
            f"{m.get('mean_iou', 0):.4f}",
            f"{m.get('mean_precision', 0):.4f}",
            f"{m.get('mean_recall', 0):.4f}",
        ])

    fig, ax = plt.subplots(figsize=(12, max(3, len(rows) * 0.5 + 1.5)))
    ax.axis("off")
    table = ax.table(
        cellText    = rows,
        colLabels   = col_labels,
        loc         = "center",
        cellLoc     = "center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.6)
    fig.suptitle("Per-Cycle Metrics Summary", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
