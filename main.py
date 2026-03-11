"""
main.py  –  Entry-point for Active Learning ACDC segmentation.

Usage:
    python main.py
    python main.py --skip-download
    python main.py --skip-preprocess
    python main.py --skip-al
    python main.py --config path/to/config.yaml
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import torch
import yaml


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def setup_environment(cfg: dict) -> None:
    """Create output directories. Never mkdir a .json path."""
    import shutil
    for key, path_str in cfg["paths"].items():
        p = Path(path_str)
        if p.suffix == ".json":
            p.parent.mkdir(parents=True, exist_ok=True)
            if p.is_dir():
                shutil.rmtree(p)
                print(f"  [Fix] Removed stale dir at JSON path: {p}")
        else:
            p.mkdir(parents=True, exist_ok=True)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        d = torch.device("cuda")
        print(f"  Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        d = torch.device("cpu")
        print("  Using CPU")
    return d


def set_seed(seed: int) -> None:
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def banner(text: str) -> None:
    print(f"\n{'═'*60}\n  {text}\n{'═'*60}")


# ── Steps ─────────────────────────────────────────────────────────────────────

def step_download(cfg: dict) -> Path:
    banner("Step 1 – Downloading ACDC Dataset")
    raw_dir = Path(cfg["paths"]["raw_data"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    if any(raw_dir.rglob("*.h5")):
        print(f"  Dataset already at {raw_dir}")
        return raw_dir
    try:
        import kagglehub, shutil
        kpath = kagglehub.dataset_download("anhoangvo/acdc-dataset")
        print(f"  Downloaded to: {kpath}")
        shutil.copytree(Path(kpath), raw_dir, dirs_exist_ok=True)
        print(f"  Copied to {raw_dir}")
    except Exception as e:
        print(f"  [WARNING] Download failed: {e}")
    return raw_dir


def step_preprocess(cfg: dict, raw_dir: Path) -> None:
    banner("Step 2 – Preprocessing")
    images_dir   = Path(cfg["paths"]["processed_images"])
    labeled_json = Path(cfg["paths"]["labeled_pool"])
    if labeled_json.is_file() and any(images_dir.glob("*.npy")):
        print("  Processed data already exists. Skipping.")
        return
    from preprocessing.preprocess_acdc import run_preprocessing
    run_preprocessing(cfg, raw_dir)


def step_al(cfg: dict, device: torch.device) -> dict:
    banner("Step 3 – Active Learning Loop")
    from active_learning.al_loop import run_active_learning
    return run_active_learning(cfg, device)


def step_final_inference(cfg: dict, device: torch.device) -> None:
    banner("Step 4 – Final Inference & Visualisation")
    from models.unet_model import build_model
    from evaluation.inference import load_best_checkpoint, run_inference
    from preprocessing.dataset_utils import PoolManager, make_inference_loader
    from visualization.plots import save_metrics_table_plot

    model = build_model(cfg, device)
    try:
        model = load_best_checkpoint(
            model=model, models_dir=cfg["paths"]["output_models"], device=device,
        )
    except FileNotFoundError as e:
        print(f"  [SKIP] {e}")
        return

    pool = PoolManager(cfg["paths"]["labeled_pool"], cfg["paths"]["unlabeled_pool"])
    loader = make_inference_loader(
        slice_ids=pool.labeled,
        images_dir=cfg["paths"]["processed_images"],
        labels_dir=cfg["paths"]["processed_labels"],
        batch_size=cfg["training"]["batch_size"],
        num_workers=cfg["training"]["num_workers"],
    )

    preds_dir = Path(cfg["paths"]["output_preds"])
    plots_dir = Path(cfg["paths"]["output_plots"]) / "segmentation_results"
    metrics   = run_inference(
        model=model, loader=loader, cfg=cfg, device=device,
        preds_dir=preds_dir, plots_dir=plots_dir, max_vis=12,
    )

    # Metrics summary table
    logs_dir = Path(cfg["paths"]["output_logs"])
    all_m = []
    for mf in sorted(logs_dir.glob("cycle_*_metrics.json")):
        with open(mf) as f:
            all_m.append(json.load(f))
    if all_m:
        save_metrics_table_plot(all_m, plots_dir / "metrics_summary_table.png")

    names = ["RV", "Myocardium", "LV"]
    print(f"\n{'─'*50}")
    print("  FINAL RESULTS")
    print(f"{'─'*50}")
    print(f"  Dice       : {metrics['mean_dice']:.4f}")
    print(f"  HD95       : {metrics['mean_hd95']:.2f} mm")
    print(f"  IoU        : {metrics['mean_iou']:.4f}")
    print(f"  Precision  : {metrics['mean_precision']:.4f}")
    print(f"  Recall     : {metrics['mean_recall']:.4f}")
    for name, d in zip(names, metrics.get("class_dice", [])):
        print(f"  Dice {name:<12}: {d:.4f}")
    print(f"{'─'*50}")

    with open(logs_dir / "final_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config",          default="configs/config.yaml")
    p.add_argument("--skip-download",   action="store_true")
    p.add_argument("--skip-preprocess", action="store_true")
    p.add_argument("--skip-al",         action="store_true")
    p.add_argument("--reset-pool",      action="store_true",
                   help="Reset labeled/unlabeled pools to the initial seed "
                        "split before running AL.  Use this when re-running "
                        "after a completed previous AL experiment.")
    args = p.parse_args()
    t0   = time.time()

    print("\n" + "═"*60)
    print("  Active Learning · ACDC Cardiac MRI · PyTorch + MONAI")
    print("═"*60)

    cfg    = load_config(args.config)
    setup_environment(cfg)
    set_seed(cfg["training"]["seed"])
    device = get_device()

    raw_dir = (
        step_download(cfg)
        if not args.skip_download
        else Path(cfg["paths"]["raw_data"])
    )
    if not args.skip_preprocess:
        step_preprocess(cfg, raw_dir)

    # ── Optional explicit pool reset ─────────────────────────────────────────
    if getattr(args, "reset_pool", False):
        banner("Resetting AL pools to initial seed split")
        from preprocessing.preprocess_acdc import reset_splits
        reset_splits(cfg)
        print("  Done.")

    al_results = {}
    if not args.skip_al:
        al_results = step_al(cfg, device)
        if al_results.get("test_dice"):
            best = max(range(len(al_results["test_dice"])),
                       key=lambda i: al_results["test_dice"][i])
            print(f"\n  Best cycle: {best}  "
                  f"Dice: {al_results['test_dice'][best]:.4f}  "
                  f"HD95: {al_results['test_hd95'][best]:.2f}")

    step_final_inference(cfg, device)
    print(f"\n  Total runtime: {(time.time()-t0)/60:.1f} min")
    print("═"*60 + "\n")


if __name__ == "__main__":
    main()