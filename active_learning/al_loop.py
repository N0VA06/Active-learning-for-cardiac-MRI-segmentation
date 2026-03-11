from __future__ import annotations
import json
from pathlib import Path
from typing import Dict

import torch

from active_learning.query_strategy import build_query_strategy
from active_learning.uncertainty import mc_dropout_inference
from evaluation.metrics import evaluate_model
from models.unet_model import build_model
from preprocessing.dataset_utils import (
    PoolManager, make_dataloaders, make_inference_loader,
)
from training.loss import build_loss
from training.train import train_one_round
from visualization.plots import save_al_progress_plot, save_training_curves


def run_active_learning(cfg: dict, device: torch.device) -> Dict:
    al_cfg = cfg["active_learning"]
    t_cfg  = cfg["training"]
    p_cfg  = cfg["paths"]

    images_dir = Path(p_cfg["processed_images"])
    labels_dir = Path(p_cfg["processed_labels"])
    models_dir = Path(p_cfg["output_models"])
    plots_dir  = Path(p_cfg["output_plots"])
    logs_dir   = Path(p_cfg["output_logs"])

    for d in (logs_dir, models_dir):
        d.mkdir(parents=True, exist_ok=True)

    pool     = PoolManager(p_cfg["labeled_pool"], p_cfg["unlabeled_pool"])
    initial_lbl = al_cfg["initial_labeled"]
    total_slices = len(pool.labeled) + len(pool.unlabeled)
    seed_slice_threshold = initial_lbl * 25  
    if total_slices > 0 and len(pool.labeled) > seed_slice_threshold:
        from preprocessing.preprocess_acdc import reset_splits
        reset_splits(cfg)
        pool = PoolManager(p_cfg["labeled_pool"], p_cfg["unlabeled_pool"])
    strategy = build_query_strategy(cfg)
    loss_fn  = build_loss(cfg)

    results: Dict = {"cycles": [], "labeled_counts": [],
                     "val_dice": [], "test_dice": [], "test_hd95": []}

    for cycle in range(al_cfg["num_cycles"] + 1):
        print(f"\n{'═'*60}")
        print(f"  AL Cycle {cycle}  |  "
              f"Labeled: {len(pool.labeled)}  |  "
              f"Unlabeled: {len(pool.unlabeled)}")
        print(f"{'═'*60}")

        if not pool.labeled:
            print("  [SKIP] Empty labeled pool.")
            continue

        model = build_model(cfg, device)

        train_loader, val_loader = make_dataloaders(
            labeled_ids=pool.labeled,
            images_dir=images_dir,
            labels_dir=labels_dir,
            batch_size=t_cfg["batch_size"],
            num_workers=t_cfg["num_workers"],
            val_fraction=t_cfg["val_fraction"],
            seed=t_cfg["seed"] + cycle,
        )
        print(f"\n  [Train] {len(train_loader.dataset)} train  "
              f"{len(val_loader.dataset)} val")

        best_dice, history = train_one_round(
            model=model, train_loader=train_loader, val_loader=val_loader,
            loss_fn=loss_fn, cfg=cfg, device=device,
            al_cycle=cycle, save_dir=models_dir,
        )

        # Save training curves
        curves_dir = plots_dir / "training_curves"
        curves_dir.mkdir(parents=True, exist_ok=True)
        save_training_curves(
            history=history, al_cycle=cycle,
            save_path=curves_dir / f"cycle_{cycle:02d}_curves.png",
        )

        # Evaluate
        print(f"\n  [Eval] Cycle {cycle}")
        metrics = evaluate_model(
            model=model, loader=val_loader, cfg=cfg, device=device,
        )
        names = ["RV", "MYO", "LV"]
        class_str = "  ".join(
            f"{n}={d:.3f}" for n, d in zip(names, metrics.get("class_dice", []))
        )
        print(f"  Dice {metrics['mean_dice']:.4f}  "
              f"HD95 {metrics['mean_hd95']:.2f}  "
              f"IoU {metrics['mean_iou']:.4f}  |  {class_str}")

        results["cycles"].append(cycle)
        results["labeled_counts"].append(len(pool.labeled))
        results["val_dice"].append(best_dice)
        results["test_dice"].append(metrics["mean_dice"])
        results["test_hd95"].append(metrics["mean_hd95"])

        with open(logs_dir / f"cycle_{cycle:02d}_metrics.json", "w") as f:
            json.dump({"cycle": cycle, "labeled_slices": len(pool.labeled),
                       **metrics}, f, indent=2)

        if cycle == al_cfg["num_cycles"] or not pool.unlabeled:
            print("\n  [AL] All cycles complete.")
            break

        # MC Dropout uncertainty
        print(f"\n  [Uncertainty] {al_cfg['mc_samples']} MC passes ...")
        inf_loader = make_inference_loader(
            slice_ids=pool.unlabeled, images_dir=images_dir,
            labels_dir=labels_dir, batch_size=t_cfg["batch_size"],
            num_workers=t_cfg["num_workers"],
        )
        scores = mc_dropout_inference(
            model=model, loader=inf_loader,
            T=al_cfg["mc_samples"], device=device,
            method=al_cfg["uncertainty"],
        )

        print(f"\n  [Query] Selecting {al_cfg['query_size']} patient(s) ...")
        selected, _ = strategy.select(
            unlabeled_ids=pool.unlabeled,
            uncertainty_scores=scores,
            k=al_cfg["query_size"],
        )
        pool.move_to_labeled(selected)
        print(f"  Labeled: {len(pool.labeled)}  Unlabeled: {len(pool.unlabeled)}")

    # Final AL progress plot
    seg_dir = plots_dir / "segmentation_results"
    seg_dir.mkdir(parents=True, exist_ok=True)
    save_al_progress_plot(
        cycles=results["cycles"],
        labeled_counts=results["labeled_counts"],
        dice_scores=results["test_dice"],
        save_path=seg_dir / "al_progress.png",
    )

    with open(logs_dir / "al_summary.json", "w") as f:
        json.dump(results, f, indent=2)

    return results