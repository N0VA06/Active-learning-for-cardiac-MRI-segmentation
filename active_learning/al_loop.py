"""
active_learning/al_loop.py
===========================
Main AL loop. Dispatches to the right uncertainty scorer based on config.

uncertainty values in config:
    "mul"      → multiplicative_ud_uncertainty  (U × D)   ← recommended
    "combined" → combined_mc_exit_uncertainty   (α·U + β·D additive)
    "entropy"  → mc_dropout_inference
    "bald"     → mc_dropout_inference (bald)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import torch

from active_learning.query_strategy import build_query_strategy
from active_learning.uncertainty import (
    mc_dropout_inference,
    combined_mc_exit_uncertainty,
    multiplicative_ud_uncertainty,
)
from evaluation.metrics import evaluate_model
from preprocessing.dataset_utils import (
    PoolManager, make_dataloaders, make_inference_loader,
)
from training.loss import build_loss
from training.train import train_one_round
from visualization.plots import save_al_progress_plot, save_training_curves


def _build_model(cfg: dict, device: torch.device):
    ee_cfg = cfg.get("early_exit", {})
    if ee_cfg.get("enabled", False):
        from models.early_exit_unet import build_early_exit_model
        return build_early_exit_model(cfg, device)
    from models.unet_model import build_model
    return build_model(cfg, device)


def _score_uncertainty(
    model:           torch.nn.Module,
    inf_loader,
    labeled_loader,
    cfg:             dict,
    device:          torch.device,
) -> Dict[str, float]:
    al_cfg  = cfg["active_learning"]
    ee_cfg  = cfg.get("early_exit", {})
    method  = al_cfg["uncertainty"]
    T       = al_cfg["mc_samples"]

    if method == "mul":
        print(f"\n  [Uncertainty] Multiplicative U×D  (T={T})")
        return multiplicative_ud_uncertainty(
            model            = model,
            unlabeled_loader = inf_loader,
            labeled_loader   = labeled_loader,
            T                = T,
            device           = device,
            method           = "entropy",
        )

    if method == "combined" and ee_cfg.get("enabled", False):
        alpha = ee_cfg.get("alpha", 0.5)
        beta  = ee_cfg.get("beta",  0.5)
        print(f"\n  [Uncertainty] Combined MC+Exit (additive)  "
              f"T={T}  α={alpha}  β={beta}")
        return combined_mc_exit_uncertainty(
            model  = model,
            loader = inf_loader,
            T      = T,
            device = device,
            alpha  = alpha,
            beta   = beta,
            method = "entropy",
        )

    print(f"\n  [Uncertainty] MC-Dropout ({method})  T={T}")
    return mc_dropout_inference(
        model  = model,
        loader = inf_loader,
        T      = T,
        device = device,
        method = method,
    )


def run_active_learning(cfg: dict, device: torch.device) -> Dict:
    al_cfg = cfg["active_learning"]
    t_cfg  = cfg["training"]
    p_cfg  = cfg["paths"]
    ee_cfg = cfg.get("early_exit", {})

    images_dir = Path(p_cfg["processed_images"])
    labels_dir = Path(p_cfg["processed_labels"])
    models_dir = Path(p_cfg["output_models"])
    plots_dir  = Path(p_cfg["output_plots"])
    logs_dir   = Path(p_cfg["output_logs"])

    for d in (logs_dir, models_dir):
        d.mkdir(parents=True, exist_ok=True)

    pool = PoolManager(p_cfg["labeled_pool"], p_cfg["unlabeled_pool"])

    initial_lbl         = al_cfg["initial_labeled"]
    total_slices        = len(pool.labeled) + len(pool.unlabeled)
    seed_slice_threshold = initial_lbl * 25
    if total_slices > 0 and len(pool.labeled) > seed_slice_threshold:
        from preprocessing.preprocess_acdc import reset_splits
        reset_splits(cfg)
        pool = PoolManager(p_cfg["labeled_pool"], p_cfg["unlabeled_pool"])

    strategy = build_query_strategy(cfg)
    loss_fn  = build_loss(cfg)

    if ee_cfg.get("enabled", False):
        print(f"[AL] Early-exit ON  exits={ee_cfg.get('exit_indices', [1,2])}  "
              f"aux_λ={ee_cfg.get('aux_loss_weight', 0.3)}")
    print(f"[AL] Query strategy: {al_cfg['uncertainty']}")

    results: Dict = {
        "cycles": [], "labeled_counts": [],
        "val_dice": [], "test_dice": [], "test_hd95": [],
    }

    for cycle in range(al_cfg["num_cycles"] + 1):
        print(f"\n{'═'*60}")
        print(f"  AL Cycle {cycle}  |  "
              f"Labeled: {len(pool.labeled)}  |  "
              f"Unlabeled: {len(pool.unlabeled)}")
        print(f"{'═'*60}")

        if not pool.labeled:
            print("  [SKIP] Empty labeled pool.")
            continue

        model = _build_model(cfg, device)

        train_loader, val_loader = make_dataloaders(
            labeled_ids  = pool.labeled,
            images_dir   = images_dir,
            labels_dir   = labels_dir,
            batch_size   = t_cfg["batch_size"],
            num_workers  = t_cfg["num_workers"],
            val_fraction = t_cfg["val_fraction"],
            seed         = t_cfg["seed"] + cycle,
        )
        print(f"\n  [Train] {len(train_loader.dataset)} train  "
              f"{len(val_loader.dataset)} val")

        best_dice, history = train_one_round(
            model=model, train_loader=train_loader, val_loader=val_loader,
            loss_fn=loss_fn, cfg=cfg, device=device,
            al_cycle=cycle, save_dir=models_dir,
        )

        curves_dir = plots_dir / "training_curves"
        curves_dir.mkdir(parents=True, exist_ok=True)
        save_training_curves(
            history=history, al_cycle=cycle,
            save_path=curves_dir / f"cycle_{cycle:02d}_curves.png",
        )

        print(f"\n  [Eval] Cycle {cycle}")
        metrics = evaluate_model(model=model, loader=val_loader,
                                 cfg=cfg, device=device)
        names     = ["RV", "MYO", "LV"]
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

        # ── Uncertainty + diversity scoring ───────────────────────────────
        inf_loader = make_inference_loader(
            slice_ids=pool.unlabeled, images_dir=images_dir,
            labels_dir=labels_dir, batch_size=t_cfg["batch_size"],
            num_workers=t_cfg["num_workers"],
        )
        # Labeled loader needed for diversity reference (U×D only)
        lbl_loader = make_inference_loader(
            slice_ids=pool.labeled, images_dir=images_dir,
            labels_dir=labels_dir, batch_size=t_cfg["batch_size"],
            num_workers=t_cfg["num_workers"],
        )

        scores = _score_uncertainty(model, inf_loader, lbl_loader, cfg, device)

        print(f"\n  [Query] Selecting {al_cfg['query_size']} patient(s) ...")
        selected, _ = strategy.select(
            unlabeled_ids=pool.unlabeled,
            uncertainty_scores=scores,
            k=al_cfg["query_size"],
        )
        pool.move_to_labeled(selected)
        print(f"  Labeled: {len(pool.labeled)}  Unlabeled: {len(pool.unlabeled)}")

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