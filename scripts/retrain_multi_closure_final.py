#!/usr/bin/env python
"""Retrain the multi-closure PI sweep winner at the manuscript-canonical
training budget (300 epochs, patience 30) across seeds 11/23/37 and write
test-set metrics in the same shape as the other final-eval entries.

Writes to ``reports/manuscript_hpo/multi_closure_final.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from roaster_piml.thesis_full_state import (  # noqa: E402
    MultiClosureFullStateModel,
    set_seed,
    train_model as train_fullstate_model,
)
from scripts.tune_manuscript_models import (  # noqa: E402
    load_manuscript_cohort,
    split_cohort,
    pooled_rollout_metrics_fullstate,
    roast_bootstrap_rollout_r2,
)


DEFAULT_INPUT = ROOT / "data" / "processed" / "roast_timeseries_p2_only.csv"
BEST_CONFIG_PATH = ROOT / "reports" / "manuscript_hpo" / "multi_closure" / "best_config.json"
OUTPATH = ROOT / "reports" / "manuscript_hpo" / "multi_closure_final.json"

FINAL_SEEDS = (11, 23, 37)
FINAL_EPOCHS = 300
FINAL_PATIENCE = 30


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    args = parser.parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Cohort not found: {args.input}")
    if not BEST_CONFIG_PATH.exists():
        raise FileNotFoundError(f"Best config not found: {BEST_CONFIG_PATH}")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    best_record = json.loads(BEST_CONFIG_PATH.read_text(encoding="utf-8"))
    cfg = best_record["config"]
    print(f"Retraining multi-closure winner across {len(FINAL_SEEDS)} seeds at {FINAL_EPOCHS} epochs")
    print(f"config: {json.dumps(cfg, indent=2)}")

    sequences = load_manuscript_cohort(args.input, None, ["P10", "P13", "P16"], ["p2"])
    train_seq, val_seq, test_seq = split_cohort(sequences)
    print(f"Cohort: train={len(train_seq)} val={len(val_seq)} test={len(test_seq)} device={device}")

    per_seed: dict[str, dict] = {}
    for seed in FINAL_SEEDS:
        print(f"\n=== seed {seed} ===", flush=True)
        set_seed(seed)
        model = MultiClosureFullStateModel(
            he_hidden_widths=tuple(cfg["he_hidden_widths"]),
            moisture_hidden_widths=tuple(cfg["moisture_hidden_widths"]),
            reaction_hidden_widths=tuple(cfg["reaction_hidden_widths"]),
        )
        # No fixed priors per the repair-focused framing; init_net trainable.
        start = time.perf_counter()
        meta = train_fullstate_model(
            model,
            train_seq,
            val_seq,
            epochs=FINAL_EPOCHS,
            lr=float(cfg["lr"]),
            weight_decay=float(cfg["weight_decay"]),
            batch_size=int(cfg["batch_size"]),
            device=device,
            tgo_weight=float(cfg["tgo_weight"]),
            detach_state_steps=bool(cfg["detach_state_steps"]),
            warmup_steps=int(cfg["warmup_steps"]),
            patience=FINAL_PATIENCE,
        )
        train_time = time.perf_counter() - start

        val_one, val_roll, _ = pooled_rollout_metrics_fullstate(model, val_seq)
        test_one, test_roll, test_per_roast = pooled_rollout_metrics_fullstate(model, test_seq)
        ci_lo, ci_hi = roast_bootstrap_rollout_r2(test_per_roast, n_boot=1000, seed=seed)

        param_count = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
        per_seed[str(seed)] = {
            "rollout_metrics": {
                "r2": float(test_roll.r2),
                "rmse": float(test_roll.rmse),
                "mae": float(test_roll.mae),
                "n": int(test_roll.n),
            },
            "one_step_metrics": {
                "r2": float(test_one.r2),
                "rmse": float(test_one.rmse),
                "mae": float(test_one.mae),
                "n": int(test_one.n),
            },
            "rollout_r2_ci95": [float(ci_lo), float(ci_hi)],
            "param_count": param_count,
            "per_roast_r2": {
                rid: float(
                    1.0
                    - np.sum((np.asarray(p["actual"]) - np.asarray(p["rollout_pred"])) ** 2)
                    / max(np.sum((np.asarray(p["actual"]) - np.mean(p["actual"])) ** 2), 1e-12)
                )
                for rid, p in test_per_roast.items()
            },
            "val_rollout_r2": float(val_roll.r2),
            "training": {
                "best_val_loss": float(meta["best_val_loss"]),
                "best_epoch": int(meta["best_epoch"]),
                "epochs_run": int(meta["epochs_run"]),
                "wall_time_sec": float(train_time),
            },
        }
        print(
            f"  test R^2: {test_roll.r2:.4f}  CI95=[{ci_lo:.4f}, {ci_hi:.4f}]  "
            f"params={param_count}  wall={train_time:.0f}s",
            flush=True,
        )

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    OUTPATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPATH.write_text(json.dumps({
        "config": cfg,
        "final_seeds": list(FINAL_SEEDS),
        "final_epochs": FINAL_EPOCHS,
        "final_patience": FINAL_PATIENCE,
        "per_seed": per_seed,
    }, indent=2), encoding="utf-8")
    print(f"\nSaved to {OUTPATH}")
    seed11 = per_seed["11"]
    print(
        f"\nSEED 11: test rollout R^2 = {seed11['rollout_metrics']['r2']:.4f}  "
        f"CI95=[{seed11['rollout_r2_ci95'][0]:.4f}, {seed11['rollout_r2_ci95'][1]:.4f}]  "
        f"params={seed11['param_count']}"
    )


if __name__ == "__main__":
    main()
