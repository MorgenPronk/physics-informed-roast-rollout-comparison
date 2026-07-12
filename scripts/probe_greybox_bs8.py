#!/usr/bin/env python
"""Targeted bs=8 probe for the physics-informed grey-box on the manuscript cohort.

Trains a single configuration of the LearnedHeFullStateModel that the main
overnight sweep deliberately excluded for compute reasons (batch_size=8 with
the largest closure architecture), and reports rollout R^2 on both the
validation and held-out test splits. This resolves whether the sweep's
exclusion of bs=8 materially undercounted the physics-informed performance.

The config matches the original dry-run trial that produced val R^2~0.92 on
the manuscript cohort.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from roaster_piml.model import compute_metrics  # noqa: E402
from roaster_piml.thesis_full_state import (  # noqa: E402
    LearnedHeFullStateModel,
    set_seed,
    train_model,
)
from scripts.tune_manuscript_models import (  # noqa: E402
    load_manuscript_cohort,
    split_cohort,
    pooled_rollout_metrics_fullstate,
    roast_bootstrap_rollout_r2,
)


CONFIG = {
    "hidden_widths": (256, 128, 64, 32),
    "lr": 1.72e-3,
    "weight_decay": 1e-5,
    "batch_size": 8,
    "epochs": 300,
    "patience": 30,
    "tgo_weight": 0.0,
    "detach_state_steps": False,
    "warmup_steps": 0,
    "seed": 11,
}

DEFAULT_INPUT = ROOT / "data" / "processed" / "roast_timeseries_p2_only.csv"
OUTPATH = ROOT / "reports" / "manuscript_hpo" / "greybox_bs8_probe.json"


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    args = parser.parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Cohort file not found: {args.input}")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading cohort from {args.input}", flush=True)
    sequences = load_manuscript_cohort(args.input, None, ["P10", "P13", "P16"], ["p2"])
    train_seq, val_seq, test_seq = split_cohort(sequences)
    print(
        f"Cohort: total={len(sequences)} train={len(train_seq)} "
        f"val={len(val_seq)} test={len(test_seq)} device={device}",
        flush=True,
    )

    set_seed(CONFIG["seed"])
    model = LearnedHeFullStateModel(hidden_widths=CONFIG["hidden_widths"])
    start = time.perf_counter()
    meta = train_model(
        model,
        train_seq,
        val_seq,
        epochs=CONFIG["epochs"],
        lr=CONFIG["lr"],
        weight_decay=CONFIG["weight_decay"],
        batch_size=CONFIG["batch_size"],
        device=device,
        tgo_weight=CONFIG["tgo_weight"],
        detach_state_steps=CONFIG["detach_state_steps"],
        warmup_steps=CONFIG["warmup_steps"],
        patience=CONFIG["patience"],
    )
    train_time = time.perf_counter() - start
    print(
        f"Trained in {train_time:.1f}s | best_val_loss={meta['best_val_loss']:.4f} "
        f"best_epoch={meta['best_epoch']} epochs_run={meta['epochs_run']}",
        flush=True,
    )

    val_one, val_roll, val_per_roast = pooled_rollout_metrics_fullstate(model, val_seq)
    test_one, test_roll, test_per_roast = pooled_rollout_metrics_fullstate(model, test_seq)
    ci_lo, ci_hi = roast_bootstrap_rollout_r2(test_per_roast, n_boot=1000, seed=11)

    result = {
        "config": {k: list(v) if isinstance(v, tuple) else v for k, v in CONFIG.items()},
        "training": {
            "best_val_loss": float(meta["best_val_loss"]),
            "best_epoch": int(meta["best_epoch"]),
            "epochs_run": int(meta["epochs_run"]),
            "wall_time_sec": float(train_time),
            "history": meta["history"],
        },
        "val": {
            "one_step_r2": float(val_one.r2),
            "rollout_r2": float(val_roll.r2),
            "rollout_rmse": float(val_roll.rmse),
        },
        "test": {
            "one_step_r2": float(test_one.r2),
            "rollout_r2": float(test_roll.r2),
            "rollout_rmse": float(test_roll.rmse),
            "rollout_r2_ci95": [float(ci_lo), float(ci_hi)],
        },
        "param_count": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "per_roast_test_r2": {
            rid: float(
                1.0
                - np.sum((np.asarray(payload["actual"]) - np.asarray(payload["rollout_pred"])) ** 2)
                / max(np.sum((np.asarray(payload["actual"]) - np.mean(payload["actual"])) ** 2), 1e-12)
            )
            for rid, payload in test_per_roast.items()
        },
    }
    OUTPATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        f"\nbs=8 probe complete.\n"
        f"  val rollout R^2 : {val_roll.r2:.4f}\n"
        f"  test rollout R^2: {test_roll.r2:.4f}  CI95=[{ci_lo:.4f}, {ci_hi:.4f}]\n"
        f"  test rollout RMSE: {test_roll.rmse:.4f}\n"
        f"  saved to {OUTPATH}",
        flush=True,
    )


if __name__ == "__main__":
    main()
