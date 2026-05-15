#!/usr/bin/env python
"""Three priors probes — anchoring the leftmost end of the physics-content spectrum.

Variants (selected via ``--variant``):

    true_mechanistic       No training. All learnable parameters frozen.
                           Initial latent state uses literature priors:
                               T_b(0) = 20 C  (room temperature, green beans)
                               X_b(0) = 0.12  (literature green-bean moisture)
                               T_rp(0) = measured T_c(0)
                               H_e(0) ~ softplus(0) (state is bounded; physical zero
                                                     would require modifying model)
                           This is the "what does pure literature give you" baseline.

    scalar_tuned_priors    Initial state from literature priors as above; init_net
                           frozen at zero output; the learnable global scalar physics
                           parameters (log_he, log_ab, log_mb, log_x_coeff, log_ar,
                           log_het, log_hv, ...) ARE trained. Measures the marginal
                           value of scalar tuning on top of fixed priors.

    pi_fixed_priors        Same fixed priors and frozen init_net as above, but with
                           the learned-closure MLP enabled. Measures whether the PI
                           closure's contribution survives without the init_net's
                           learned per-roast latent-state adjustments.

Result for each variant is written to
``reports/manuscript_hpo/priors_probe_<variant>.json``.
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
    ConstantHeFullStateModel,
    LearnedHeFullStateModel,
    set_seed,
    train_model,
)
from scripts.tune_manuscript_models import (  # noqa: E402
    load_manuscript_cohort,
    pooled_rollout_metrics_fullstate,
    roast_bootstrap_rollout_r2,
    split_cohort,
)


VARIANTS = {
    "true_mechanistic": {
        "model": "constant",
        "fixed_initial_bean_temp_c": 20.0,
        "fixed_initial_moisture_ratio": 0.12,
        "freeze_init_net": True,
        "freeze_all_scalars": True,
        "train": False,
        "epochs": 0,
        "lr": 0.0,
        "weight_decay": 0.0,
        "batch_size": 16,
        "hidden_widths": None,
    },
    "scalar_tuned_priors": {
        "model": "constant",
        "fixed_initial_bean_temp_c": 20.0,
        "fixed_initial_moisture_ratio": 0.12,
        "freeze_init_net": True,
        "freeze_all_scalars": False,
        "train": True,
        "epochs": 300,
        "lr": 5.4e-4,                      # white-box sweep winner's lr
        "weight_decay": 1e-5,
        "batch_size": 16,
        "hidden_widths": None,
    },
    "scalar_tuned_priors_with_init_net": {
        # Same as scalar_tuned_priors but with init_net trainable.
        # Decomposes whether the manuscript's "Mechanistic" failure (R^2=-0.44)
        # came from learning the init_net or from the absence of fixed priors.
        "model": "constant",
        "fixed_initial_bean_temp_c": 20.0,
        "fixed_initial_moisture_ratio": 0.12,
        "freeze_init_net": False,
        "freeze_all_scalars": False,
        "train": True,
        "epochs": 300,
        "lr": 5.4e-4,
        "weight_decay": 1e-5,
        "batch_size": 16,
        "hidden_widths": None,
    },
    "pi_fixed_priors": {
        "model": "learned",
        "fixed_initial_bean_temp_c": 20.0,
        "fixed_initial_moisture_ratio": 0.12,
        "freeze_init_net": True,
        "freeze_all_scalars": False,
        "train": True,
        "epochs": 300,
        "lr": 1.19e-3,                     # grey-box sweep winner's lr
        "weight_decay": 1e-5,
        "batch_size": 16,
        "hidden_widths": [128, 64, 32],     # grey-box sweep winner's arch
    },
}

DEFAULT_INPUT = ROOT.parent.parent.parent / "data" / "processed" / "roast_timeseries_p2_only.csv"


def build_model(variant: str, cfg: dict):
    common = dict(
        fixed_initial_bean_temp_c=cfg["fixed_initial_bean_temp_c"],
        fixed_initial_moisture_ratio=cfg["fixed_initial_moisture_ratio"],
    )
    if cfg["model"] == "constant":
        model = ConstantHeFullStateModel(**common)
    elif cfg["model"] == "learned":
        model = LearnedHeFullStateModel(
            hidden_widths=tuple(cfg["hidden_widths"]),
            **common,
        )
    else:
        raise ValueError(f"unknown model variant {cfg['model']!r}")

    if cfg["freeze_init_net"]:
        for p in model.init_net.parameters():
            p.requires_grad = False

    if cfg["freeze_all_scalars"]:
        for name, p in model.named_parameters():
            # Freeze everything except the closure MLP (he_net) if present.
            if "he_net." in name:
                continue
            p.requires_grad = False

    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True, choices=list(VARIANTS))
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Cohort not found: {args.input}")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = VARIANTS[args.variant]
    sequences = load_manuscript_cohort(args.input, None, ["P10", "P13", "P16"], ["p2"])
    train_seq, val_seq, test_seq = split_cohort(sequences)
    print(
        f"Variant: {args.variant}  Cohort: total={len(sequences)} "
        f"train={len(train_seq)} val={len(val_seq)} test={len(test_seq)} device={device}",
        flush=True,
    )

    set_seed(args.seed)
    model = build_model(args.variant, cfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Params: trainable={trainable} total={total}", flush=True)

    history = []
    best_epoch = 0
    train_time = 0.0
    best_val_loss = float("nan")

    if cfg["train"] and trainable > 0:
        start = time.perf_counter()
        meta = train_model(
            model,
            train_seq,
            val_seq,
            epochs=int(cfg["epochs"]),
            lr=float(cfg["lr"]),
            weight_decay=float(cfg["weight_decay"]),
            batch_size=int(cfg["batch_size"]),
            device=device,
            tgo_weight=0.0,
            detach_state_steps=False,
            warmup_steps=0,
            patience=30,
        )
        train_time = time.perf_counter() - start
        history = meta["history"]
        best_epoch = int(meta["best_epoch"])
        best_val_loss = float(meta["best_val_loss"])
        print(
            f"Trained {meta['epochs_run']} epochs in {train_time:.0f}s | "
            f"best_epoch={best_epoch} best_val_loss={best_val_loss:.3f}",
            flush=True,
        )
    else:
        model.to(device)
        model.eval()
        print("Skipping training (no trainable params or train disabled).", flush=True)

    _, val_roll, _ = pooled_rollout_metrics_fullstate(model, val_seq)
    _, test_roll, test_per_roast = pooled_rollout_metrics_fullstate(model, test_seq)
    ci_lo, ci_hi = roast_bootstrap_rollout_r2(test_per_roast, n_boot=1000, seed=args.seed)

    result = {
        "variant": args.variant,
        "seed": args.seed,
        "config": cfg,
        "trainable_params": int(trainable),
        "total_params": int(total),
        "training": {
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "wall_time_sec": train_time,
            "history": history,
        },
        "val": {
            "rollout_r2": float(val_roll.r2),
            "rollout_rmse": float(val_roll.rmse),
        },
        "test": {
            "rollout_r2": float(test_roll.r2),
            "rollout_rmse": float(test_roll.rmse),
            "rollout_r2_ci95": [float(ci_lo), float(ci_hi)],
        },
        "per_roast_test_r2": {
            rid: float(
                1.0
                - np.sum((np.asarray(p["actual"]) - np.asarray(p["rollout_pred"])) ** 2)
                / max(np.sum((np.asarray(p["actual"]) - np.mean(p["actual"])) ** 2), 1e-12)
            )
            for rid, p in test_per_roast.items()
        },
    }
    outpath = ROOT / "reports" / "manuscript_hpo" / f"priors_probe_{args.variant}.json"
    outpath.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        f"\n[{args.variant}] test rollout R^2: {test_roll.r2:.4f}  "
        f"CI95=[{ci_lo:.4f}, {ci_hi:.4f}]  RMSE={test_roll.rmse:.2f}\n"
        f"Saved to {outpath}",
        flush=True,
    )


if __name__ == "__main__":
    main()
