#!/usr/bin/env python
"""Regenerate seed-11 per-roast test trajectories for the three model classes
whose final-eval JSONs only stored per-roast R^2 scalars (multi-closure PI,
bounded FF residual, unbounded FF residual).

The original retrain scripts (retrain_multi_closure_final.py,
retrain_residual_ff_final.py) wrote per_roast_r2 but not the raw actual /
rollout-pred series needed for Figure 5 (representative rollouts). This
script reproduces those seed-11 trainings and dumps the trajectories.

Reads the winning configs from each class's best_config.json (NOT the
*_final.json, which mirrors them but lacks the full schema) and writes a
single consolidated JSON at
``reports/manuscript_hpo/seed11_rollouts.json`` containing the per-roast
trajectories for each model class.

Trajectories for the mechanistic baseline, single-closure PI, and matched-
input neural baseline are already in final_test_metrics.json (per_seed.11);
this script merges those into the output so Figure 5 reads from one place.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from roaster_piml.thesis_full_state import (  # noqa: E402
    LearnedHeFullStateModel,
    MultiClosureFullStateModel,
    set_seed,
    train_model as train_fullstate_model,
)
from roaster_piml.thesis_residual import (  # noqa: E402
    ResidualFeedForwardModel,
    train_residual_model,
)
from scripts.tune_manuscript_models import (  # noqa: E402
    load_manuscript_cohort,
    split_cohort,
    pooled_rollout_metrics_fullstate,
    pooled_rollout_metrics_residual,
)


HPO_DIR = ROOT / "reports" / "manuscript_hpo"
CHECKPOINTS_DIR = HPO_DIR / "checkpoints"
INPUT_CSV = ROOT / "data" / "processed" / "roast_timeseries_p2_only.csv"

OUTPATH = HPO_DIR / "seed11_rollouts.json"

SEED = 11


def _train_multi_closure(cfg: dict, train_seq, val_seq, test_seq, device: str):
    set_seed(SEED)
    model = MultiClosureFullStateModel(
        he_hidden_widths=tuple(cfg["he_hidden_widths"]),
        moisture_hidden_widths=tuple(cfg["moisture_hidden_widths"]),
        reaction_hidden_widths=tuple(cfg["reaction_hidden_widths"]),
    )
    train_fullstate_model(
        model, train_seq, val_seq,
        epochs=300, lr=float(cfg["lr"]),
        weight_decay=float(cfg["weight_decay"]),
        batch_size=int(cfg["batch_size"]),
        device=device,
        tgo_weight=float(cfg["tgo_weight"]),
        detach_state_steps=bool(cfg["detach_state_steps"]),
        warmup_steps=int(cfg["warmup_steps"]),
        patience=30,
    )
    _, _, per_roast = pooled_rollout_metrics_fullstate(model, test_seq)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return per_roast


def _train_residual(cfg: dict, base_model, train_seq, val_seq, test_seq, device: str):
    set_seed(SEED)
    residual = ResidualFeedForwardModel(
        hidden_widths=tuple(cfg["hidden_widths"]),
        max_delta=float(cfg["max_delta"]),
    )
    train_residual_model(
        residual, base_model, train_seq, val_seq,
        epochs=200, lr=float(cfg["lr"]),
        weight_decay=float(cfg["weight_decay"]),
        batch_size=int(cfg["batch_size"]),
        device=device,
        warmup_steps=int(cfg["warmup_steps"]),
        residual_weight=float(cfg["residual_weight"]),
        patience=25,
    )
    _, _, per_roast = pooled_rollout_metrics_residual(residual, base_model, test_seq)
    del residual
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return per_roast


def _load_greybox_seed11(device: str) -> LearnedHeFullStateModel:
    path = CHECKPOINTS_DIR / "greybox_seed11.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing seed-11 greybox checkpoint at {path}; the FF residual "
            "trainings stack on this base."
        )
    payload = torch.load(path, map_location=device)
    model = LearnedHeFullStateModel(hidden_widths=tuple(payload["hidden_widths"]))
    model.load_state_dict(payload["state_dict"])
    model.to(torch.device(device))
    model.eval()
    return model


def _per_roast_to_lists(per_roast: dict) -> dict:
    out = {}
    for rid, d in per_roast.items():
        out[rid] = {
            "actual": list(map(float, d["actual"])),
            "rollout_pred": list(map(float, d["rollout_pred"])),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INPUT_CSV)
    parser.add_argument("--skip-multi", action="store_true")
    parser.add_argument("--skip-bounded", action="store_true")
    parser.add_argument("--skip-unbounded", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    sequences = load_manuscript_cohort(args.input, None, ["P10", "P13", "P16"], ["p2"])
    train_seq, val_seq, test_seq = split_cohort(sequences)
    print(f"Cohort: train={len(train_seq)} val={len(val_seq)} test={len(test_seq)}")

    # ----------------------------- Existing trajectories -----------------------------
    final = json.loads((HPO_DIR / "final_test_metrics.json").read_text(encoding="utf-8"))
    existing = final["per_seed"]["11"]
    out = {
        "seed": SEED,
        "test_roast_ids": list(existing["whitebox_constant_he_fullstate"]["per_roast"].keys()),
        "models": {
            "mechanistic": _per_roast_to_lists(existing["whitebox_constant_he_fullstate"]["per_roast"]),
            "pi_single": _per_roast_to_lists(existing["greybox_learned_he_fullstate"]["per_roast"]),
            "nn_baseline": _per_roast_to_lists(existing["blackbox_core"]["per_roast"]),
        },
    }

    # ----------------------------- Multi-closure PI -----------------------------
    if not args.skip_multi:
        cfg = json.loads((HPO_DIR / "multi_closure" / "best_config.json").read_text(encoding="utf-8"))["config"]
        print("\n=== Multi-closure PI (seed 11) ===")
        print(json.dumps(cfg, indent=2))
        t0 = time.perf_counter()
        per = _train_multi_closure(cfg, train_seq, val_seq, test_seq, device)
        print(f"  multi-closure done in {time.perf_counter() - t0:.0f}s")
        out["models"]["multi_closure"] = _per_roast_to_lists(per)

    # ----------------------------- Bounded FF residual -----------------------------
    if not args.skip_bounded:
        cfg = json.loads((HPO_DIR / "residual_ff" / "best_config.json").read_text(encoding="utf-8"))["config"]
        print("\n=== Bounded FF residual (seed 11) ===")
        print(json.dumps(cfg, indent=2))
        base = _load_greybox_seed11(device)
        t0 = time.perf_counter()
        per = _train_residual(cfg, base, train_seq, val_seq, test_seq, device)
        print(f"  bounded FF residual done in {time.perf_counter() - t0:.0f}s")
        out["models"]["residual_bounded"] = _per_roast_to_lists(per)
        del base
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ----------------------------- Unbounded FF residual -----------------------------
    if not args.skip_unbounded:
        cfg = json.loads((HPO_DIR / "residual_ff_unbounded" / "best_config.json").read_text(encoding="utf-8"))["config"]
        print("\n=== Unbounded FF residual (seed 11) ===")
        print(json.dumps(cfg, indent=2))
        base = _load_greybox_seed11(device)
        t0 = time.perf_counter()
        per = _train_residual(cfg, base, train_seq, val_seq, test_seq, device)
        print(f"  unbounded FF residual done in {time.perf_counter() - t0:.0f}s")
        out["models"]["residual_unbounded"] = _per_roast_to_lists(per)
        del base
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    OUTPATH.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nWrote {OUTPATH}  (models: {list(out['models'].keys())})")


if __name__ == "__main__":
    main()
