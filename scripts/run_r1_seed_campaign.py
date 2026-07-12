#!/usr/bin/env python
"""Round-1 review seed campaign (R1.2 seed expansion + R3.1 diagnostics).

Extends the manuscript's three retraining seeds (11/23/37) with 41/53 for
all six model classes, and runs an instrumented probe family on the
mechanistic-baseline question (R3.1). Every run logs best-epoch scalar
parameter values and train-set rollout R^2 in addition to the usual
val/test metrics — the missing pieces for the reviewer response's
learned-parameter table.

Writes ONLY to reports/r1_seed_campaign/ (existing manuscript artifacts
are never touched). Greybox checkpoints for new seeds are saved under
reports/r1_seed_campaign/checkpoints/ for residual stacking.

Probe variants (ConstantHe scaffold, probe-standard training config):
    baseline           Manuscript mechanistic baseline (whitebox winner cfg).
    control            Priors fixed (20 C / 0.12), init_net frozen.
    control_with_init  Priors fixed, init_net trainable (decomposition).
    globals_only       No priors, init_net frozen: global-estimate-only
                       baseline (reviewer-response variant).
    free_correct_init  No priors, all trainable, but initial-state scalars
                       re-initialized to the physically correct values
                       (offset = 20 - mean train Tc[0]; xb logit = ln 5).

Phases:
    probes   Probe family x --probe-seeds (default 11,23,37,41,53).
    classes  greybox, multi_closure, residual_ff (bounded+unbounded),
             blackbox x --class-seeds (default 41,53).
    all      Both.

Smoke test:  python scripts/run_r1_seed_campaign.py --phase probes \
                 --probe-variants baseline --probe-seeds 41
"""

from __future__ import annotations

import argparse
import json
import math
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
    MultiClosureFullStateModel,
    set_seed,
    train_model as train_fullstate_model,
)
from roaster_piml.thesis_residual import (  # noqa: E402
    ResidualFeedForwardModel,
    train_residual_model,
)
from scripts.tune_manuscript_models import (  # noqa: E402
    CORE_FEATURES,
    CohortIds,
    load_grouped_dataframe,
    load_manuscript_cohort,
    pooled_rollout_metrics_fullstate,
    pooled_rollout_metrics_residual,
    retrain_and_evaluate_blackbox,
    roast_bootstrap_rollout_r2,
    split_cohort,
)

DEFAULT_INPUT = ROOT / "data" / "processed" / "roast_timeseries_p2_only.csv"
HPO_DIR = ROOT / "reports" / "manuscript_hpo"
OUTDIR = ROOT / "reports" / "r1_seed_campaign"
CKPT_DIR = OUTDIR / "checkpoints"

PROBE_TRAIN = dict(lr=5.4e-4, weight_decay=1e-5, batch_size=16, epochs=300,
                   patience=30, tgo_weight=0.0, detach_state_steps=False,
                   warmup_steps=0)

FINAL_BUDGET = {  # manuscript-canonical budgets
    "greybox": dict(epochs=300, patience=30),
    "multi_closure": dict(epochs=300, patience=30),
    "residual_ff": dict(epochs=200, patience=25),
    "blackbox": dict(epochs=200, patience=25),
    "whitebox": dict(epochs=300, patience=30),
}


def _best_config(model_class: str) -> dict:
    path = HPO_DIR / model_class / "best_config.json"
    cfg = dict(json.loads(path.read_text(encoding="utf-8"))["config"])
    budget = FINAL_BUDGET.get(model_class)
    if budget:
        cfg.update(budget)
    return cfg


def extract_scalar_params(model: torch.nn.Module) -> dict:
    """Raw values of all 0-dim parameters plus derived physical quantities."""
    raw = {
        name: float(p.detach().cpu())
        for name, p in model.named_parameters()
        if p.ndim == 0
    }
    derived = {}
    for name, val in raw.items():
        if name.startswith("log_") and "probe" not in name:
            derived[name.replace("log_", "") + "_(exp)"] = float(math.exp(val))
    if "log_probe_k" in raw:
        derived["probe_k_(softplus)"] = float(math.log1p(math.exp(raw["log_probe_k"])))
    if "init_xb_logit" in raw:
        derived["xb0_global"] = float(0.02 + 0.12 / (1.0 + math.exp(-raw["init_xb_logit"])))
    return {"raw": raw, "derived": derived}


def _metrics_dict(m) -> dict:
    return {"r2": float(m.r2), "rmse": float(m.rmse), "mae": float(m.mae), "n": int(m.n)}


def _first_nan_epoch(history: list[dict]) -> int | None:
    for h in history:
        tl = h.get("train_loss")
        if tl is None or (isinstance(tl, float) and math.isnan(tl)):
            return int(h["epoch"])
    return None


def eval_fullstate(model, train_seq, val_seq, test_seq, seed: int) -> dict:
    _, train_roll, _ = pooled_rollout_metrics_fullstate(model, train_seq)
    _, val_roll, _ = pooled_rollout_metrics_fullstate(model, val_seq)
    _, test_roll, test_per_roast = pooled_rollout_metrics_fullstate(model, test_seq)
    ci_lo, ci_hi = roast_bootstrap_rollout_r2(test_per_roast, n_boot=1000, seed=seed)
    return {
        "train_rollout": _metrics_dict(train_roll),
        "val_rollout": _metrics_dict(val_roll),
        "test_rollout": _metrics_dict(test_roll),
        "test_rollout_r2_ci95": [float(ci_lo), float(ci_hi)],
        "per_roast_test_r2": {
            rid: float(
                1.0
                - np.sum((np.asarray(p["actual"]) - np.asarray(p["rollout_pred"])) ** 2)
                / max(np.sum((np.asarray(p["actual"]) - np.mean(p["actual"])) ** 2), 1e-12)
            )
            for rid, p in test_per_roast.items()
        },
    }


# ---------------------------------------------------------------- probes
def build_probe_model(variant: str, wb_cfg: dict, train_seq) -> tuple[torch.nn.Module, dict]:
    """Return (model, train_cfg) for a probe variant."""
    if variant == "baseline":
        model = ConstantHeFullStateModel()
        train_cfg = {k: wb_cfg[k] for k in
                     ("lr", "weight_decay", "batch_size", "epochs", "patience",
                      "tgo_weight", "detach_state_steps", "warmup_steps")}
        return model, train_cfg

    train_cfg = dict(PROBE_TRAIN)
    if variant == "control":
        model = ConstantHeFullStateModel(fixed_initial_bean_temp_c=20.0,
                                         fixed_initial_moisture_ratio=0.12)
        _freeze_init_net(model)
    elif variant == "control_with_init":
        model = ConstantHeFullStateModel(fixed_initial_bean_temp_c=20.0,
                                         fixed_initial_moisture_ratio=0.12)
    elif variant == "globals_only":
        model = ConstantHeFullStateModel()
        _freeze_init_net(model)
    elif variant == "free_correct_init":
        model = ConstantHeFullStateModel()
        mean_tc0 = float(np.mean([s.tc[0] for s in train_seq]))
        with torch.no_grad():
            model.init_tb_offset.fill_(20.0 - mean_tc0)
            model.init_xb_logit.fill_(math.log(5.0))  # sigmoid -> 0.833 -> xb0 = 0.12
    else:
        raise ValueError(f"unknown probe variant {variant!r}")
    return model, train_cfg


def _freeze_init_net(model) -> None:
    for p in model.init_net.parameters():
        p.requires_grad = False


def run_probe(variant: str, seed: int, wb_cfg: dict, train_seq, val_seq, test_seq,
              device: str) -> dict:
    set_seed(seed)
    model, cfg = build_probe_model(variant, wb_cfg, train_seq)
    init_params = extract_scalar_params(model)
    trainable = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    start = time.perf_counter()
    meta = train_fullstate_model(
        model, train_seq, val_seq,
        epochs=int(cfg["epochs"]), lr=float(cfg["lr"]),
        weight_decay=float(cfg["weight_decay"]), batch_size=int(cfg["batch_size"]),
        device=device, tgo_weight=float(cfg["tgo_weight"]),
        detach_state_steps=bool(cfg["detach_state_steps"]),
        warmup_steps=int(cfg["warmup_steps"]), patience=int(cfg["patience"]),
    )
    wall = time.perf_counter() - start
    model.eval()
    result = {
        "variant": variant,
        "seed": seed,
        "trainable_params": trainable,
        "train_cfg": cfg,
        "initial_scalar_params": init_params,
        "best_epoch_scalar_params": extract_scalar_params(model),
        "training": {
            "best_epoch": int(meta["best_epoch"]),
            "best_val_loss": float(meta["best_val_loss"]),
            "epochs_run": int(meta["epochs_run"]),
            "first_nan_epoch": _first_nan_epoch(meta["history"]),
            "wall_time_sec": wall,
            "history": meta["history"],
        },
        **eval_fullstate(model, train_seq, val_seq, test_seq, seed),
    }
    return result


# ---------------------------------------------------------------- classes
def run_greybox(seed: int, train_seq, val_seq, test_seq, device: str) -> dict:
    cfg = _best_config("greybox")
    set_seed(seed)
    model = LearnedHeFullStateModel(hidden_widths=tuple(cfg["hidden_widths"]))
    start = time.perf_counter()
    meta = train_fullstate_model(
        model, train_seq, val_seq,
        epochs=int(cfg["epochs"]), lr=float(cfg["lr"]),
        weight_decay=float(cfg["weight_decay"]), batch_size=int(cfg["batch_size"]),
        device=device, tgo_weight=float(cfg["tgo_weight"]),
        detach_state_steps=bool(cfg["detach_state_steps"]),
        warmup_steps=int(cfg["warmup_steps"]), patience=int(cfg["patience"]),
    )
    wall = time.perf_counter() - start
    model.eval()
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": model.state_dict(), "hidden_widths": tuple(cfg["hidden_widths"]),
         "config": cfg, "seed": seed},
        CKPT_DIR / f"greybox_seed{seed}.pt",
    )
    return {
        "seed": seed, "config": cfg,
        "trainable_params": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "best_epoch_scalar_params": extract_scalar_params(model),
        "training": {"best_epoch": int(meta["best_epoch"]),
                     "best_val_loss": float(meta["best_val_loss"]),
                     "epochs_run": int(meta["epochs_run"]),
                     "first_nan_epoch": _first_nan_epoch(meta["history"]),
                     "wall_time_sec": wall, "history": meta["history"]},
        **eval_fullstate(model, train_seq, val_seq, test_seq, seed),
    }


def run_multi_closure(seed: int, train_seq, val_seq, test_seq, device: str) -> dict:
    cfg = _best_config("multi_closure")
    set_seed(seed)
    model = MultiClosureFullStateModel(
        he_hidden_widths=tuple(cfg["he_hidden_widths"]),
        moisture_hidden_widths=tuple(cfg["moisture_hidden_widths"]),
        reaction_hidden_widths=tuple(cfg["reaction_hidden_widths"]),
    )
    start = time.perf_counter()
    meta = train_fullstate_model(
        model, train_seq, val_seq,
        epochs=int(cfg["epochs"]), lr=float(cfg["lr"]),
        weight_decay=float(cfg["weight_decay"]), batch_size=int(cfg["batch_size"]),
        device=device, tgo_weight=float(cfg["tgo_weight"]),
        detach_state_steps=bool(cfg["detach_state_steps"]),
        warmup_steps=int(cfg["warmup_steps"]), patience=int(cfg["patience"]),
    )
    wall = time.perf_counter() - start
    model.eval()
    return {
        "seed": seed, "config": cfg,
        "trainable_params": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "best_epoch_scalar_params": extract_scalar_params(model),
        "training": {"best_epoch": int(meta["best_epoch"]),
                     "best_val_loss": float(meta["best_val_loss"]),
                     "epochs_run": int(meta["epochs_run"]),
                     "first_nan_epoch": _first_nan_epoch(meta["history"]),
                     "wall_time_sec": wall, "history": meta["history"]},
        **eval_fullstate(model, train_seq, val_seq, test_seq, seed),
    }


def run_residual_ff(kind: str, seed: int, train_seq, val_seq, test_seq, device: str) -> dict:
    cfg = _best_config("residual_ff" if kind == "bounded" else "residual_ff_unbounded")
    ckpt = CKPT_DIR / f"greybox_seed{seed}.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"greybox checkpoint for seed {seed} missing: {ckpt}")
    payload = torch.load(ckpt, map_location=device)
    base = LearnedHeFullStateModel(hidden_widths=tuple(payload["hidden_widths"]))
    base.load_state_dict(payload["state_dict"])
    base.to(torch.device(device))
    base.eval()

    set_seed(seed)
    residual = ResidualFeedForwardModel(
        hidden_widths=tuple(cfg["hidden_widths"]),
        max_delta=float(cfg["max_delta"]),
    )
    start = time.perf_counter()
    meta = train_residual_model(
        residual, base, train_seq, val_seq,
        epochs=int(cfg["epochs"]), lr=float(cfg["lr"]),
        weight_decay=float(cfg["weight_decay"]), batch_size=int(cfg["batch_size"]),
        device=device, warmup_steps=int(cfg["warmup_steps"]),
        residual_weight=float(cfg["residual_weight"]), patience=int(cfg["patience"]),
    )
    wall = time.perf_counter() - start
    _, train_roll, _ = pooled_rollout_metrics_residual(residual, base, train_seq)
    _, val_roll, _ = pooled_rollout_metrics_residual(residual, base, val_seq)
    _, test_roll, test_per_roast = pooled_rollout_metrics_residual(residual, base, test_seq)
    ci_lo, ci_hi = roast_bootstrap_rollout_r2(test_per_roast, n_boot=1000, seed=seed)
    return {
        "seed": seed, "kind": kind, "config": cfg,
        "trainable_params": int(sum(p.numel() for p in residual.parameters() if p.requires_grad)),
        "training": {"best_epoch": int(meta["best_epoch"]),
                     "best_val_loss": float(meta["best_val_loss"]),
                     "epochs_run": int(meta["epochs_run"]),
                     "first_nan_epoch": _first_nan_epoch(meta["history"]),
                     "wall_time_sec": wall, "history": meta["history"]},
        "train_rollout": _metrics_dict(train_roll),
        "val_rollout": _metrics_dict(val_roll),
        "test_rollout": _metrics_dict(test_roll),
        "test_rollout_r2_ci95": [float(ci_lo), float(ci_hi)],
        "per_roast_test_r2": {
            rid: float(
                1.0
                - np.sum((np.asarray(p["actual"]) - np.asarray(p["rollout_pred"])) ** 2)
                / max(np.sum((np.asarray(p["actual"]) - np.mean(p["actual"])) ** 2), 1e-12)
            )
            for rid, p in test_per_roast.items()
        },
    }


def _save(name: str, obj: dict) -> Path:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    path = OUTDIR / f"{name}.json"
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--phase", choices=["probes", "classes", "all"], default="all")
    parser.add_argument("--probe-seeds", type=str, default="11,23,37,41,53")
    parser.add_argument("--class-seeds", type=str, default="41,53")
    parser.add_argument(
        "--probe-variants", type=str,
        default="baseline,control,control_with_init,globals_only,free_correct_init")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    probe_seeds = [int(s) for s in args.probe_seeds.split(",") if s.strip()]
    class_seeds = [int(s) for s in args.class_seeds.split(",") if s.strip()]
    variants = [v.strip() for v in args.probe_variants.split(",") if v.strip()]

    print(f"Device: {args.device}", flush=True)
    sequences = load_manuscript_cohort(args.input, None, ["P10", "P13", "P16"], ["p2"])
    train_seq, val_seq, test_seq = split_cohort(sequences)
    cohort_ids = CohortIds(
        train_ids=[s.roast_id for s in train_seq],
        val_ids=[s.roast_id for s in val_seq],
        test_ids=[s.roast_id for s in test_seq],
    )
    print(f"Cohort: total={len(sequences)} train={len(train_seq)} "
          f"val={len(val_seq)} test={len(test_seq)}", flush=True)

    wb_cfg = _best_config("whitebox")

    if args.phase in ("probes", "all"):
        for variant in variants:
            for seed in probe_seeds:
                tag = f"probe_{variant}_seed{seed}"
                out = OUTDIR / f"{tag}.json"
                if out.exists():
                    print(f"[skip] {tag} (exists)", flush=True)
                    continue
                print(f"[run ] {tag}", flush=True)
                res = run_probe(variant, seed, wb_cfg, train_seq, val_seq,
                                test_seq, args.device)
                _save(tag, res)
                print(f"       test R2={res['test_rollout']['r2']:+.4f} "
                      f"train R2={res['train_rollout']['r2']:+.4f} "
                      f"best_ep={res['training']['best_epoch']} "
                      f"nan_ep={res['training']['first_nan_epoch']}", flush=True)

    if args.phase in ("classes", "all"):
        grouped = load_grouped_dataframe(args.input, [s.roast_id for s in sequences],
                                         CORE_FEATURES)
        bb_cfg = _best_config("blackbox")
        for seed in class_seeds:
            for name, fn in (
                ("greybox", lambda s: run_greybox(s, train_seq, val_seq, test_seq, args.device)),
                ("multi_closure", lambda s: run_multi_closure(s, train_seq, val_seq, test_seq, args.device)),
                ("residual_ff_bounded", lambda s: run_residual_ff("bounded", s, train_seq, val_seq, test_seq, args.device)),
                ("residual_ff_unbounded", lambda s: run_residual_ff("unbounded", s, train_seq, val_seq, test_seq, args.device)),
                ("blackbox", lambda s: retrain_and_evaluate_blackbox(
                    bb_cfg, s, cohort_ids, grouped, CORE_FEATURES, args.device)),
            ):
                tag = f"class_{name}_seed{seed}"
                out = OUTDIR / f"{tag}.json"
                if out.exists():
                    print(f"[skip] {tag} (exists)", flush=True)
                    continue
                print(f"[run ] {tag}", flush=True)
                res = fn(seed)
                _save(tag, res)
                r2 = (res.get("test_rollout") or res.get("rollout_metrics", {})).get("r2")
                print(f"       test rollout R2={r2:+.4f}" if r2 is not None else "       done",
                      flush=True)

    print("\nCampaign phase complete.", flush=True)


if __name__ == "__main__":
    main()
