#!/usr/bin/env python
"""Run the placement-spectrum study on the synthetic benchmark cohort.

Retrains the manuscript-winning configuration of four model classes
(mechanistic baseline, single-closure PI, multi-closure PI, matched-input
neural baseline) at the manuscript-canonical final budget on the synthetic
cohort produced by scripts/generate_synthetic_fixture.py, and reports held-out
rollout R^2 per class. The purpose is to check that the synthetic benchmark
reproduces the qualitative placement ordering (baseline fails, single-closure
partial repair, multi-closure strong repair, neural baseline strongest); the
numbers are properties of the fixture, not of the industrial cohort.

Usage:
    python scripts/run_synthetic_study.py [--seed 11] [--device cuda]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.tune_manuscript_models import (  # noqa: E402
    CORE_FEATURES,
    CohortIds,
    _promote_to_final_budget,
    filter_by_id,
    load_grouped_dataframe,
    load_manuscript_cohort,
    retrain_and_evaluate_blackbox,
    retrain_and_evaluate_fullstate,
    split_cohort,
)
from roaster_piml.thesis_full_state import (  # noqa: E402
    ConstantHeFullStateModel,
    LearnedHeFullStateModel,
    MultiClosureFullStateModel,
)

INPUT = ROOT / "data" / "synthetic" / "roast_timeseries_synthetic_p2_only.csv"
OUTDIR = ROOT / "reports" / "synthetic_study"


def best_config(model_class: str) -> dict:
    path = ROOT / "reports" / "manuscript_hpo" / model_class / "best_config.json"
    return json.loads(path.read_text(encoding="utf-8"))["config"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--classes", type=str,
                        default="whitebox,greybox,multi_closure,blackbox",
                        help="comma-separated subset of classes to run")
    args = parser.parse_args()
    wanted = {c.strip() for c in args.classes.split(",")}
    OUTDIR.mkdir(parents=True, exist_ok=True)

    sequences = load_manuscript_cohort(INPUT, None, ["P10", "P13", "P16"], ["p2"])
    train_seq, val_seq, test_seq = split_cohort(sequences)
    ids = CohortIds(
        [s.roast_id for s in train_seq],
        [s.roast_id for s in val_seq],
        [s.roast_id for s in test_seq],
    )
    print(f"synthetic cohort: {len(sequences)} roasts "
          f"({len(train_seq)}/{len(val_seq)}/{len(test_seq)})", flush=True)

    out = OUTDIR / f"seed{args.seed}_results.json"
    results: dict[str, dict] = (
        json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    )

    def factory_whitebox():
        return ConstantHeFullStateModel()

    def factory_greybox():
        cfg = best_config("greybox")
        return LearnedHeFullStateModel(hidden_widths=tuple(cfg["hidden_widths"]))

    def factory_multi():
        cfg = best_config("multi_closure")
        return MultiClosureFullStateModel(
            he_hidden_widths=tuple(cfg["he_hidden_widths"]),
            moisture_hidden_widths=tuple(cfg["moisture_hidden_widths"]),
            reaction_hidden_widths=tuple(cfg["reaction_hidden_widths"]),
        )

    for name, factory in [("whitebox", factory_whitebox),
                          ("greybox", factory_greybox),
                          ("multi_closure", factory_multi)]:
        if name not in wanted:
            continue
        cfg = _promote_to_final_budget(best_config(name), name)
        t0 = time.perf_counter()
        print(f"[{name}] training at final budget...", flush=True)
        try:
            payload, _model = retrain_and_evaluate_fullstate(
                model_factory=factory, config=cfg, seed=args.seed,
                cohort_ids=ids, train_sequences=train_seq,
                val_sequences=val_seq, test_sequences=test_seq,
                device=args.device,
            )
            payload["training"].pop("history", None)
            payload["wall_time_sec"] = time.perf_counter() - t0
            results[name] = payload
            r2 = payload.get("rollout_metrics", payload).get("r2", None) if isinstance(payload.get("rollout_metrics", None), dict) else payload.get("test_rollout_r2")
            print(f"[{name}] done in {payload['wall_time_sec']:.0f}s; payload keys: {list(payload)[:8]}", flush=True)
        except Exception as exc:  # divergence is an expected outcome for whitebox
            results[name] = {"error": str(exc), "wall_time_sec": time.perf_counter() - t0}
            print(f"[{name}] FAILED: {exc}", flush=True)
        (OUTDIR / f"seed{args.seed}_results.json").write_text(
            json.dumps(results, indent=1, default=str), encoding="utf-8")

    # Neural baseline
    if "blackbox" not in wanted:
        _summarize(results)
        return
    cfg = _promote_to_final_budget(best_config("blackbox"), "blackbox")
    grouped = load_grouped_dataframe(INPUT, [s.roast_id for s in sequences], CORE_FEATURES)
    t0 = time.perf_counter()
    print("[blackbox] training at final budget...", flush=True)
    try:
        payload = retrain_and_evaluate_blackbox(
            cfg, args.seed, ids, grouped, CORE_FEATURES, args.device)
        if isinstance(payload, tuple):
            payload = payload[0]
        payload["wall_time_sec"] = time.perf_counter() - t0
        results["blackbox"] = payload
        print(f"[blackbox] done in {payload['wall_time_sec']:.0f}s", flush=True)
    except Exception as exc:
        results["blackbox"] = {"error": str(exc), "wall_time_sec": time.perf_counter() - t0}
        print(f"[blackbox] FAILED: {exc}", flush=True)

    out.write_text(json.dumps(results, indent=1, default=str), encoding="utf-8")
    print(f"wrote {out}", flush=True)
    _summarize(results)


def _summarize(results: dict[str, dict]) -> None:
    print("\n=== synthetic-study summary (held-out rollout R^2) ===", flush=True)
    for name, payload in results.items():
        if "error" in payload:
            print(f"{name:14s} ERROR: {payload['error'][:80]}", flush=True)
            continue
        rm = payload.get("rollout_metrics") or payload.get("test", {}).get("rollout_metrics")
        r2 = rm.get("r2") if isinstance(rm, dict) else None
        print(f"{name:14s} R2 = {r2}", flush=True)


if __name__ == "__main__":
    main()
