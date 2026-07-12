#!/usr/bin/env python
"""Compute 50/50 ensemble results for the manuscript ensemble discussion.

Loads per-roast step-level rollout predictions from seed11_rollouts.json and
computes two equal-weight ensembles:
  - multi_closure + nn_baseline   (14,196 + 705 = 14,901 params)
  - residual_unbounded + nn_baseline (2,433 + 705 = 3,138 params)

Equal weights are used because both partners achieve similar R² on held-out
roasts and no additional tuning is applied; the goal is to confirm that the
near-zero per-roast error correlation translates into a measurable variance
reduction, not to optimise the combination.

For each ensemble reports:
  - pooled rollout R² with 95% bootstrap CI (n_boot=1000, seed=11)
  - per-roast R² vector
  - Pearson r between the ensemble's per-roast R² and each parent's
  - combined parameter count

Writes reports/manuscript_hpo/ensemble_analysis.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

HPO_DIR = ROOT / "reports" / "manuscript_hpo"
ROLLOUTS_PATH = HPO_DIR / "seed11_rollouts.json"
OUTPATH = HPO_DIR / "ensemble_analysis.json"

# Trainable parameter counts from Table 2 of the manuscript
PARAM_COUNTS = {
    "multi_closure": 14_196,
    "residual_unbounded": 2_433,
    "nn_baseline": 705,
}

ENSEMBLES = [
    {
        "name": "multi_closure_plus_nn",
        "label": "Multi-closure PI + Neural baseline",
        "model_a": "multi_closure",
        "model_b": "nn_baseline",
    },
    {
        "name": "residual_unbounded_plus_nn",
        "label": "Unbounded FF residual + Neural baseline",
        "model_a": "residual_unbounded",
        "model_b": "nn_baseline",
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _r2_pooled(actuals: list[np.ndarray], preds: list[np.ndarray]) -> float:
    """Pooled (cohort-level) R² across all roasts."""
    all_a = np.concatenate(actuals)
    all_p = np.concatenate(preds)
    ss_res = np.sum((all_a - all_p) ** 2)
    ss_tot = np.sum((all_a - all_a.mean()) ** 2)
    if ss_tot < 1e-12:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def _r2_per_roast(actual: np.ndarray, pred: np.ndarray) -> float:
    ss_res = np.sum((actual - pred) ** 2)
    ss_tot = np.sum((actual - actual.mean()) ** 2)
    if ss_tot < 1e-12:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def _bootstrap_ci(
    actuals: list[np.ndarray],
    preds: list[np.ndarray],
    n_boot: int = 1000,
    seed: int = 11,
    ci: float = 0.95,
) -> tuple[float, float]:
    """95% roast-bootstrap CI for pooled R², matching the paper's method."""
    rng = np.random.default_rng(seed)
    n = len(actuals)
    boot_r2 = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        a = [actuals[i] for i in idx]
        p = [preds[i] for i in idx]
        boot_r2.append(_r2_pooled(a, p))
    boot_r2 = np.array(boot_r2)
    lo = float(np.percentile(boot_r2, 100 * (1 - ci) / 2))
    hi = float(np.percentile(boot_r2, 100 * (1 + ci) / 2))
    return lo, hi


def _pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return float("nan")
    return float(np.corrcoef(x[mask], y[mask])[0, 1])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    data = json.loads(ROLLOUTS_PATH.read_text(encoding="utf-8"))
    roast_ids: list[str] = data["test_roast_ids"]
    models: dict[str, dict] = data["models"]

    print(f"Loaded seed11_rollouts.json: {len(roast_ids)} roasts, "
          f"models={list(models.keys())}")

    # Pre-build per-roast arrays for every model we need
    arrays: dict[str, dict[str, np.ndarray]] = {}
    for model_name, roast_dict in models.items():
        arrays[model_name] = {}
        for rid in roast_ids:
            entry = roast_dict[rid]
            arrays[model_name][rid] = {
                "actual": np.asarray(entry["actual"], dtype=float),
                "pred":   np.asarray(entry["rollout_pred"], dtype=float),
            }

    # Per-roast R² for each individual model (needed for correlation check)
    per_roast_r2: dict[str, dict[str, float]] = {}
    for model_name in models:
        per_roast_r2[model_name] = {
            rid: _r2_per_roast(
                arrays[model_name][rid]["actual"],
                arrays[model_name][rid]["pred"],
            )
            for rid in roast_ids
        }

    results: dict[str, dict] = {}

    for ens in ENSEMBLES:
        name = ens["name"]
        ma, mb = ens["model_a"], ens["model_b"]
        print(f"\n=== {ens['label']} ===")

        # 50/50 average of step-level predictions
        actuals, ens_preds = [], []
        ens_per_roast_r2: dict[str, float] = {}
        for rid in roast_ids:
            a = arrays[ma][rid]["actual"]
            pa = arrays[ma][rid]["pred"]
            pb = arrays[mb][rid]["pred"]
            assert np.array_equal(a, arrays[mb][rid]["actual"]), \
                f"Actual mismatch for roast {rid}"
            ens_pred = 0.5 * pa + 0.5 * pb
            actuals.append(a)
            ens_preds.append(ens_pred)
            ens_per_roast_r2[rid] = _r2_per_roast(a, ens_pred)

        pooled = _r2_pooled(actuals, ens_preds)
        ci_lo, ci_hi = _bootstrap_ci(actuals, ens_preds, n_boot=1000, seed=11)

        # Per-roast Pearson r between ensemble and each parent
        ens_vec = np.array([ens_per_roast_r2[r] for r in roast_ids])
        vec_a   = np.array([per_roast_r2[ma][r]  for r in roast_ids])
        vec_b   = np.array([per_roast_r2[mb][r]  for r in roast_ids])

        r_with_a = _pearson_r(ens_vec, vec_a)
        r_with_b = _pearson_r(ens_vec, vec_b)

        param_count = PARAM_COUNTS[ma] + PARAM_COUNTS[mb]

        print(f"  Pooled R²       : {pooled:.4f}  CI95=[{ci_lo:.4f}, {ci_hi:.4f}]")
        print(f"  Parent A ({ma:>20s}): R²={_r2_pooled(actuals, [arrays[ma][r]['pred'] for r in roast_ids]):.4f}  "
              f"r_with_ensemble={r_with_a:+.3f}")
        print(f"  Parent B ({mb:>20s}): R²={_r2_pooled(actuals, [arrays[mb][r]['pred'] for r in roast_ids]):.4f}  "
              f"r_with_ensemble={r_with_b:+.3f}")
        print(f"  Combined params : {param_count:,}")

        results[name] = {
            "label": ens["label"],
            "model_a": ma,
            "model_b": mb,
            "pooled_r2": pooled,
            "ci95": [ci_lo, ci_hi],
            "param_count": param_count,
            "per_roast_r2": ens_per_roast_r2,
            "r_with_model_a": r_with_a,
            "r_with_model_b": r_with_b,
            "parent_pooled_r2": {
                ma: _r2_pooled(actuals, [arrays[ma][r]["pred"] for r in roast_ids]),
                mb: _r2_pooled(actuals, [arrays[mb][r]["pred"] for r in roast_ids]),
            },
        }

    OUTPATH.write_text(json.dumps({"ensembles": results}, indent=2), encoding="utf-8")
    print(f"\nSaved to {OUTPATH}")


if __name__ == "__main__":
    main()
