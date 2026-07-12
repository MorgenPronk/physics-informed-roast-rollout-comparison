#!/usr/bin/env python
"""Statistical-separability analysis across the physics-content spectrum.

Reads the per-roast test rollout R^2 values for every model variant we have
data for, and produces:

    - Paired-Wilcoxon p-values for every pair of models (per-roast diffs).
    - TOST equivalence-test p-values with epsilon=0.02 R^2 (i.e. test the
      null 'pairs differ by > 0.02' against the alt 'pairs equivalent
      within +/- 0.02').
    - Pearson correlation between per-roast R^2 vectors for every pair.
    - Bootstrap 95% CI on the median paired delta.
    - A summary table sorted by mean R^2 with columns for params, compute,
      and the seed-11 test R^2 with its bootstrap CI.

The script reads from final_test_metrics.json (for the four canonical models)
plus any greybox_bs8_probe.json, residual_ff_probe*.json, and
priors_probe_*.json files in reports/manuscript_hpo/. It is idempotent and
safe to run while sweeps are in progress.
"""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
HPO_DIR = ROOT / "reports" / "manuscript_hpo"
OUTDIR = HPO_DIR / "spectrum_stats"


def _per_roast_from_seedpayload(payload: dict[str, Any]) -> dict[str, float]:
    return {rid: float(v) for rid, v in payload.get("per_roast_r2", {}).items()}


def _summary_from_seedpayload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "rollout_r2": float(payload["rollout_metrics"]["r2"]),
        "rollout_rmse": float(payload["rollout_metrics"]["rmse"]),
        "rollout_r2_ci95": [float(x) for x in payload["rollout_r2_ci95"]],
        "param_count": int(payload.get("param_count", 0)),
        "per_roast_r2": _per_roast_from_seedpayload(payload),
    }


def _summary_from_probe(probe: dict[str, Any]) -> dict[str, Any]:
    test = probe["test"]
    return {
        "rollout_r2": float(test["rollout_r2"]),
        "rollout_rmse": float(test.get("rollout_rmse", float("nan"))),
        "rollout_r2_ci95": [float(x) for x in test.get("rollout_r2_ci95", [float("nan"), float("nan")])],
        "param_count": int(probe.get("param_count", probe.get("trainable_params", 0))),
        "per_roast_r2": {rid: float(v) for rid, v in probe.get("per_roast_test_r2", {}).items()},
    }


def load_all_models() -> "OrderedDict[str, dict]":
    """Build an OrderedDict of model_name -> summary_dict from all available sources."""
    models: OrderedDict[str, dict] = OrderedDict()

    # Final-eval main models (seed 11).
    final_path = HPO_DIR / "final_test_metrics.json"
    if final_path.exists():
        final_payload = json.loads(final_path.read_text(encoding="utf-8"))
        seed11 = final_payload.get("per_seed", {}).get("11", {})
        order = [
            ("whitebox_constant_he_fullstate", "mechanistic"),
            ("greybox_learned_he_fullstate", "pi_closure_sweep"),
            ("residual_lstm_on_greybox", "residual_lstm"),
            ("blackbox_core", "nn_baseline"),
        ]
        for key, label in order:
            if key in seed11:
                models[label] = _summary_from_seedpayload(seed11[key])

    # bs=8 PI probe.
    bs8_probe = HPO_DIR / "greybox_bs8_probe.json"
    if bs8_probe.exists():
        models["pi_closure_bs8"] = _summary_from_probe(json.loads(bs8_probe.read_text(encoding="utf-8")))

    # Bounded FF residual probe.
    bounded_ff = HPO_DIR / "residual_ff_probe.json"
    if bounded_ff.exists():
        models["residual_ff_bounded"] = _summary_from_probe(json.loads(bounded_ff.read_text(encoding="utf-8")))

    # Unbounded FF residual probe.
    unbounded_ff = HPO_DIR / "residual_ff_probe_unbounded.json"
    if unbounded_ff.exists():
        models["residual_ff_unbounded"] = _summary_from_probe(json.loads(unbounded_ff.read_text(encoding="utf-8")))

    # Priors probes.
    for variant_name, label in [
        ("true_mechanistic", "true_mechanistic"),
        ("scalar_tuned_priors", "scalar_tuned_priors"),
        ("scalar_tuned_priors_with_init_net", "scalar_tuned_priors_with_init_net"),
        ("pi_fixed_priors", "pi_fixed_priors"),
    ]:
        probe_path = HPO_DIR / f"priors_probe_{variant_name}.json"
        if probe_path.exists():
            models[label] = _summary_from_probe(json.loads(probe_path.read_text(encoding="utf-8")))

    # Multi-closure and FF-residual variants: per-seed final retrains. Pull
    # seed 11 to match the rest of the headline-row reporting.
    for filename, label in [
        ("multi_closure_final.json", "multi_closure_pi"),
        ("residual_ff_final.json", "residual_ff_bounded_sweep"),
        ("residual_ff_unbounded_final.json", "residual_ff_unbounded_sweep"),
    ]:
        path = HPO_DIR / filename
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        seed11 = payload.get("per_seed", {}).get("11")
        if seed11 is None:
            continue
        models[label] = {
            "rollout_r2": float(seed11["rollout_metrics"]["r2"]),
            "rollout_rmse": float(seed11["rollout_metrics"]["rmse"]),
            "rollout_r2_ci95": [float(x) for x in seed11["rollout_r2_ci95"]],
            "param_count": int(seed11["param_count"]),
            "per_roast_r2": {rid: float(v) for rid, v in seed11.get("per_roast_r2", {}).items()},
        }

    return models


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------


def aligned_per_roast(a: dict[str, float], b: dict[str, float]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    keys = sorted(set(a) & set(b))
    av = np.array([a[k] for k in keys], dtype=float)
    bv = np.array([b[k] for k in keys], dtype=float)
    mask = np.isfinite(av) & np.isfinite(bv)
    return av[mask], bv[mask], [k for k, m in zip(keys, mask) if m]


def paired_wilcoxon(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    diff = a - b
    if diff.size < 2 or np.all(diff == 0):
        return {"statistic": float("nan"), "p_value": float("nan"), "n": int(diff.size)}
    try:
        r = stats.wilcoxon(diff, zero_method="wilcox", alternative="two-sided")
        return {"statistic": float(r.statistic), "p_value": float(r.pvalue), "n": int(diff.size)}
    except Exception as e:  # pragma: no cover
        return {"statistic": float("nan"), "p_value": float("nan"), "n": int(diff.size), "error": repr(e)}


def tost_equivalence(a: np.ndarray, b: np.ndarray, epsilon: float = 0.02) -> dict[str, float]:
    """Two One-Sided Tests for equivalence within +/- epsilon, on the mean
    paired difference. Returns the larger of the two one-sided p-values;
    if that p < alpha, conclude equivalence within +/- epsilon."""
    diff = a - b
    n = diff.size
    if n < 3:
        return {"p_value_max": float("nan"), "epsilon": epsilon, "n": int(n)}
    mean = float(np.mean(diff))
    sd = float(np.std(diff, ddof=1))
    if sd == 0:
        # Identical; equivalence trivially holds if |mean| < epsilon.
        equiv = abs(mean) < epsilon
        return {"p_value_max": 0.0 if equiv else 1.0, "epsilon": epsilon, "n": int(n), "mean_diff": mean}
    se = sd / np.sqrt(n)
    # H1: difference > -epsilon
    t1 = (mean - (-epsilon)) / se
    p1 = 1.0 - stats.t.cdf(t1, df=n - 1)
    # H2: difference <  +epsilon
    t2 = (mean - (+epsilon)) / se
    p2 = stats.t.cdf(t2, df=n - 1)
    return {
        "p_value_max": float(max(p1, p2)),
        "epsilon": epsilon,
        "n": int(n),
        "mean_diff": mean,
        "se_diff": float(se),
    }


def bootstrap_median_delta_ci(
    a: np.ndarray, b: np.ndarray, n_boot: int = 2000, seed: int = 0
) -> tuple[float, float, float]:
    if a.size < 2:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    diff = a - b
    medians = []
    for _ in range(n_boot):
        idx = rng.integers(0, diff.size, size=diff.size)
        medians.append(float(np.median(diff[idx])))
    lo, hi = np.quantile(medians, [0.025, 0.975])
    return float(np.median(diff)), float(lo), float(hi)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epsilon", type=float, default=0.02, help="TOST equivalence margin")
    parser.add_argument("--outdir", type=Path, default=OUTDIR)
    args = parser.parse_args()

    models = load_all_models()
    args.outdir.mkdir(parents=True, exist_ok=True)
    print(f"Loaded {len(models)} models: {', '.join(models)}")
    if not models:
        print("No data files found yet.")
        return

    # Summary table sorted by rollout R^2.
    rows = []
    for label, s in models.items():
        rows.append(
            {
                "model": label,
                "rollout_r2": s["rollout_r2"],
                "rollout_rmse": s["rollout_rmse"],
                "ci_lo": s["rollout_r2_ci95"][0],
                "ci_hi": s["rollout_r2_ci95"][1],
                "param_count": s["param_count"],
                "n_per_roast": len(s["per_roast_r2"]),
            }
        )
    rows.sort(key=lambda r: r["rollout_r2"], reverse=True)

    summary_md = ["# Spectrum statistical-separability analysis", "", "## Headline test rollout R²", "",
                  "| Model | R² | RMSE | 95% CI | Params | n roasts |",
                  "|---|---:|---:|---|---:|---:|"]
    for r in rows:
        summary_md.append(
            f"| {r['model']} | {r['rollout_r2']:.4f} | {r['rollout_rmse']:.2f} | "
            f"[{r['ci_lo']:.4f}, {r['ci_hi']:.4f}] | {r['param_count']:,} | {r['n_per_roast']} |"
        )
    summary_md.append("")

    # Pairwise tests.
    pair_records: list[dict[str, Any]] = []
    for label_a, label_b in combinations(models.keys(), 2):
        a_map = models[label_a]["per_roast_r2"]
        b_map = models[label_b]["per_roast_r2"]
        a, b, keys = aligned_per_roast(a_map, b_map)
        if a.size < 3:
            continue
        wilcox = paired_wilcoxon(a, b)
        tost = tost_equivalence(a, b, epsilon=args.epsilon)
        med, lo, hi = bootstrap_median_delta_ci(a, b)
        corr = float(np.corrcoef(a, b)[0, 1]) if a.size >= 2 else float("nan")
        pair_records.append(
            {
                "model_a": label_a,
                "model_b": label_b,
                "n_roasts": int(a.size),
                "median_delta_r2": float(med),
                "median_delta_ci95": [float(lo), float(hi)],
                "mean_delta_r2": float(np.mean(a - b)),
                "wilcoxon_p_value": wilcox["p_value"],
                "tost_p_value_max": tost["p_value_max"],
                "tost_epsilon": args.epsilon,
                "per_roast_corr": corr,
            }
        )

    summary_md.append(f"## Pairwise paired-Wilcoxon + TOST (ε={args.epsilon}) + per-roast correlation")
    summary_md.append("")
    summary_md.append(
        "Reading the columns:\n\n"
        "- **median Δ**: median of per-roast differences a − b (positive => a typically higher).\n"
        "- **Wilcoxon p**: paired signed-rank test for any non-zero median difference. Low p means the models differ in their per-roast scores.\n"
        "- **TOST p**: equivalence test for |mean difference| < ε. Low p means the models are statistically equivalent within ±ε.\n"
        "- **corr**: Pearson correlation of per-roast R² vectors. High => same roasts hard for both; low => different failure modes."
    )
    summary_md.append("")
    summary_md.append("| a | b | n | median Δ R² | Δ CI95 | mean Δ | Wilcoxon p | TOST p | corr |")
    summary_md.append("|---|---|---:|---:|---|---:|---:|---:|---:|")
    for r in pair_records:
        ci = r["median_delta_ci95"]
        summary_md.append(
            f"| {r['model_a']} | {r['model_b']} | {r['n_roasts']} | "
            f"{r['median_delta_r2']:+.4f} | [{ci[0]:+.4f}, {ci[1]:+.4f}] | "
            f"{r['mean_delta_r2']:+.4f} | {r['wilcoxon_p_value']:.2e} | "
            f"{r['tost_p_value_max']:.2e} | {r['per_roast_corr']:+.3f} |"
        )

    out_md = args.outdir / "spectrum_stats.md"
    out_json = args.outdir / "spectrum_stats.json"
    out_md.write_text("\n".join(summary_md) + "\n", encoding="utf-8")
    out_json.write_text(
        json.dumps({"summary": rows, "pairwise": pair_records, "epsilon": args.epsilon}, indent=2),
        encoding="utf-8",
    )
    print(f"\nWritten to {out_md}")
    # Echo the top of the markdown so the user can see quickly.
    for line in summary_md[:30]:
        print(line)


if __name__ == "__main__":
    main()
