#!/usr/bin/env python
"""Build the position-spectrum figure (Figure 3).

Three vertically stacked panels sharing the same six-class x-axis:
  A. Held-out rollout R^2 with 95% roast-bootstrap CI (forest plot).
  B. Per-seed R^2 dots (seeds 11/23/37) to make seed stability visible.
  C. Trainable-parameter count on log scale.

Inputs read from reports/manuscript_hpo/*.json. Writes
manuscript/scientific_reports/submission_latex/figures/position_spectrum.{pdf,png}.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HPO_DIR = ROOT / "reports" / "manuscript_hpo"
FIG_DIR = ROOT / "manuscript" / "scientific_reports" / "submission_latex" / "figures"

# Display order matches Table 1 in the manuscript.
SPEC = [
    {
        "label": "1. Mechanistic\nbaseline",
        "short": "mechanistic",
        "source": "final",
        "key": "whitebox_constant_he_fullstate",
        "color": "#7f7f7f",
    },
    {
        "label": "2. Single-closure\nPI",
        "short": "pi_single",
        "source": "final",
        "key": "greybox_learned_he_fullstate",
        "color": "#1f77b4",
    },
    {
        "label": "3. Multi-closure\nPI",
        "short": "multi_closure",
        "source": "multi_closure_final.json",
        "color": "#1f77b4",
    },
    {
        "label": "4. Bounded FF\nresidual",
        "short": "residual_bounded",
        "source": "residual_ff_final.json",
        "color": "#ff7f0e",
    },
    {
        "label": "5. Unbounded FF\nresidual",
        "short": "residual_unbounded",
        "source": "residual_ff_unbounded_final.json",
        "color": "#ff7f0e",
    },
    {
        "label": "6. Neural\nbaseline",
        "short": "nn_baseline",
        "source": "final",
        "key": "blackbox_core",
        "color": "#2ca02c",
    },
]

SEEDS = ("11", "23", "37")


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def collect_metrics() -> list[dict]:
    final = _load_json(HPO_DIR / "final_test_metrics.json")
    multi = _load_json(HPO_DIR / "multi_closure_final.json")
    res_b = _load_json(HPO_DIR / "residual_ff_final.json")
    res_u = _load_json(HPO_DIR / "residual_ff_unbounded_final.json")
    source_map = {
        "multi_closure_final.json": multi,
        "residual_ff_final.json": res_b,
        "residual_ff_unbounded_final.json": res_u,
    }

    rows = []
    for entry in SPEC:
        if entry["source"] == "final":
            key = entry["key"]
            seed11 = final["per_seed"]["11"][key]
            seeds = [
                float(final["per_seed"][s][key]["rollout_metrics"]["r2"])
                for s in SEEDS
            ]
            rows.append({
                "label": entry["label"],
                "short": entry["short"],
                "r2": float(seed11["rollout_metrics"]["r2"]),
                "ci_lo": float(seed11["rollout_r2_ci95"][0]),
                "ci_hi": float(seed11["rollout_r2_ci95"][1]),
                "params": int(seed11["param_count"]),
                "seed_r2": seeds,
                "color": entry["color"],
            })
        else:
            payload = source_map[entry["source"]]
            seed11 = payload["per_seed"]["11"]
            seeds = [
                float(payload["per_seed"][s]["rollout_metrics"]["r2"])
                for s in SEEDS
            ]
            rows.append({
                "label": entry["label"],
                "short": entry["short"],
                "r2": float(seed11["rollout_metrics"]["r2"]),
                "ci_lo": float(seed11["rollout_r2_ci95"][0]),
                "ci_hi": float(seed11["rollout_r2_ci95"][1]),
                "params": int(seed11["param_count"]),
                "seed_r2": seeds,
                "color": entry["color"],
            })
    return rows


def build_figure(rows: list[dict], outpath_pdf: Path, outpath_png: Path) -> None:
    n = len(rows)
    xs = np.arange(n)
    fig, axes = plt.subplots(
        3, 1, figsize=(10.5, 6.0), sharex=True,
        gridspec_kw={"height_ratios": [1.2, 1.2, 1.0], "hspace": 0.15},
    )
    ax_r2, ax_seed, ax_params = axes

    # --- Panel A: R^2 with bootstrap CI ----------------------------------
    for x, row in zip(xs, rows):
        lo = row["r2"] - row["ci_lo"]
        hi = row["ci_hi"] - row["r2"]
        ax_r2.errorbar(
            x, row["r2"],
            yerr=[[lo], [hi]],
            fmt="o", color=row["color"],
            markersize=8, linewidth=1.6, capsize=4,
        )
    ax_r2.axhline(0.0, color="#bbbbbb", linewidth=0.8, linestyle="--", zorder=0)
    ax_r2.set_ylabel("Held-out rollout $R^2$\n(seed 11, 95\\% CI)")
    ax_r2.set_ylim(-0.7, 1.05)
    ax_r2.grid(True, axis="y", alpha=0.3)

    # Annotate seed-11 R^2 above each marker
    for x, row in zip(xs, rows):
        ax_r2.text(
            x, row["ci_hi"] + 0.05,
            f"{row['r2']:.3f}",
            ha="center", va="bottom", fontsize=8.5, color=row["color"],
        )

    # --- Panel B: per-seed R^2 dots (symlog so mechanistic seed 23 is visible) ---
    seed_markers = ["o", "s", "^"]
    seed_labels = [f"seed {s}" for s in SEEDS]
    for x, row in zip(xs, rows):
        # Connect with a thin vertical line to make range visible
        seed_vals = row["seed_r2"]
        ax_seed.plot(
            [x, x], [min(seed_vals), max(seed_vals)],
            color=row["color"], alpha=0.4, linewidth=1.4, zorder=1,
        )
        for s_val, marker in zip(seed_vals, seed_markers):
            ax_seed.plot(
                x, s_val,
                marker=marker, color=row["color"], markersize=7,
                markeredgecolor="white", markeredgewidth=0.6, zorder=2,
            )
        # Annotate range to the right of the marker column
        rng = max(seed_vals) - min(seed_vals)
        if row["short"] == "mechanistic":
            ax_seed.text(
                x + 0.18, max(seed_vals),
                f"$\\Delta$={rng:.2f}",
                ha="left", va="center", fontsize=8, color="#444444",
            )
        else:
            ax_seed.text(
                x + 0.18, np.mean(seed_vals),
                f"$\\Delta$={rng:.3f}",
                ha="left", va="center", fontsize=8, color="#444444",
            )
    ax_seed.axhline(0.0, color="#bbbbbb", linewidth=0.8, linestyle="--", zorder=0)
    ax_seed.set_ylabel("Per-seed rollout $R^2$\n(seeds 11/23/37)")
    ax_seed.set_ylim(-1.85, 1.15)
    ax_seed.grid(True, axis="y", alpha=0.3)

    # Custom legend for seed markers
    legend_handles = [
        plt.Line2D([], [], marker=m, color="#444444", linestyle="",
                   markersize=6, markeredgecolor="white", label=lab)
        for m, lab in zip(seed_markers, seed_labels)
    ]
    ax_seed.legend(handles=legend_handles, loc="lower right", fontsize=8,
                   framealpha=0.9, ncol=3)

    # --- Panel C: parameter count (log scale) ----------------------------
    for x, row in zip(xs, rows):
        ax_params.bar(x, row["params"], color=row["color"], alpha=0.75,
                      edgecolor=row["color"], linewidth=1.0, width=0.7)
        ax_params.text(
            x, row["params"] * 1.18,
            f"{row['params']:,}",
            ha="center", va="bottom", fontsize=8.5, color=row["color"],
        )
    ax_params.set_yscale("log")
    ax_params.set_ylim(50, 5e4)
    ax_params.set_ylabel("Trainable parameters\n(log scale)")
    ax_params.grid(True, axis="y", which="both", alpha=0.3)

    ax_params.set_xticks(xs)
    ax_params.set_xticklabels([r["label"] for r in rows], rotation=0, fontsize=8.5)
    ax_params.set_xlim(-0.6, n - 0.4)

    # Tight layout
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout()
    outpath_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath_pdf, bbox_inches="tight")
    fig.savefig(outpath_png, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"Wrote {outpath_pdf}")
    print(f"Wrote {outpath_png}")


def main() -> None:
    rows = collect_metrics()
    print("Collected rows:")
    for r in rows:
        print(f"  {r['short']:>20s}  R2={r['r2']:.4f}  params={r['params']:>6d}  "
              f"seed_R2={r['seed_r2']}")
    build_figure(
        rows,
        FIG_DIR / "position_spectrum.pdf",
        FIG_DIR / "position_spectrum.png",
    )


if __name__ == "__main__":
    main()
