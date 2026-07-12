#!/usr/bin/env python
"""Build the position-spectrum figure (Figure 3).

Three vertically stacked panels sharing the same six-class x-axis:
  A. Held-out rollout R^2 with 95% roast-bootstrap CI (forest plot).
  B. Per-seed R^2 dots (all retraining seeds: 11/23/37 plus the round-1
     campaign extension in reports/r1_seed_campaign) to make seed stability
     visible. Seed 11 (the headline seed) is drawn filled; additional seeds
     are drawn open, since the extra-seed set differs by class (Methods).
     The panel uses a broken y-axis: an upper segment for the repaired
     classes (where seed spreads of 0.01-0.12 are resolvable) and a lower
     segment for the divergent mechanistic baseline (spread 1.23), which
     would otherwise compress the interesting variation into ~2% of the
     axis.
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

# Round-1 review seed extension (reports/r1_seed_campaign): additional
# retraining seeds per class. Residual classes use seed 59 in place of 53
# because the seed-53 single-closure base's training-roast rollout diverged,
# preventing residual stacking.
CAMPAIGN_DIR_NAME = "r1_seed_campaign"
CAMPAIGN_EXTRA = {
    "mechanistic": [("probe_baseline_seed{s}.json", (41, 53))],
    "pi_single": [("class_greybox_seed{s}.json", (41, 53, 59))],
    "multi_closure": [("class_multi_closure_seed{s}.json", (41, 53, 59))],
    "residual_bounded": [("class_residual_ff_bounded_seed{s}.json", (41, 59))],
    "residual_unbounded": [("class_residual_ff_unbounded_seed{s}.json", (41, 59))],
    "nn_baseline": [("class_blackbox_seed{s}.json", (41, 53, 59))],
}


def _campaign_extra_seeds(short: str) -> list[float]:
    campaign_dir = HPO_DIR.parent / CAMPAIGN_DIR_NAME
    values: list[float] = []
    for pattern, seeds in CAMPAIGN_EXTRA.get(short, []):
        for s in seeds:
            path = campaign_dir / pattern.format(s=s)
            if not path.exists():
                continue
            payload = _load_json(path)
            metrics = payload.get("test_rollout") or payload.get("rollout_metrics") or {}
            r2 = metrics.get("r2")
            if r2 is not None and r2 == r2:  # exclude NaN
                values.append(float(r2))
    return values


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
            seeds.extend(_campaign_extra_seeds(entry["short"]))
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
            seeds.extend(_campaign_extra_seeds(entry["short"]))
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
        4, 1, figsize=(10.5, 6.6), sharex=True,
        gridspec_kw={"height_ratios": [1.2, 1.0, 0.5, 1.0], "hspace": 0.18},
    )
    ax_r2, ax_seed_hi, ax_seed_lo, ax_params = axes

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

    # --- Panel B: per-seed R^2 dots (all retraining seeds), broken axis ----
    # Upper segment: repaired classes; lower segment: mechanistic baseline.
    for ax in (ax_seed_hi, ax_seed_lo):
        for x, row in zip(xs, rows):
            # Connect with a thin vertical line to make range visible
            seed_vals = row["seed_r2"]
            ax.plot(
                [x, x], [min(seed_vals), max(seed_vals)],
                color=row["color"], alpha=0.4, linewidth=1.4, zorder=1,
            )
            # seed_vals[0] is seed 11 (headline, matches Panel A): filled
            # marker. Remaining seeds are drawn open; their identities differ
            # by class (Methods), so markers do not encode seed ids.
            for i, s_val in enumerate(seed_vals):
                if i == 0:
                    ax.plot(
                        x, s_val,
                        marker="o", color=row["color"], markersize=7,
                        markeredgecolor="white", markeredgewidth=0.6,
                        zorder=3,
                    )
                else:
                    ax.plot(
                        x, s_val,
                        marker="o", color=row["color"], markersize=6,
                        markerfacecolor="none", markeredgecolor=row["color"],
                        markeredgewidth=1.2, zorder=2,
                    )

    # Annotate range to the right of the marker column, on the segment
    # where the class's points live.
    for x, row in zip(xs, rows):
        seed_vals = row["seed_r2"]
        rng = max(seed_vals) - min(seed_vals)
        if row["short"] == "mechanistic":
            ax_seed_lo.text(
                x + 0.18, max(seed_vals),
                f"$\\Delta$={rng:.2f}",
                ha="left", va="center", fontsize=8, color="#444444",
            )
        else:
            ax_seed_hi.text(
                x + 0.18, float(np.mean(seed_vals)),
                f"$\\Delta$={rng:.3f}",
                ha="left", va="center", fontsize=8, color="#444444",
            )

    ax_seed_hi.set_ylim(0.62, 1.01)
    ax_seed_lo.set_ylim(-1.85, 0.05)
    ax_seed_lo.axhline(0.0, color="#bbbbbb", linewidth=0.8, linestyle="--",
                       zorder=0)
    ax_seed_hi.set_ylabel("Per-seed rollout $R^2$\n(all retraining seeds)")
    ax_seed_hi.grid(True, axis="y", alpha=0.3)
    ax_seed_lo.grid(True, axis="y", alpha=0.3)

    # Broken-axis styling: hide the adjoining spines and draw break marks.
    ax_seed_hi.spines["bottom"].set_visible(False)
    ax_seed_hi.tick_params(bottom=False)
    d = 0.5  # slope of the break marks
    break_kw = dict(
        marker=[(-1, -d), (1, d)], markersize=8, linestyle="none",
        color="#444444", mec="#444444", mew=1, clip_on=False,
    )
    ax_seed_hi.plot([0, 1], [0, 0], transform=ax_seed_hi.transAxes,
                    **break_kw)
    ax_seed_lo.plot([0, 1], [1, 1], transform=ax_seed_lo.transAxes,
                    **break_kw)

    # Custom legend: headline seed vs additional retraining seeds
    legend_handles = [
        plt.Line2D([], [], marker="o", color="#444444", linestyle="",
                   markersize=6, markeredgecolor="white",
                   label="seed 11 (headline)"),
        plt.Line2D([], [], marker="o", color="#444444", linestyle="",
                   markersize=6, markerfacecolor="none",
                   markeredgecolor="#444444",
                   label="additional seeds"),
    ]
    ax_seed_hi.legend(handles=legend_handles, loc="lower right", fontsize=8,
                      framealpha=0.9, ncol=2)

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
