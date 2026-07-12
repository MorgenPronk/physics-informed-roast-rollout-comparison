#!/usr/bin/env python
"""Build the hyperparameter-sensitivity figure for the round-1 response letter (R2.3).

One panel per manuscript model class, in placement-spectrum order. Each panel
scatters all 40 sweep trials as validation rollout R^2 (the sweep's selection
metric, at the sweep-time 100-epoch cap) against learning rate, colored by
batch size, with the validation-selected winner starred. This is the
"error landscape over hyperparameters" view requested by Reviewer 2.

Inputs read from reports/manuscript_hpo/<class>/all_trials.jsonl. Writes
manuscript/scientific_reports/response_figures/response_hyperparameter_sensitivity.{pdf,png}.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
HPO_DIR = ROOT / "reports" / "manuscript_hpo"
OUT_DIR = ROOT / "manuscript" / "scientific_reports" / "response_figures"

# Display order matches Table 1 in the manuscript.
SPEC = [
    {"dir": "whitebox", "label": "1. Mechanistic baseline", "color": "#7f7f7f"},
    {"dir": "greybox", "label": "2. Single-closure PI", "color": "#1f77b4"},
    {"dir": "multi_closure", "label": "3. Multi-closure PI", "color": "#1f77b4"},
    {"dir": "residual_ff", "label": "4. Bounded FF residual", "color": "#ff7f0e"},
    {"dir": "residual_ff_unbounded", "label": "5. Unbounded FF residual", "color": "#ff7f0e"},
    {"dir": "blackbox", "label": "6. Neural baseline", "color": "#2ca02c"},
]

BS_COLORS = {16: "#4C72B0", 32: "#DD8452", 64: "#55A868", "full": "#8172B3"}


def load_trials(class_dir: str) -> list[dict]:
    path = HPO_DIR / class_dir / "all_trials.jsonl"
    trials = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if "error" in rec:
                continue
            if not math.isfinite(float(rec["mean_val_rollout_r2"])):
                continue
            trials.append(rec)
    return trials


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(12.0, 6.8))
    axes = axes.flatten()

    for ax, spec in zip(axes, SPEC):
        trials = load_trials(spec["dir"])
        best = max(trials, key=lambda t: float(t["mean_val_rollout_r2"]))
        for t in trials:
            bs = t["config"].get("batch_size")
            key = bs if bs in BS_COLORS else "full"
            marker_kwargs = dict(
                s=34, color=BS_COLORS[key], alpha=0.8,
                edgecolor="black", linewidth=0.4, zorder=2,
            )
            ax.scatter(float(t["config"]["lr"]), float(t["mean_val_rollout_r2"]), **marker_kwargs)
        ax.scatter(
            float(best["config"]["lr"]), float(best["mean_val_rollout_r2"]),
            s=190, marker="*", color="gold", edgecolor="black", linewidth=0.7,
            zorder=3,
        )
        ax.set_xscale("log")
        ax.margins(y=0.14)
        ax.set_title(spec["label"], fontsize=10.5)
        ax.grid(alpha=0.25, which="both")
        ax.tick_params(labelsize=8.5)

    for idx, ax in enumerate(axes):
        if idx >= 3:
            ax.set_xlabel("Learning rate (log scale)", fontsize=9.5)
        if idx % 3 == 0:
            ax.set_ylabel(r"Validation rollout $R^2$", fontsize=9.5)

    handles = [
        Line2D([], [], linestyle="none", marker="o", markersize=6.5,
               markerfacecolor=BS_COLORS[bs], markeredgecolor="black",
               markeredgewidth=0.4, label=f"batch size {bs}")
        for bs in (16, 32, 64, "full")
    ]
    handles.append(
        Line2D([], [], linestyle="none", marker="*", markersize=13,
               markerfacecolor="gold", markeredgecolor="black",
               markeredgewidth=0.7, label="validation-selected winner")
    )
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, -0.015))
    fig.suptitle(
        "Hyperparameter error landscape: validation rollout $R^2$ across all 40 sweep trials per model class",
        fontsize=12, y=0.995,
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.97))

    for suffix in (".pdf", ".png"):
        fig.savefig(
            (OUT_DIR / "response_hyperparameter_sensitivity").with_suffix(suffix),
            dpi=300, bbox_inches="tight",
        )
    plt.close(fig)
    print(f"Wrote {OUT_DIR / 'response_hyperparameter_sensitivity'}.(pdf|png)")


if __name__ == "__main__":
    main()
