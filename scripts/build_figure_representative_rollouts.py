#!/usr/bin/env python
"""Build the representative-rollouts figure (Figure 5).

Plots the measured bean-probe temperature and the autonomous-rollout
predictions of the four model classes that anchor the paper's argument
for one deterministic-longest held-out test roast:

  - Mechanistic baseline (broken scaffold)
  - Multi-closure PI (strongest in-scaffold repair)
  - Unbounded FF residual (strongest out-of-scaffold repair)
  - Matched-input neural baseline (predictive ceiling)

Reads seed-11 trajectories from
reports/manuscript_hpo/seed11_rollouts.json (produced by
scripts/regenerate_seed11_trajectories.py) and writes
manuscript/scientific_reports/submission_latex/figures/representative_rollouts.{pdf,png}.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HPO_DIR = ROOT / "reports" / "manuscript_hpo"
FIG_DIR = ROOT / "manuscript" / "scientific_reports" / "submission_latex" / "figures"

INPUT_JSON = HPO_DIR / "seed11_rollouts.json"

# Model lines to plot (in legend / draw order). Mirrors the six-row
# position spectrum in Table 1 / Figure 3 so the representative rollout
# shows every model class anchored on the same held-out roast.
LINES = [
    {"key": "mechanistic",         "label": "1. Mechanistic baseline",   "color": "#7f7f7f", "ls": ":",  "lw": 1.6},
    {"key": "pi_single",           "label": "2. Single-closure PI",      "color": "#5599c9", "ls": "--", "lw": 1.4},
    {"key": "multi_closure",       "label": "3. Multi-closure PI",       "color": "#1f77b4", "ls": "-",  "lw": 1.6},
    {"key": "residual_bounded",    "label": "4. Bounded FF residual",    "color": "#ffb466", "ls": "--", "lw": 1.4},
    {"key": "residual_unbounded",  "label": "5. Unbounded FF residual",  "color": "#ff7f0e", "ls": "-",  "lw": 1.6},
    {"key": "nn_baseline",         "label": "6. Neural baseline",        "color": "#2ca02c", "ls": "-",  "lw": 1.6},
]


def _r2(actual: list[float], pred: list[float]) -> float:
    a = np.asarray(actual, dtype=float)
    p = np.asarray(pred, dtype=float)
    denom = np.sum((a - a.mean()) ** 2)
    if denom < 1e-12:
        return float("nan")
    return float(1.0 - np.sum((a - p) ** 2) / denom)


def pick_representative_roast(data: dict) -> str:
    """Deterministic-longest held-out roast, as in Methods."""
    mech = data["models"]["mechanistic"]
    longest = max(mech.items(), key=lambda kv: len(kv[1]["actual"]))
    return longest[0]


def build_figure(data: dict, roast_id: str,
                 outpath_pdf: Path, outpath_png: Path) -> None:
    n = len(data["models"]["mechanistic"][roast_id]["actual"])
    t_idx = np.arange(n)  # sample index along the trimmed charge-to-dump window
    actual = np.asarray(data["models"]["mechanistic"][roast_id]["actual"], dtype=float)

    fig, ax = plt.subplots(figsize=(8.6, 5.0))

    # Plot ground-truth measured trace first so model lines overlay it
    ax.plot(t_idx, actual, color="black", linewidth=2.0, label="Measured $T_c$", zorder=10)

    handles = []
    for spec in LINES:
        model_key = spec["key"]
        if model_key not in data["models"]:
            print(f"  WARNING: missing model {model_key} in trajectories JSON")
            continue
        per = data["models"][model_key].get(roast_id)
        if per is None:
            print(f"  WARNING: roast {roast_id} not in {model_key}")
            continue
        pred = np.asarray(per["rollout_pred"], dtype=float)
        r2 = _r2(per["actual"], per["rollout_pred"])
        line, = ax.plot(
            t_idx, pred,
            color=spec["color"], linestyle=spec["ls"], linewidth=spec["lw"],
            label=f"{spec['label']} ($R^2={r2:+.3f}$)",
            zorder=5,
        )
        handles.append(line)

    ax.set_xlabel("Sample index from charge")
    ax.set_ylabel("Bean-probe temperature $T_c$ [$^\\circ$C]")
    ax.set_title(f"Representative held-out rollout (test roast, seed 11): {roast_id}",
                 fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.95)

    fig.tight_layout()
    outpath_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath_pdf, bbox_inches="tight")
    fig.savefig(outpath_png, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"Wrote {outpath_pdf}")
    print(f"Wrote {outpath_png}")


def main() -> None:
    if not INPUT_JSON.exists():
        raise FileNotFoundError(
            f"Missing {INPUT_JSON}. Run scripts/regenerate_seed11_trajectories.py first."
        )
    data = json.loads(INPUT_JSON.read_text(encoding="utf-8"))
    roast_id = pick_representative_roast(data)
    n_samples = len(data["models"]["mechanistic"][roast_id]["actual"])
    print(f"Representative roast: {roast_id}  ({n_samples} samples)")
    print(f"Models available: {sorted(data['models'].keys())}")
    build_figure(
        data, roast_id,
        FIG_DIR / "representative_rollouts.pdf",
        FIG_DIR / "representative_rollouts.png",
    )


if __name__ == "__main__":
    main()
