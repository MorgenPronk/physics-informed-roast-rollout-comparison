#!/usr/bin/env python
"""Build the per-roast failure-mode correlation heatmap (Figure 4).

For each of the six model classes at seed 11, compute the per-roast rollout
R^2 vector on the 36 held-out test roasts. Then plot the 6x6 matrix of
Pearson correlations between those per-roast vectors as an annotated heatmap.

Anchors the claim that PIML repairs share their hard roasts with one another
(high pairwise r) while the neural baseline finds different roasts hard
(near-zero correlation with every PI variant).

Inputs read from reports/manuscript_hpo/*.json. Writes
manuscript/scientific_reports/submission_latex/figures/per_roast_correlation_heatmap.{pdf,png}.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[1]
HPO_DIR = ROOT / "reports" / "manuscript_hpo"
FIG_DIR = ROOT / "manuscript" / "scientific_reports" / "submission_latex" / "figures"

# Same display order as Figure 3 / Table 1.
SPEC = [
    {"label": "1. Mechanistic", "short": "mech",  "source": "final",
     "key": "whitebox_constant_he_fullstate"},
    {"label": "2. Single-closure PI", "short": "PI single", "source": "final",
     "key": "greybox_learned_he_fullstate"},
    {"label": "3. Multi-closure PI", "short": "PI multi", "source": "multi_closure_final.json"},
    {"label": "4. Bounded FF residual", "short": "Res bounded", "source": "residual_ff_final.json"},
    {"label": "5. Unbounded FF residual", "short": "Res unbounded", "source": "residual_ff_unbounded_final.json"},
    {"label": "6. Neural baseline", "short": "NN", "source": "final",
     "key": "blackbox_core"},
]


def _r2(actual: list[float], pred: list[float]) -> float:
    a = np.asarray(actual, dtype=float)
    p = np.asarray(pred, dtype=float)
    denom = np.sum((a - a.mean()) ** 2)
    if denom < 1e-12:
        return float("nan")
    return float(1.0 - np.sum((a - p) ** 2) / denom)


def collect_per_roast_r2() -> tuple[list[str], dict[str, np.ndarray]]:
    final = json.loads((HPO_DIR / "final_test_metrics.json").read_text(encoding="utf-8"))
    multi = json.loads((HPO_DIR / "multi_closure_final.json").read_text(encoding="utf-8"))
    res_b = json.loads((HPO_DIR / "residual_ff_final.json").read_text(encoding="utf-8"))
    res_u = json.loads((HPO_DIR / "residual_ff_unbounded_final.json").read_text(encoding="utf-8"))
    source_map = {
        "multi_closure_final.json": multi,
        "residual_ff_final.json": res_b,
        "residual_ff_unbounded_final.json": res_u,
    }

    # Anchor the roast order on the mechanistic per-roast dict
    anchor_roasts = list(final["per_seed"]["11"]["whitebox_constant_he_fullstate"]["per_roast"].keys())

    out: dict[str, np.ndarray] = {}
    for entry in SPEC:
        short = entry["short"]
        if entry["source"] == "final":
            key = entry["key"]
            per = final["per_seed"]["11"][key]["per_roast"]
            vals = [_r2(per[r]["actual"], per[r]["rollout_pred"]) for r in anchor_roasts]
        else:
            payload = source_map[entry["source"]]
            per = payload["per_seed"]["11"]["per_roast_r2"]
            vals = [float(per[r]) for r in anchor_roasts]
        out[short] = np.asarray(vals, dtype=float)
    return [e["short"] for e in SPEC], out


def build_figure(order: list[str], per_roast: dict[str, np.ndarray],
                 outpath_pdf: Path, outpath_png: Path) -> None:
    n = len(order)
    corr = np.full((n, n), np.nan)
    for i, a in enumerate(order):
        for j, b in enumerate(order):
            va, vb = per_roast[a], per_roast[b]
            mask = np.isfinite(va) & np.isfinite(vb)
            if mask.sum() >= 3:
                corr[i, j] = float(np.corrcoef(va[mask], vb[mask])[0, 1])

    # Diverging colormap centered on 0
    cmap = LinearSegmentedColormap.from_list(
        "rb", ["#2166ac", "#f7f7f7", "#b2182b"],
    )

    fig, ax = plt.subplots(figsize=(7.6, 6.6))
    im = ax.imshow(corr, cmap=cmap, vmin=-1.0, vmax=1.0, aspect="equal")

    # Tick labels with two-line wrapping for readability
    long_labels = [e["label"] for e in SPEC]
    ax.set_xticks(range(n))
    ax.set_xticklabels(long_labels, rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(n))
    ax.set_yticklabels(long_labels, fontsize=9)

    # Annotate every cell with the correlation value
    for i in range(n):
        for j in range(n):
            val = corr[i, j]
            if not np.isfinite(val):
                continue
            # Black text on light cells, white on saturated cells
            text_color = "white" if abs(val) > 0.6 else "#222222"
            ax.text(j, i, f"{val:+.2f}",
                    ha="center", va="center",
                    fontsize=9.5, color=text_color, fontweight="bold")

    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Pearson $r$ between per-roast $R^2$ vectors", fontsize=10)
    cb.ax.tick_params(labelsize=9)

    ax.set_title(
        "Per-roast failure-mode correlation across model classes\n"
        "(seed 11, 36 held-out test roasts)",
        fontsize=11, pad=12,
    )
    ax.tick_params(top=False, bottom=True, labeltop=False, labelbottom=True)

    fig.tight_layout()
    outpath_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath_pdf, bbox_inches="tight")
    fig.savefig(outpath_png, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"Wrote {outpath_pdf}")
    print(f"Wrote {outpath_png}")

    # Echo PI-vs-NN block for sanity
    print("\nPI x NN baseline column (should be near zero):")
    nn_idx = order.index("NN")
    for i, name in enumerate(order):
        if i == nn_idx:
            continue
        print(f"  {name:>15s} vs NN: r = {corr[i, nn_idx]:+.3f}")


def main() -> None:
    order, per_roast = collect_per_roast_r2()
    print(f"Collected per-roast R^2 for {len(order)} models, "
          f"n_roasts={len(next(iter(per_roast.values())))}")
    build_figure(
        order, per_roast,
        FIG_DIR / "per_roast_correlation_heatmap.pdf",
        FIG_DIR / "per_roast_correlation_heatmap.png",
    )


if __name__ == "__main__":
    main()
