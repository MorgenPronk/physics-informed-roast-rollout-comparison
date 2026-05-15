#!/usr/bin/env python
"""Diagnostic plots from the manuscript HPO sweep.

Reads the per-trial JSONL logs written by ``scripts/tune_manuscript_models.py``
and produces a directory of figures + a markdown index, supporting post-hoc
interpretation of:

    * Parameter efficiency (Pareto: val R^2 vs neural-parameter count)
    * Hyperparameter sensitivity (val R^2 vs lr / batch_size / hidden /
      weight_decay / dropout / residual_weight / max_delta / hidden_size)
    * Convergence behavior (val loss vs epoch for top-K trials per class)
    * Time-to-convergence (best_epoch and wall time vs architecture)
    * Compute-vs-quality (wall time vs val R^2)
    * Per-roast hardness profile (boxplot of per-roast val R^2 by class)
    * Cross-class comparison on the same axes (PI vs neural baseline)

The script is idempotent and safe to run while the sweep is still in progress;
it simply uses whatever rows are present in each ``all_trials.jsonl`` at read
time. Run as:

    .venv/Scripts/python scripts/plot_hpo_diagnostics.py
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HPO_DIR = ROOT / "reports" / "manuscript_hpo"
DEFAULT_OUTDIR = DEFAULT_HPO_DIR / "diagnostics"

CLASS_LABELS = {
    "blackbox": "Neural Net Baseline",
    "greybox": "Physics Informed Model",
    "residual": "Residual LSTM",
    "whitebox": "Mechanistic",
}
CLASS_COLORS = {
    "blackbox": "#c95d2e",
    "greybox": "#2e7d8a",
    "residual": "#4a8b3c",
    "whitebox": "#a4753c",
}
CLASS_ORDER = ("whitebox", "greybox", "residual", "blackbox")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_trials(jsonl_path: Path) -> list[dict[str, Any]]:
    if not jsonl_path.exists():
        return []
    trials: list[dict[str, Any]] = []
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "error" in rec:
                # Skip failure rows; they don't have metrics.
                continue
            trials.append(rec)
    return trials


def _arch_label(widths: Sequence[int]) -> str:
    return "→".join(str(w) for w in widths)


def _bs_label(value: Any) -> str:
    if value == "full":
        return "full"
    return str(int(value))


# ---------------------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------------------


def _save_figure(fig: plt.Figure, base_path: Path) -> None:
    base_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base_path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(base_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _per_class_xy(
    trials: list[dict[str, Any]],
    x_getter,
    y_key: str = "mean_val_rollout_r2",
) -> tuple[list[float], list[float]]:
    xs: list[float] = []
    ys: list[float] = []
    for t in trials:
        try:
            x = x_getter(t)
            y = float(t[y_key])
        except (KeyError, TypeError, ValueError):
            continue
        if x is None:
            continue
        if not (math.isfinite(float(x)) if isinstance(x, (int, float)) else True):
            continue
        if not math.isfinite(y):
            continue
        xs.append(float(x) if isinstance(x, (int, float)) else x)
        ys.append(y)
    return xs, ys


# ---------------------------------------------------------------------------
# Per-class figures
# ---------------------------------------------------------------------------


def figure_pareto(class_name: str, trials: list[dict[str, Any]], outdir: Path) -> None:
    if not trials:
        return
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    xs = [int(t["param_count"]) for t in trials]
    ys = [float(t["mean_val_rollout_r2"]) for t in trials]
    archs = [tuple(t["config"].get("hidden_widths", [t["config"].get("hidden_size", "?")])) for t in trials]
    unique_archs = sorted(set(archs), key=lambda a: sum(a) if all(isinstance(x, int) for x in a) else 0)
    cmap = plt.colormaps["tab10"]
    for idx, arch in enumerate(unique_archs):
        mask = [a == arch for a in archs]
        if not any(mask):
            continue
        ax.scatter(
            [xs[i] for i, m in enumerate(mask) if m],
            [ys[i] for i, m in enumerate(mask) if m],
            color=cmap(idx % 10),
            s=42,
            edgecolor="black",
            linewidth=0.4,
            label=_arch_label(arch) if isinstance(arch[0], int) else str(arch[0]),
        )
    # Pareto frontier (max R^2 for ≤param budget).
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    frontier_x, frontier_y = [], []
    best_so_far = -math.inf
    for i in order:
        if ys[i] > best_so_far:
            best_so_far = ys[i]
            frontier_x.append(xs[i])
            frontier_y.append(ys[i])
    ax.plot(frontier_x, frontier_y, color="black", linewidth=1.2, linestyle="--", alpha=0.6, label="Pareto front")
    ax.set_xscale("log")
    ax.set_xlabel("Neural parameter count (log)")
    ax.set_ylabel(r"Mean val rollout $R^2$")
    ax.set_title(f"{CLASS_LABELS[class_name]} — parameter efficiency Pareto")
    ax.grid(alpha=0.25, which="both")
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="lower right")
    fig.tight_layout()
    _save_figure(fig, outdir / f"{class_name}_pareto_r2_vs_params")


def figure_hyperparameter_scatter(
    class_name: str,
    trials: list[dict[str, Any]],
    outdir: Path,
) -> None:
    if not trials:
        return
    # Define per-class set of hyperparameters to scan.
    hp_definitions: list[dict[str, Any]] = []
    if class_name in ("blackbox", "greybox"):
        hp_definitions += [
            {"key": "lr", "label": "Learning rate", "scale": "log"},
            {"key": "weight_decay", "label": "Weight decay", "scale": "symlog"},
            {"key": "batch_size", "label": "Batch size", "scale": "categorical"},
            {"key": "hidden_widths", "label": "Hidden widths", "scale": "categorical_tuple"},
        ]
    if class_name == "blackbox":
        hp_definitions.append({"key": "dropout", "label": "Dropout", "scale": "linear"})
    if class_name == "whitebox":
        # White-box has no closure MLP — only training schedule varies.
        hp_definitions += [
            {"key": "lr", "label": "Learning rate", "scale": "log"},
            {"key": "weight_decay", "label": "Weight decay", "scale": "symlog"},
            {"key": "batch_size", "label": "Batch size", "scale": "categorical"},
        ]
    if class_name == "residual":
        hp_definitions += [
            {"key": "lr", "label": "Learning rate", "scale": "log"},
            {"key": "hidden_size", "label": "LSTM hidden size", "scale": "categorical"},
            {"key": "max_delta", "label": "Max delta (K)", "scale": "categorical"},
            {"key": "residual_weight", "label": "Residual weight", "scale": "log"},
        ]

    n_hp = len(hp_definitions)
    if n_hp == 0:
        return
    cols = 2
    rows = (n_hp + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(7.0 * cols, 4.0 * rows))
    axes = np.atleast_1d(axes).flatten()

    color = CLASS_COLORS[class_name]
    for ax, hp in zip(axes, hp_definitions):
        key = hp["key"]
        scale = hp["scale"]
        if scale == "categorical_tuple":
            x_raw = [tuple(t["config"].get(key, [])) for t in trials]
        else:
            x_raw = [t["config"].get(key) for t in trials]
        y_vals = [float(t["mean_val_rollout_r2"]) for t in trials]
        finite = [
            (xv, yv) for xv, yv in zip(x_raw, y_vals)
            if xv is not None and math.isfinite(yv)
        ]
        if not finite:
            ax.set_title(f"{hp['label']} (no data)")
            continue
        if scale in ("log", "linear", "symlog"):
            xs = [float(p[0]) for p in finite]
            ys = [p[1] for p in finite]
            ax.scatter(xs, ys, s=36, color=color, alpha=0.75, edgecolor="black", linewidth=0.4)
            if scale == "log":
                ax.set_xscale("log")
            elif scale == "symlog":
                ax.set_xscale("symlog", linthresh=1e-7)
        else:
            categories = sorted({p[0] for p in finite}, key=lambda v: (str(v),))
            cat_x = {cat: idx for idx, cat in enumerate(categories)}
            xs = [cat_x[p[0]] for p in finite]
            ys = [p[1] for p in finite]
            rng = np.random.default_rng(0)
            jitter = (rng.random(len(xs)) - 0.5) * 0.25
            ax.scatter(np.asarray(xs, dtype=float) + jitter, ys, s=36, color=color, alpha=0.75, edgecolor="black", linewidth=0.4)
            ax.set_xticks(range(len(categories)))
            if scale == "categorical_tuple":
                ax.set_xticklabels([_arch_label(c) if isinstance(c, tuple) else str(c) for c in categories], rotation=20, ha="right")
            else:
                ax.set_xticklabels([_bs_label(c) for c in categories])
        ax.set_xlabel(hp["label"])
        ax.set_ylabel(r"Mean val rollout $R^2$")
        ax.grid(alpha=0.25)
        ax.set_title(hp["label"])

    # Hide any unused subplots.
    for ax in axes[n_hp:]:
        ax.axis("off")
    fig.suptitle(f"{CLASS_LABELS[class_name]} — hyperparameter sensitivity", y=1.02, fontsize=12)
    fig.tight_layout()
    _save_figure(fig, outdir / f"{class_name}_hyperparameter_sensitivity")


def figure_convergence_curves(
    class_name: str,
    trials: list[dict[str, Any]],
    outdir: Path,
    top_k: int = 5,
) -> None:
    if not trials:
        return
    finite = [t for t in trials if math.isfinite(float(t["mean_val_rollout_r2"]))]
    if not finite:
        return
    top = sorted(finite, key=lambda t: t["mean_val_rollout_r2"], reverse=True)[:top_k]
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.6))
    cmap = plt.colormaps["viridis"]
    for idx, t in enumerate(top):
        seed_key = next(iter(t.get("per_seed_history", {})), None)
        if seed_key is None:
            continue
        hist = t["per_seed_history"][seed_key]
        if not hist:
            continue
        epochs = [float(h["epoch"]) for h in hist]
        train_loss = [float(h["train_loss"]) for h in hist]
        val_loss = [float(h["val_loss"]) for h in hist]
        color = cmap(idx / max(top_k - 1, 1))
        label = f"#{t['trial_idx']} R²={t['mean_val_rollout_r2']:.3f}"
        axes[0].plot(epochs, train_loss, color=color, linewidth=1.6, label=label)
        axes[1].plot(epochs, val_loss, color=color, linewidth=1.6, label=label)
        # Mark best_epoch.
        best_ep = t.get("per_seed_best_epoch", {}).get(seed_key)
        if best_ep:
            axes[1].axvline(best_ep, color=color, linestyle=":", linewidth=0.8, alpha=0.5)
    for ax, title in zip(axes, ["Train loss", "Val loss"]):
        ax.set_xlabel("Epoch")
        ax.set_ylabel(title)
        ax.set_yscale("log")
        ax.grid(alpha=0.25, which="both")
        ax.set_title(title)
        ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.suptitle(f"{CLASS_LABELS[class_name]} — top-{top_k} trial convergence", y=1.02, fontsize=12)
    fig.tight_layout()
    _save_figure(fig, outdir / f"{class_name}_convergence_top{top_k}")


def figure_best_epoch_distribution(
    class_name: str,
    trials: list[dict[str, Any]],
    outdir: Path,
) -> None:
    if not trials:
        return
    best_eps: list[int] = []
    r2s: list[float] = []
    for t in trials:
        seed_key = next(iter(t.get("per_seed_best_epoch", {})), None)
        if seed_key is None:
            continue
        be = t["per_seed_best_epoch"].get(seed_key)
        r2 = float(t["mean_val_rollout_r2"])
        if be is None or not math.isfinite(r2):
            continue
        best_eps.append(int(be))
        r2s.append(r2)
    if not best_eps:
        return
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.scatter(best_eps, r2s, color=CLASS_COLORS[class_name], s=42, alpha=0.8, edgecolor="black", linewidth=0.4)
    ax.set_xlabel("Best-val epoch")
    ax.set_ylabel(r"Mean val rollout $R^2$")
    ax.set_title(f"{CLASS_LABELS[class_name]} — convergence speed vs final quality")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    _save_figure(fig, outdir / f"{class_name}_best_epoch_vs_r2")


def figure_compute_vs_quality(
    class_name: str,
    trials: list[dict[str, Any]],
    outdir: Path,
) -> None:
    if not trials:
        return
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    xs = [float(t["wall_time_sec"]) for t in trials if math.isfinite(float(t["mean_val_rollout_r2"]))]
    ys = [float(t["mean_val_rollout_r2"]) for t in trials if math.isfinite(float(t["mean_val_rollout_r2"]))]
    sizes = [max(8, math.log10(max(int(t["param_count"]), 1) + 1) * 20) for t in trials if math.isfinite(float(t["mean_val_rollout_r2"]))]
    ax.scatter(xs, ys, s=sizes, color=CLASS_COLORS[class_name], alpha=0.7, edgecolor="black", linewidth=0.4)
    ax.set_xscale("log")
    ax.set_xlabel("Trial wall time [s] (log)")
    ax.set_ylabel(r"Mean val rollout $R^2$")
    ax.set_title(f"{CLASS_LABELS[class_name]} — compute vs quality (marker size ∝ log params)")
    ax.grid(alpha=0.25, which="both")
    fig.tight_layout()
    _save_figure(fig, outdir / f"{class_name}_compute_vs_quality")


def figure_per_roast_hardness(
    class_name: str,
    trials: list[dict[str, Any]],
    outdir: Path,
    top_k: int = 5,
) -> None:
    if not trials:
        return
    finite = [t for t in trials if math.isfinite(float(t["mean_val_rollout_r2"]))]
    if not finite:
        return
    top = sorted(finite, key=lambda t: t["mean_val_rollout_r2"], reverse=True)[:top_k]
    # Take the top-1 trial and show its per-roast R^2 distribution.
    seed_key = next(iter(top[0].get("per_seed_per_roast_val_r2", {})), None)
    if seed_key is None:
        return
    per_roast = top[0]["per_seed_per_roast_val_r2"][seed_key]
    if not per_roast:
        return
    roast_ids = sorted(per_roast.keys(), key=lambda r: per_roast[r])
    r2_values = [per_roast[r] for r in roast_ids]
    fig, ax = plt.subplots(figsize=(max(8.0, len(roast_ids) * 0.18), 5.0))
    ax.bar(range(len(roast_ids)), r2_values, color=CLASS_COLORS[class_name], edgecolor="black", linewidth=0.4)
    ax.axhline(0, color="black", linewidth=0.6, alpha=0.6)
    ax.set_xticks(range(len(roast_ids)))
    ax.set_xticklabels(roast_ids, rotation=80, fontsize=6)
    ax.set_ylabel(r"Per-roast val rollout $R^2$")
    ax.set_title(
        f"{CLASS_LABELS[class_name]} — per-val-roast hardness (top trial #{top[0]['trial_idx']})"
    )
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    _save_figure(fig, outdir / f"{class_name}_per_roast_top1")


# ---------------------------------------------------------------------------
# Cross-class figures
# ---------------------------------------------------------------------------


def figure_cross_class_pareto(
    trials_by_class: dict[str, list[dict[str, Any]]],
    outdir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    for class_name, trials in trials_by_class.items():
        if not trials:
            continue
        xs = [int(t["param_count"]) for t in trials if math.isfinite(float(t["mean_val_rollout_r2"]))]
        ys = [float(t["mean_val_rollout_r2"]) for t in trials if math.isfinite(float(t["mean_val_rollout_r2"]))]
        ax.scatter(xs, ys, s=40, color=CLASS_COLORS[class_name], alpha=0.6,
                   edgecolor="black", linewidth=0.4, label=CLASS_LABELS[class_name])
    ax.set_xscale("log")
    ax.set_xlabel("Neural parameter count (log)")
    ax.set_ylabel(r"Mean val rollout $R^2$")
    ax.set_title("Cross-class parameter efficiency")
    ax.grid(alpha=0.25, which="both")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    _save_figure(fig, outdir / "cross_class_pareto_r2_vs_params")


def figure_cross_class_compute(
    trials_by_class: dict[str, list[dict[str, Any]]],
    outdir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    for class_name, trials in trials_by_class.items():
        if not trials:
            continue
        xs = [float(t["wall_time_sec"]) for t in trials if math.isfinite(float(t["mean_val_rollout_r2"]))]
        ys = [float(t["mean_val_rollout_r2"]) for t in trials if math.isfinite(float(t["mean_val_rollout_r2"]))]
        ax.scatter(xs, ys, s=40, color=CLASS_COLORS[class_name], alpha=0.6,
                   edgecolor="black", linewidth=0.4, label=CLASS_LABELS[class_name])
    ax.set_xscale("log")
    ax.set_xlabel("Trial wall time [s] (log)")
    ax.set_ylabel(r"Mean val rollout $R^2$")
    ax.set_title("Cross-class compute vs quality")
    ax.grid(alpha=0.25, which="both")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    _save_figure(fig, outdir / "cross_class_compute_vs_quality")


def figure_cross_class_per_roast(
    trials_by_class: dict[str, list[dict[str, Any]]],
    outdir: Path,
) -> None:
    """Boxplot of per-roast val R^2 distributions across the top trial of each class."""
    data = []
    labels = []
    colors = []
    for class_name, trials in trials_by_class.items():
        finite = [t for t in trials if math.isfinite(float(t["mean_val_rollout_r2"]))]
        if not finite:
            continue
        top = max(finite, key=lambda t: t["mean_val_rollout_r2"])
        seed_key = next(iter(top.get("per_seed_per_roast_val_r2", {})), None)
        if seed_key is None:
            continue
        per_roast = top["per_seed_per_roast_val_r2"][seed_key]
        if not per_roast:
            continue
        data.append(list(per_roast.values()))
        labels.append(CLASS_LABELS[class_name])
        colors.append(CLASS_COLORS[class_name])
    if not data:
        return
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    bp = ax.boxplot(data, patch_artist=True, tick_labels=labels)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    # Overlay individual points.
    rng = np.random.default_rng(0)
    for idx, (vals, color) in enumerate(zip(data, colors)):
        jitter = (rng.random(len(vals)) - 0.5) * 0.18
        ax.scatter(np.full(len(vals), idx + 1, dtype=float) + jitter, vals, s=14, alpha=0.6, color=color)
    ax.axhline(0, color="black", linewidth=0.6, alpha=0.6)
    ax.set_ylabel(r"Per-val-roast rollout $R^2$")
    ax.set_title("Per-roast val rollout R^2 — top trial of each class")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    _save_figure(fig, outdir / "cross_class_per_roast_top1")


# ---------------------------------------------------------------------------
# Top-trial summary tables
# ---------------------------------------------------------------------------


def write_summary_tables(
    trials_by_class: dict[str, list[dict[str, Any]]],
    outdir: Path,
) -> None:
    lines: list[str] = ["# HPO diagnostics summary", ""]
    for class_name, trials in trials_by_class.items():
        finite = [t for t in trials if math.isfinite(float(t["mean_val_rollout_r2"]))]
        if not finite:
            lines.extend([f"## {CLASS_LABELS[class_name]}", "", "_No trials_", ""])
            continue
        ranked = sorted(finite, key=lambda t: t["mean_val_rollout_r2"], reverse=True)
        lines.append(f"## {CLASS_LABELS[class_name]} ({len(finite)} trials)")
        lines.append("")
        lines.append("| Rank | Trial | Val R² | Params | Best epoch | Wall (s) | Key config |")
        lines.append("|---:|---:|---:|---:|---:|---:|---|")
        for rank, t in enumerate(ranked[:10], start=1):
            seed_key = next(iter(t.get("per_seed_best_epoch", {})), None)
            best_ep = t["per_seed_best_epoch"][seed_key] if seed_key else "?"
            cfg = t["config"]
            arch = cfg.get("hidden_widths") or cfg.get("hidden_size")
            key_cfg = f"hidden={arch} lr={cfg.get('lr', '?'):.2e} bs={cfg.get('batch_size', '?')}"
            lines.append(
                f"| {rank} | {t['trial_idx']} | {t['mean_val_rollout_r2']:.4f} | "
                f"{t['param_count']:,} | {best_ep} | {t['wall_time_sec']:.1f} | {key_cfg} |"
            )
        lines.append("")
    (outdir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_index(
    trials_by_class: dict[str, list[dict[str, Any]]],
    outdir: Path,
) -> None:
    lines = ["# HPO diagnostic figures", "", "_Auto-generated by `scripts/plot_hpo_diagnostics.py`_", ""]
    for class_name in CLASS_ORDER:
        if not trials_by_class.get(class_name):
            continue
        lines.append(f"## {CLASS_LABELS[class_name]}")
        lines.append("")
        for stem, desc in [
            (f"{class_name}_pareto_r2_vs_params", "Parameter efficiency Pareto"),
            (f"{class_name}_hyperparameter_sensitivity", "Hyperparameter sensitivity scatter"),
            (f"{class_name}_convergence_top5", "Top-5 convergence curves"),
            (f"{class_name}_best_epoch_vs_r2", "Convergence speed vs final quality"),
            (f"{class_name}_compute_vs_quality", "Compute vs quality"),
            (f"{class_name}_per_roast_top1", "Per-roast hardness (top trial)"),
        ]:
            png = outdir / f"{stem}.png"
            if png.exists():
                lines.append(f"- {desc}: `{png.relative_to(ROOT).as_posix()}`")
        lines.append("")
    lines.append("## Cross-class")
    lines.append("")
    for stem, desc in [
        ("cross_class_pareto_r2_vs_params", "Cross-class parameter efficiency"),
        ("cross_class_compute_vs_quality", "Cross-class compute vs quality"),
        ("cross_class_per_roast_top1", "Cross-class per-roast hardness (top trial each)"),
    ]:
        png = outdir / f"{stem}.png"
        if png.exists():
            lines.append(f"- {desc}: `{png.relative_to(ROOT).as_posix()}`")
    (outdir / "index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hpo-dir", type=Path, default=DEFAULT_HPO_DIR)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    trials_by_class: dict[str, list[dict[str, Any]]] = {
        class_name: load_trials(args.hpo_dir / class_name / "all_trials.jsonl")
        for class_name in CLASS_ORDER
    }
    for class_name, trials in trials_by_class.items():
        print(f"{class_name}: {len(trials)} trials loaded")

    for class_name, trials in trials_by_class.items():
        figure_pareto(class_name, trials, args.outdir)
        figure_hyperparameter_scatter(class_name, trials, args.outdir)
        figure_convergence_curves(class_name, trials, args.outdir, top_k=args.top_k)
        figure_best_epoch_distribution(class_name, trials, args.outdir)
        figure_compute_vs_quality(class_name, trials, args.outdir)
        figure_per_roast_hardness(class_name, trials, args.outdir, top_k=args.top_k)

    figure_cross_class_pareto(trials_by_class, args.outdir)
    figure_cross_class_compute(trials_by_class, args.outdir)
    figure_cross_class_per_roast(trials_by_class, args.outdir)

    write_summary_tables(trials_by_class, args.outdir)
    write_index(trials_by_class, args.outdir)

    print(f"\nDiagnostics written to {args.outdir}")


if __name__ == "__main__":
    main()
