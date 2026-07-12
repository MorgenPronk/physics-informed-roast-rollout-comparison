#!/usr/bin/env python
"""Build the roast-profile schematic used to explain preprocessing windows."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "manuscript" / "scientific_reports" / "submission_latex" / "figures"
STEM = "roast_profile_preprocessing_schematic"


def smooth_edge(y: np.ndarray, passes: int = 3) -> np.ndarray:
    """Lightly smooth an interpolated schematic without changing endpoints."""
    out = y.copy()
    for _ in range(passes):
        padded = np.pad(out, (2, 2), mode="edge")
        out = (
            padded[:-4]
            + 4 * padded[1:-3]
            + 6 * padded[2:-2]
            + 4 * padded[3:-1]
            + padded[4:]
        ) / 16
    return out


def build_profile() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    time = np.linspace(-120, 820, 941)
    modeled_t = np.array([0, 35, 70, 100, 150, 220, 300, 400, 485, 600, 700])
    measured_y = np.array([165, 120, 100, 95, 105, 135, 160, 183, 202, 222, 240])
    latent_y = np.array([25, 55, 86, 107, 140, 162, 180, 199, 214, 235, 252])

    measured = np.empty_like(time)
    latent = np.full_like(time, np.nan)

    warmup = time < 0
    measured[warmup] = 157 + 18 * (1 - np.exp(-(time[warmup] + 120) / 48))

    modeled = (time >= 0) & (time <= 700)
    measured[modeled] = smooth_edge(np.interp(time[modeled], modeled_t, measured_y))
    latent[modeled] = smooth_edge(np.interp(time[modeled], modeled_t, latent_y))

    cooling = time > 700
    measured[cooling] = 40 + (240 - 40) * np.exp(-(time[cooling] - 700) / 45)
    latent[cooling] = 35 + (252 - 35) * np.exp(-(time[cooling] - 700) / 55)

    return time, measured, latent


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    time, measured, latent = build_profile()

    plt.rcParams.update(
        {
            "font.size": 8.5,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.5,
        }
    )
    fig, ax = plt.subplots(figsize=(7.4, 4.2))

    ax.axvspan(-120, 0, color="#d9d9d9", alpha=0.55, lw=0)
    ax.axvspan(700, 820, color="#d9d9d9", alpha=0.55, lw=0)
    ax.axvspan(0, 700, color="#eaf4f7", alpha=0.32, lw=0)

    measured_color = "#1b6f8a"
    latent_color = "#bf6a2d"
    ax.plot(time, measured, color=measured_color, lw=2.4, label=r"Measured bean-probe temperature, $T_c$")
    ax.plot(time, latent, color=latent_color, lw=2.4, label="Conceptual latent bean temperature")

    event_lines = [
        (0, "Charge\nbeans enter drum", "#333333", "-"),
        (100, "Turning\npoint", measured_color, "--"),
        (485, "First\ncrack", "#c93a36", "--"),
        (675, "Second\ncrack", "#c93a36", "--"),
        (700, "Dump\nto cooling tray", "#333333", "-"),
    ]
    for xpos, label, color, style in event_lines:
        ax.axvline(xpos, color=color, linestyle=style, lw=1.45, alpha=0.95)
        y = 268 if xpos in (0, 100) else (278 if xpos == 485 else (254 if xpos == 675 else 230))
        ha = "right" if xpos == 675 else ("left" if xpos == 700 else "center")
        ax.text(xpos, y, label, color=color, fontsize=8.5, ha=ha, va="top")

    stage_y = 15
    stage_specs = [
        (-60, "Excluded\nwarmup"),
        (50, "Charge-drop\ntransient"),
        (180, "Drying"),
        (350, "Yellowing"),
        (590, "Development"),
        (760, "Excluded\ncooling"),
    ]
    for xpos, label in stage_specs:
        ax.text(xpos, stage_y, label, ha="center", va="bottom", fontsize=8.5, color="#333333")

    ax.annotate(
        "Modeled window retained for training and rollout",
        xy=(350, 296),
        xytext=(350, 296),
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="#333333",
    )
    ax.annotate(
        "",
        xy=(0, 291),
        xytext=(700, 291),
        arrowprops=dict(arrowstyle="<->", color="#333333", lw=1.3),
    )

    ax.set_xlim(-120, 820)
    ax.set_ylim(0, 305)
    ax.set_xlabel("Time relative to bean charge (s)")
    ax.set_ylabel(r"Temperature ($^\circ$C)")
    ax.set_xticks([-100, 0, 100, 300, 500, 700, 800])
    ax.set_yticks(np.arange(0, 301, 50))
    ax.grid(True, color="#d7d7d7", lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2, frameon=False)

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{STEM}.png", dpi=300)
    fig.savefig(FIG_DIR / f"{STEM}.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
