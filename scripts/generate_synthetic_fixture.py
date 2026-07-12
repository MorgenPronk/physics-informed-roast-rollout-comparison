#!/usr/bin/env python
"""Generate the synthetic benchmark cohort for the public repository.

This dataset is NOT partner telemetry and does NOT reproduce the manuscript's
reported values. It exists so the released code can be exercised end-to-end
(loading, trimming, filtering, splitting, training, rollout) without the
DUA-restricted production cohort, which is available on reasonable request
(see the manuscript's Data Availability section).

Cohort matching: the generator is conditioned ONLY on label-level aggregate
statistics of the production cohort (per-label roast counts, modeled-window
length, charge / turning-point / endpoint probe temperatures, inlet-air
temperature level, air speed, drum speed). These aggregates are embedded as
the STATS constant below; no per-roast sequence, timestamp, or identifier
derives from partner data. Timestamps use a fictitious year-2000 epoch and
roast IDs are SYNTH_*.

Generating physics: the manuscript's mechanistic scaffold (equations (1)-(5))
driven by synthetic operator-style profiles, with deliberate model-form
discrepancies relative to the scaffold the repository fits, so that every
repair position in the placement spectrum has structure to learn:

  * state-dependent heat-transfer coefficient (single-closure target);
  * moisture-loss exponent 1.5 instead of X_b^2, and a label-dependent
    exotherm law with quadratic depletion (multi-closure targets);
  * an unmodeled drum-thermal-reservoir heat path and a flow-dependent
    inlet-temperature sensor mismatch (structured probe-space error for the
    residual and neural-baseline positions);
  * per-roast latent variation in effective mass, charge temperature, and
    initial moisture (the scaffold fits global scalars);
  * first-order probe lag and measurement noise.

The recorded probe signal starts at residual drum temperature while the true
bean charge state stays latent, reproducing the metadata-limited
initialization regime studied in the paper.

Per-label heating scale is auto-calibrated so mean modeled-window lengths
match the production aggregates.

Output: data/synthetic/roast_timeseries_synthetic_p2_only.csv
(the ``_p2_only.csv`` suffix routes it through the same pre-filtered loading
path as the production cohort file; see load_manuscript_cohort).
"""

from __future__ import annotations

import csv
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "synthetic"
OUT_CSV = OUT_DIR / "roast_timeseries_synthetic_p2_only.csv"

SEED = 2000
DT = 5.0  # s, production sample interval

# Label-level aggregate statistics of the production cohort (mean, std).
# These are the ONLY quantities carried over from partner data.
STATS = {
    "P10": dict(n=107, length=(60.5, 1.5), charge=(198.0, 3.3), turn=(99.3, 3.0),
                end=(205.2, 1.4), t2_mean=468.0, vg=(6.66, 0.55), drum=(36.5, 0.05)),
    "P13": dict(n=51, length=(91.7, 2.4), charge=(208.4, 2.8), turn=(88.3, 6.2),
                end=(210.4, 6.0), t2_mean=452.9, vg=(6.78, 0.60), drum=(36.0, 0.23)),
    "P16": dict(n=63, length=(62.9, 0.8), charge=(197.1, 3.8), turn=(104.3, 2.6),
                end=(211.0, 0.9), t2_mean=442.4, vg=(6.92, 0.29), drum=(36.5, 0.03)),
}

# Generating ("true") physics constants; literature-plausible magnitudes for a
# ~60 kg industrial drum. Tuned only to produce roast-like curves.
M_DRY0 = 60.0         # kg mean effective dry bean mass (per-roast latent)
CP_BEAN_DRY = 1700.0  # J/kg/K
CP_WATER = 4180.0
A_BED = 25.0          # m^2 effective contact area
A_AIR = 0.15          # m^2 effective duct area
RHO_G = 0.60          # kg/m^3 hot air
CP_G = 1050.0         # J/kg/K
H0 = 40.0             # W/m^2/K base heat-transfer coefficient
KX = 4.0e2            # 1/s moisture prefactor (generating law, exponent 1.5)
EX = 5500.0           # K
AR = 9.0e9            # J/kg/s exotherm prefactor
ER = 9000.0           # K (generating law scales this per label)
HE_TOT = 8.0e4        # J/kg
HV = 2.3e6            # J/kg latent heat
KP = 0.010            # 1/s probe lag
C_DRUM = 60.0         # W/K unmodeled drum-reservoir coupling
K_DRUM = 0.004        # 1/s drum-reservoir relaxation toward inlet air
T2_START = 222.0      # degC inlet-air level at charge
ER_LABEL_SCALE = {"P10": 1.00, "P13": 0.92, "P16": 1.08}


def he_true(t_gas: float, v_g: float, k_h: float) -> float:
    """State-dependent generating heat-transfer coefficient (the fitted
    scaffold assumes a constant or learned closure)."""
    return k_h * H0 * (1.0 + 0.18 * np.tanh((t_gas - 400.0) / 150.0)) * (v_g / 6.8) ** 0.3


def simulate_roast(rng: np.random.Generator, label: str, k_h: float):
    st = STATS[label]
    charge_tc = rng.normal(*st["charge"]) + 7.0
    endpoint = float(np.clip(rng.normal(*st["end"]), 195.0, 228.0))
    v_air_base = max(4.5, rng.normal(*st["vg"]))
    drum_rpm = rng.normal(*st["drum"])
    # Inlet-air ramp target chosen so the roast-mean T2 approximates the
    # label aggregate under the saturating ramp shape used below.
    t2_end = T2_START + (st["t2_mean"] - T2_START) / 0.595
    horizon = st["length"][0]  # nominal steps for the ramp time constant

    # per-roast latent state the fitted scaffold cannot see
    m_dry = rng.normal(M_DRY0, 5.0)
    t_b = rng.normal(20.0, 3.0)
    x_b = float(np.clip(rng.normal(0.115, 0.012), 0.08, 0.15))
    er_gen = ER * ER_LABEL_SCALE[label]

    n_pre = rng.integers(8, 13)
    tc, t2, vg, tt = [], [], [], []
    for i in range(n_pre):
        tc.append(charge_tc + 2.0 + 0.1 * i + rng.normal(0, 0.25))
        t2.append(T2_START + rng.normal(0, 3.0))
        vg.append(v_air_base + rng.normal(0, 0.05))
        tt.append(drum_rpm + rng.normal(0, 0.05))

    t_rp = charge_tc
    t_drum = charge_tc + 40.0
    h_e_state, ar1 = 0.0, 0.0
    max_steps = 240
    for k in range(max_steps):
        ramp = 1.0 - np.exp(-2.2 * (k + 1) / horizon)
        ar1 = 0.85 * ar1 + rng.normal(0, 2.5)
        t2_rec = T2_START + (t2_end - T2_START) * ramp + ar1          # recorded
        v_g = v_air_base + (0.3 if 25 <= k < 50 else 0.0) + rng.normal(0, 0.04)
        t_gi = 0.97 * t2_rec + 6.0 * (v_g - 6.8)                      # true gas temp

        g_g = A_AIR * RHO_G * v_g
        he = he_true(t_gi, v_g, k_h)
        eps = 1.0 - np.exp(-he * A_BED / (g_g * CP_G))
        t_go = t_gi - (t_gi - t_b) * eps
        phi_gb = g_g * CP_G * (t_gi - t_go)

        t_drum += DT * K_DRUM * (t_gi - t_drum)
        phi_drum = C_DRUM * (t_drum - t_b)                            # unmodeled path

        tb_k = t_b + 273.15
        x_dot = -KX * max(x_b, 0.0) ** 1.5 * np.exp(-EX / tb_k)
        he_dot = AR * np.exp(-er_gen / tb_k) * max(0.0, 1.0 - (h_e_state / HE_TOT) ** 2)
        phi_r = m_dry * he_dot
        phi_ev = HV * (-x_dot) * m_dry

        cp_b = CP_BEAN_DRY + CP_WATER * x_b
        t_b += DT * (phi_gb + phi_drum + phi_r - phi_ev) / (m_dry * (1.0 + x_b) * cp_b)
        x_b = max(0.0, x_b + DT * x_dot)
        h_e_state = min(HE_TOT, h_e_state + DT * he_dot)
        t_rp += DT * KP * (t_b - t_rp)

        tc.append(t_rp + rng.normal(0, 0.35))
        t2.append(t2_rec)
        vg.append(v_g)
        tt.append(drum_rpm + rng.normal(0, 0.05))
        if t_rp >= endpoint and k > 15:
            break

    n_post = rng.integers(7, 11)
    for _ in range(n_post):
        tc.append(tc[-1] - (3.5 + rng.normal(0, 0.4)))
        t2.append(t2[-1] + rng.normal(0, 2.0))
        vg.append(vg[-1] + rng.normal(0, 0.05))
        tt.append(tt[-1] + rng.normal(0, 0.05))
    return np.array(tc), np.array(t2), np.array(vg), np.array(tt)


def modeled_steps(rng: np.random.Generator, label: str, k_h: float) -> float:
    """Charge-to-endpoint step count for calibration pilots."""
    st = STATS[label]
    tc, _, _, _ = simulate_roast(rng, label, k_h)
    # steps between the post-idle peak and the endpoint (dump trim excluded)
    return len(tc) - 18  # approx: idle (~10) + post-dump (~9) removed

def calibrate_heat_scale(label: str) -> float:
    """Binary-search the heating multiplier so mean modeled length matches."""
    target = STATS[label]["length"][0]
    lo, hi = 0.12, 3.0
    for _ in range(9):
        mid = 0.5 * (lo + hi)
        rng = np.random.default_rng(77)
        mean_len = float(np.mean([modeled_steps(rng, label, mid) for _ in range(6)]))
        if mean_len > target:
            lo = mid   # too slow -> too many steps -> increase heat
        else:
            hi = mid
    return 0.5 * (lo + hi)


def main() -> None:
    rng = np.random.default_rng(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    k_h = {label: calibrate_heat_scale(label) for label in STATS}
    print("calibrated heat scales:", {k: round(v, 3) for k, v in k_h.items()})

    header = ["roast_id", "timestamp", "tc", "t1", "t2", "flow_gas", "vac",
              "drum_speed", "vaf_close", "vat_open", "gas_pressure",
              "air_speed", "set_bf", "set_vac", "set_tt", "set_vaf", "set_vat"]
    rows = []
    clock = datetime(2000, 1, 1, 6, 0, 0)  # deliberately fictitious epoch
    idx = 0
    for label, st in STATS.items():
        for _ in range(st["n"]):
            idx += 1
            rid = f"SYNTH_{idx:03d}_{label}"
            tc, t2, vg, tt = simulate_roast(rng, label, k_h[label])
            burner = np.clip(28.0 + 0.10 * (t2 - float(t2.mean())) + rng.normal(0, 2.0, len(t2)), 5, 95)
            for i in range(len(tc)):
                ts = clock + timedelta(seconds=i * DT)
                rows.append([
                    rid, ts.strftime("%Y-%m-%d %H:%M:%S"),
                    round(float(tc[i]), 1),
                    round(float(t2[i] * 1.15 + 60 + rng.normal(0, 4.0)), 1),  # t1
                    round(float(t2[i]), 1),
                    round(float(burner[i]), 1),                               # flow_gas
                    round(float(500 + rng.normal(0, 40.0)), 1),               # vac
                    round(float(tt[i]), 1),
                    round(float(np.clip(25 + rng.normal(0, 8), 0, 100)), 1),  # vaf_close
                    round(float(np.clip(50 + rng.normal(0, 8), 0, 100)), 1),  # vat_open
                    round(float(140 + rng.normal(0, 4.0)), 1),                # gas_pressure
                    round(float(vg[i]), 2),
                    round(float(burner[i]), 1),                               # set_bf
                    50.0, 87.0, 26.0, 56.0,
                ])
            clock += timedelta(minutes=45)

    with OUT_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"wrote {OUT_CSV} ({len(rows)} rows, {idx} roasts)")


if __name__ == "__main__":
    main()
