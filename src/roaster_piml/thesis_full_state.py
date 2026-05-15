"""Fuller thesis-style grey-box ODE reconstruction with latent physical states."""

from __future__ import annotations

import csv
import datetime as dt
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import torch
from torch import nn

from .io_utils import parse_float, parse_timestamp, stable_hash_bucket, stable_split_key
from .model import Metrics, compute_metrics
from .roast_window import detect_charge_start_index, detect_dump_end_index, detect_modeled_window_bounds


@dataclass
class FullRoastSequence:
    roast_id: str
    timestamps: List[dt.datetime]
    tc: List[float]
    t1: List[float]
    t2: List[float]
    t3: List[float]
    flow_gas: List[float]
    bf_command: List[float]
    air_speed: List[float]
    drum_speed: List[float]
    batch_mass_kg: float | None = None
    recipe_number: int | None = None

    @property
    def length(self) -> int:
        return len(self.tc)


@dataclass
class FullRoastBatch:
    roast_ids: List[str]
    tc: torch.Tensor
    t1: torch.Tensor
    t2: torch.Tensor
    t3: torch.Tensor
    flow_gas: torch.Tensor
    bf_command: torch.Tensor
    air_speed: torch.Tensor
    drum_speed: torch.Tensor
    batch_mass_kg: torch.Tensor
    dt_seconds: torch.Tensor
    mask: torch.Tensor


@dataclass
class FullStateRunResult:
    metrics: Dict[str, Dict[str, Metrics]]
    representative_roast_id: str
    predictions: Dict[str, Dict[str, List[float]]]


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _convert_gas_temp(value: float, unit: str) -> float:
    if unit == "fahrenheit":
        return (value - 32.0) * (5.0 / 9.0)
    return value


def _trim_rows(rows: List[Dict[str, object]], trim_mode: str) -> List[Dict[str, object]]:
    if trim_mode == "none" or not rows:
        return rows
    if trim_mode == "peak_tc":
        peak_index = max(range(len(rows)), key=lambda idx: float(rows[idx]["tc"]))
        return rows[: peak_index + 1]
    if trim_mode == "first_drop_after_peak":
        peak_index = max(range(len(rows)), key=lambda idx: float(rows[idx]["tc"]))
        peak_value = float(rows[peak_index]["tc"])
        drop_threshold = 3.0
        for idx in range(peak_index + 1, len(rows)):
            tc_value = float(rows[idx]["tc"])
            if peak_value - tc_value >= drop_threshold:
                return rows[: idx + 1]
        return rows[: peak_index + 1]
    if trim_mode == "first_major_drop":
        rolling_peak = float(rows[0]["tc"])
        peak_index = 0
        drop_threshold = 6.0
        sustain_threshold = 3.0
        sustain_steps = 3
        for idx in range(1, len(rows)):
            tc_value = float(rows[idx]["tc"])
            if tc_value > rolling_peak:
                rolling_peak = tc_value
                peak_index = idx
                continue
            if idx <= peak_index:
                continue
            if rolling_peak - tc_value < drop_threshold:
                continue
            window = rows[idx : min(len(rows), idx + sustain_steps)]
            if all((rolling_peak - float(item["tc"])) >= sustain_threshold for item in window):
                return rows[: peak_index + 1]
        return rows
    raise ValueError(f"Unsupported trim_mode: {trim_mode}")


def _align_charge_start(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    if len(rows) < 20:
        return rows

    tc_values = [float(item["tc"]) for item in rows]
    charge_start = detect_charge_start_index(tc_values)
    return rows[charge_start:]


def load_full_roast_sequences(
    path: Path,
    min_length: int = 20,
    gas_temp_unit: str = "fahrenheit",
    trim_mode: str = "none",
    align_charge_start: bool = False,
    modeled_window: bool = False,
) -> List[FullRoastSequence]:
    grouped: Dict[str, List[Dict[str, object]]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            ts = parse_timestamp(row.get("timestamp", ""))
            values = {k: parse_float(row.get(k, "")) for k in ["tc", "t1", "t2", "flow_gas", "air_speed", "drum_speed"]}
            t3_value = parse_float(row.get("t3", ""))
            set_bf_value = parse_float(row.get("set_bf", ""))
            batch_mass_kg = parse_float(row.get("batch_mass_kg", ""))
            recipe_number_value = parse_float(row.get("recipe_number", ""))
            if ts is None or any(v is None for v in values.values()):
                continue
            t1_value = _convert_gas_temp(float(values["t1"]), gas_temp_unit)
            t2_value = _convert_gas_temp(float(values["t2"]), gas_temp_unit)
            t3_converted = _convert_gas_temp(float(t3_value), gas_temp_unit) if t3_value is not None else float("nan")
            bf_command = float(set_bf_value) if set_bf_value is not None else float(values["flow_gas"])
            grouped.setdefault(row.get("roast_id", "unknown"), []).append(
                {
                    "timestamp": ts,
                    "tc": float(values["tc"]),
                    "t1": t1_value,
                    "t2": t2_value,
                    "t3": t3_converted,
                    "flow_gas": float(values["flow_gas"]),
                    "bf_command": bf_command,
                    "air_speed": float(values["air_speed"]),
                    "drum_speed": float(values["drum_speed"]),
                    "batch_mass_kg": float(batch_mass_kg) if batch_mass_kg is not None else None,
                    "recipe_number": int(recipe_number_value) if recipe_number_value is not None else None,
                }
            )

    sequences: List[FullRoastSequence] = []
    for roast_id, rows in grouped.items():
        rows.sort(key=lambda item: item["timestamp"])
        if modeled_window:
            start_idx, end_idx = detect_modeled_window_bounds([float(item["tc"]) for item in rows])
            rows = rows[start_idx:end_idx]
        if align_charge_start:
            rows = _align_charge_start(rows)
        rows = _trim_rows(rows, trim_mode)
        if len(rows) < min_length:
            continue
        sequences.append(
            FullRoastSequence(
                roast_id=roast_id,
                timestamps=[item["timestamp"] for item in rows],
                tc=[float(item["tc"]) for item in rows],
                t1=[float(item["t1"]) for item in rows],
                t2=[float(item["t2"]) for item in rows],
                t3=[float(item["t3"]) for item in rows],
                flow_gas=[float(item["flow_gas"]) for item in rows],
                bf_command=[float(item["bf_command"]) for item in rows],
                air_speed=[float(item["air_speed"]) for item in rows],
                drum_speed=[float(item["drum_speed"]) for item in rows],
                batch_mass_kg=rows[0].get("batch_mass_kg"),
                recipe_number=rows[0].get("recipe_number"),
            )
        )
    sequences.sort(key=lambda seq: seq.roast_id)
    return sequences


def split_sequences_train_val_test(
    sequences: Sequence[FullRoastSequence],
    train_cutoff: int = 70,
    val_cutoff: int = 85,
    strategy: str = "legacy",
) -> tuple[List[FullRoastSequence], List[FullRoastSequence], List[FullRoastSequence]]:
    train: List[FullRoastSequence] = []
    val: List[FullRoastSequence] = []
    test: List[FullRoastSequence] = []
    for seq in sequences:
        bucket = stable_split_key(seq.roast_id) if strategy == "legacy" else stable_hash_bucket(seq.roast_id)
        if bucket < train_cutoff:
            train.append(seq)
        elif bucket < val_cutoff:
            val.append(seq)
        else:
            test.append(seq)
    return train, val, test


def _sequence_dt_seconds(seq: FullRoastSequence) -> List[float]:
    values: List[float] = []
    for i in range(len(seq.timestamps) - 1):
        dt_seconds = (seq.timestamps[i + 1] - seq.timestamps[i]).total_seconds()
        values.append(max(1.0, min(float(dt_seconds), 30.0)))
    return values


def _make_batch(sequences: Sequence[FullRoastSequence], device: torch.device) -> FullRoastBatch:
    max_len = max(seq.length for seq in sequences)
    batch_size = len(sequences)

    def zeros() -> torch.Tensor:
        return torch.zeros((batch_size, max_len), dtype=torch.float32, device=device)

    tc = zeros()
    t1 = zeros()
    t2 = zeros()
    t3 = zeros()
    flow_gas = zeros()
    bf_command = zeros()
    air_speed = zeros()
    drum_speed = zeros()
    batch_mass_kg = torch.full((batch_size,), float("nan"), dtype=torch.float32, device=device)
    dt_seconds = torch.zeros((batch_size, max_len - 1), dtype=torch.float32, device=device)
    mask = torch.zeros((batch_size, max_len - 1), dtype=torch.bool, device=device)

    roast_ids: List[str] = []
    for i, seq in enumerate(sequences):
        roast_ids.append(seq.roast_id)
        length = seq.length
        tc[i, :length] = torch.tensor(seq.tc, dtype=torch.float32, device=device)
        t1[i, :length] = torch.tensor(seq.t1, dtype=torch.float32, device=device)
        t2[i, :length] = torch.tensor(seq.t2, dtype=torch.float32, device=device)
        t3[i, :length] = torch.tensor(seq.t3, dtype=torch.float32, device=device)
        flow_gas[i, :length] = torch.tensor(seq.flow_gas, dtype=torch.float32, device=device)
        bf_command[i, :length] = torch.tensor(seq.bf_command, dtype=torch.float32, device=device)
        air_speed[i, :length] = torch.tensor(seq.air_speed, dtype=torch.float32, device=device)
        drum_speed[i, :length] = torch.tensor(seq.drum_speed, dtype=torch.float32, device=device)
        if seq.batch_mass_kg is not None:
            batch_mass_kg[i] = float(seq.batch_mass_kg)
        dts = _sequence_dt_seconds(seq)
        if dts:
            dt_seconds[i, : len(dts)] = torch.tensor(dts, dtype=torch.float32, device=device)
            mask[i, : len(dts)] = True

    return FullRoastBatch(
        roast_ids=roast_ids,
        tc=tc,
        t1=t1,
        t2=t2,
        t3=t3,
        flow_gas=flow_gas,
        bf_command=bf_command,
        air_speed=air_speed,
        drum_speed=drum_speed,
        batch_mass_kg=batch_mass_kg,
        dt_seconds=dt_seconds,
        mask=mask,
    )


def make_batches(
    sequences: Sequence[FullRoastSequence],
    batch_size: int,
    device: torch.device,
    shuffle: bool,
) -> List[FullRoastBatch]:
    ordered = sorted(sequences, key=lambda seq: seq.length, reverse=True)
    batches = [_make_batch(ordered[i : i + batch_size], device) for i in range(0, len(ordered), batch_size)]
    if shuffle:
        random.shuffle(batches)
    return batches


class FullStateRoasterModel(nn.Module):
    model_name = "full_state_base"

    DB_METERS = 6.6e-3
    MB_DRY_KG = 91.8e-3
    AB_M2 = 0.08
    X_COEFF_BASE = 4.32e9 / ((DB_METERS * 1.0e3) ** 2)
    X_ACTIVATION_BASE = 9889.0
    AR_BASE = 1.162e8
    HET_BASE = 232.0e3
    HV_BASE = 2790.0e3
    ER_BASE = 5500.0

    def __init__(
        self,
        *,
        fixed_bean_mass_kg: float | None = None,
        fixed_initial_bean_temp_c: float | None = None,
        fixed_air_area_m2: float | None = None,
        fixed_initial_moisture_ratio: float | None = None,
        batch_mass_basis: str = "dry",
    ) -> None:
        super().__init__()
        self.log_ab_scale = nn.Parameter(torch.tensor(0.0))
        self.log_mb_scale = nn.Parameter(torch.tensor(0.0))
        self.log_probe_k = nn.Parameter(torch.tensor(-5.0))
        self.log_air_area_scale = nn.Parameter(torch.tensor(0.0))
        self.init_tb_offset = nn.Parameter(torch.tensor(8.0))
        self.init_xb_logit = nn.Parameter(torch.tensor(-2.4))

        self.log_x_coeff_scale = nn.Parameter(torch.tensor(0.0))
        self.log_x_activation_scale = nn.Parameter(torch.tensor(0.0))
        self.log_ar_scale = nn.Parameter(torch.tensor(0.0))
        self.log_het_scale = nn.Parameter(torch.tensor(0.0))
        self.log_reaction_activation_scale = nn.Parameter(torch.tensor(0.0))
        self.log_hv_scale = nn.Parameter(torch.tensor(0.0))

        self.init_net = nn.Sequential(
            nn.Linear(6, 32),
            nn.Tanh(),
            nn.Linear(32, 4),
        )
        final_linear = self.init_net[-1]
        if isinstance(final_linear, nn.Linear):
            nn.init.zeros_(final_linear.weight)
            nn.init.zeros_(final_linear.bias)

        self.fixed_bean_mass_kg = float(fixed_bean_mass_kg) if fixed_bean_mass_kg is not None else None
        self.fixed_initial_bean_temp_c = (
            float(fixed_initial_bean_temp_c) if fixed_initial_bean_temp_c is not None else None
        )
        self.fixed_air_area_m2 = float(fixed_air_area_m2) if fixed_air_area_m2 is not None else None
        self.fixed_initial_moisture_ratio = (
            float(fixed_initial_moisture_ratio) if fixed_initial_moisture_ratio is not None else None
        )
        if batch_mass_basis not in {"dry", "wet"}:
            raise ValueError(f"Unsupported batch_mass_basis: {batch_mass_basis}")
        self.batch_mass_basis = batch_mass_basis
        if self.fixed_bean_mass_kg is not None:
            self.log_mb_scale.requires_grad_(False)
        if self.fixed_initial_bean_temp_c is not None:
            self.init_tb_offset.requires_grad_(False)
        if self.fixed_air_area_m2 is not None:
            self.log_air_area_scale.requires_grad_(False)
        if self.fixed_initial_moisture_ratio is not None:
            self.init_xb_logit.requires_grad_(False)

    def he(self, tg_c: torch.Tensor, vg: torch.Tensor, tt: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _air_area(self) -> torch.Tensor:
        if self.fixed_air_area_m2 is not None:
            return torch.tensor(self.fixed_air_area_m2, dtype=torch.float32, device=self.log_air_area_scale.device)
        return 0.01 * torch.exp(self.log_air_area_scale)

    def _ab(self) -> torch.Tensor:
        return self.AB_M2 * torch.exp(self.log_ab_scale)

    def _initial_xb_reference(self) -> torch.Tensor:
        if self.fixed_initial_moisture_ratio is not None:
            return torch.tensor(
                self.fixed_initial_moisture_ratio,
                dtype=torch.float32,
                device=self.log_ab_scale.device,
            )
        return 0.02 + 0.12 * torch.sigmoid(self.init_xb_logit)

    def _mb_dry(self, batch_mass_kg: torch.Tensor | None = None) -> torch.Tensor:
        if batch_mass_kg is not None:
            provided = torch.isfinite(batch_mass_kg) & (batch_mass_kg > 0.0)
            if self.fixed_bean_mass_kg is not None:
                fallback = torch.full_like(batch_mass_kg, self.fixed_bean_mass_kg)
            else:
                fallback = torch.full_like(batch_mass_kg, 1.0) * (self.MB_DRY_KG * torch.exp(self.log_mb_scale))
            if self.batch_mass_basis == "wet":
                xb0 = self._initial_xb_reference().to(batch_mass_kg.device)
                provided_mass = batch_mass_kg / (1.0 + xb0)
            else:
                provided_mass = batch_mass_kg
            return torch.where(provided, provided_mass, fallback)
        if self.fixed_bean_mass_kg is not None:
            return torch.tensor(self.fixed_bean_mass_kg, dtype=torch.float32, device=self.log_ab_scale.device)
        return self.MB_DRY_KG * torch.exp(self.log_mb_scale)

    def _cpg(self, tg_k: torch.Tensor) -> torch.Tensor:
        return (
            5.3091e-17 * tg_k**6
            - 4.1550e-13 * tg_k**5
            + 1.3621e-9 * tg_k**4
            - 2.3267e-6 * tg_k**3
            + 2.1034e-3 * tg_k**2
            - 7.2075e-1 * tg_k
            + 1.0839e3
        )

    def _rho_g(self, tg_k: torch.Tensor) -> torch.Tensor:
        return 353.34 * torch.pow(torch.clamp(tg_k, min=1.0), -1.002)

    def _mu_g(self, tg_k: torch.Tensor) -> torch.Tensor:
        return (
            1.2184e-24 * tg_k**6
            - 8.1123e-21 * tg_k**5
            + 1.6089e-17 * tg_k**4
            + 1.1460e-15 * tg_k**3
            - 3.9733e-11 * tg_k**2
            + 7.1226e-8 * tg_k
            + 4.8855e-7
        )

    def _lambda_g(self, tg_k: torch.Tensor) -> torch.Tensor:
        return (
            1.3819e-20 * tg_k**6
            - 9.1506e-17 * tg_k**5
            + 2.2342e-13 * tg_k**4
            - 2.2872e-10 * tg_k**3
            + 6.8867e-8 * tg_k**2
            + 8.0128e-5 * tg_k
            + 7.6694e-4
        )

    def _cpb(self, xb: torch.Tensor) -> torch.Tensor:
        return 1.0e3 * (1.674 + 2.51 * (xb / (1.0 + xb)))

    def _x_coeff(self) -> torch.Tensor:
        return self.X_COEFF_BASE * torch.exp(self.log_x_coeff_scale)

    def _x_activation(self) -> torch.Tensor:
        return self.X_ACTIVATION_BASE * torch.exp(self.log_x_activation_scale)

    def _ar(self) -> torch.Tensor:
        return self.AR_BASE * torch.exp(self.log_ar_scale)

    def _het_total(self) -> torch.Tensor:
        return self.HET_BASE * torch.exp(self.log_het_scale)

    def _reaction_activation(self) -> torch.Tensor:
        return self.ER_BASE * torch.exp(self.log_reaction_activation_scale)

    def _hv(self) -> torch.Tensor:
        return self.HV_BASE * torch.exp(self.log_hv_scale)

    def _compute_dxb_dt(self, xb: torch.Tensor, tb_k: torch.Tensor, eps: float) -> torch.Tensor:
        """Default moisture-loss rate from Vosloo-style Arrhenius kinetics.

        Subclasses may override to swap in a learned closure for ``dxb/dt``.
        """
        x_coeff = self._x_coeff()
        x_activation = self._x_activation()
        return -x_coeff * (xb ** 2) * torch.exp(-x_activation / (tb_k + eps))

    def _compute_dhe_dt(
        self,
        he_state: torch.Tensor,
        tb_k: torch.Tensor,
        het_total: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        """Default exothermic reaction rate from Vosloo-style Arrhenius kinetics.

        Subclasses may override to swap in a learned closure for ``dH_e/dt``.
        """
        ar = self._ar()
        reaction_activation = self._reaction_activation()
        rate = ar * torch.exp(-reaction_activation / (tb_k + eps)) * (het_total - he_state) / (het_total + eps)
        return torch.clamp(rate, min=0.0)

    def step(
        self,
        tb_c: torch.Tensor,
        xb: torch.Tensor,
        he_state: torch.Tensor,
        trp_c: torch.Tensor,
        tgo_prev_c: torch.Tensor,
        tgi_c: torch.Tensor,
        vg: torch.Tensor,
        tt: torch.Tensor,
        dt_seconds: torch.Tensor,
        batch_mass_kg: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        eps = 1e-6
        tb_k = tb_c + 273.15
        tgi_k = tgi_c + 273.15
        tgo_prev_k = tgo_prev_c + 273.15

        mb_dry = self._mb_dry(batch_mass_kg)
        ab = self._ab()
        tg_mean_prev_k = 0.5 * (tgi_k + tgo_prev_k)
        air_density = self._rho_g(tg_mean_prev_k)
        gg = self._air_area() * air_density * torch.clamp(vg, min=0.1)

        he = self.he(tg_mean_prev_k - 273.15, vg, tt)
        cpg = self._cpg(tg_mean_prev_k)
        tgo_k = tgi_k - (tgi_k - tb_k) * (1.0 - torch.exp(-(he * ab) / (gg * cpg + eps)))
        tg_mean_k = 0.5 * (tgi_k + tgo_k)
        cpg = self._cpg(tg_mean_k)
        cpb = self._cpb(xb)

        dxb_dt = self._compute_dxb_dt(xb, tb_k, eps)
        het_total = self._het_total()
        dhe_dt = self._compute_dhe_dt(he_state, tb_k, het_total, eps)

        hv = self._hv()
        phi_gb = gg * cpg * (tgi_k - tgo_k)
        phi_r = mb_dry * dhe_dt
        phi_ev = hv * (-dxb_dt) * mb_dry
        numerator = phi_gb + phi_r - phi_ev
        denominator = mb_dry * (1.0 + xb) * cpb + eps
        dtb_dt = numerator / denominator

        probe_k = torch.nn.functional.softplus(self.log_probe_k)
        dtrp_dt = probe_k * (tb_c - trp_c)

        tb_next = tb_c + dt_seconds * dtb_dt
        xb_next = torch.clamp(xb + dt_seconds * dxb_dt, min=0.001, max=0.25)
        he_state_next = torch.clamp(he_state + dt_seconds * dhe_dt, min=0.0)
        he_state_next = torch.minimum(he_state_next, het_total)
        trp_next = trp_c + dt_seconds * dtrp_dt
        tgo_c = tgo_k - 273.15
        return tb_next, xb_next, he_state_next, trp_next, tgo_c

    def initialize_states(self, batch: FullRoastBatch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        init_features = torch.stack(
            [
                (batch.tc[:, 0] - 160.0) / 60.0,
                (batch.t1[:, 0] - 220.0) / 120.0,
                (batch.t2[:, 0] - 150.0) / 100.0,
                batch.flow_gas[:, 0] / 100.0,
                batch.air_speed[:, 0] / 20.0,
                batch.drum_speed[:, 0] / 40.0,
            ],
            dim=-1,
        )
        deltas = self.init_net(init_features)
        if self.fixed_initial_bean_temp_c is not None:
            tb = torch.full_like(batch.tc[:, 0], self.fixed_initial_bean_temp_c)
        else:
            tb = batch.tc[:, 0] + self.init_tb_offset + 12.0 * torch.tanh(deltas[:, 0])
        trp = batch.tc[:, 0] + 3.0 * torch.tanh(deltas[:, 1])
        if self.fixed_initial_moisture_ratio is not None:
            xb = torch.full_like(batch.tc[:, 0], self.fixed_initial_moisture_ratio)
        else:
            xb = 0.02 + 0.12 * torch.sigmoid(self.init_xb_logit + deltas[:, 2])
        he_state = torch.nn.functional.softplus(deltas[:, 3])
        return tb, xb, he_state, trp

    def rollout_batch(
        self,
        batch: FullRoastBatch,
        teacher_forced: bool = False,
        detach_state: bool = False,
        warmup_steps: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = batch.tc.shape[0]
        steps = batch.dt_seconds.shape[1]
        preds = torch.zeros((batch_size, steps), dtype=torch.float32, device=batch.tc.device)
        tgo_series = torch.zeros((batch_size, steps), dtype=torch.float32, device=batch.tc.device)
        he_series = torch.zeros((batch_size, steps), dtype=torch.float32, device=batch.tc.device)
        tb_series = torch.zeros((batch_size, steps), dtype=torch.float32, device=batch.tc.device)

        tb, xb, he_state, trp = self.initialize_states(batch)
        # T2 is the measured inlet-air temperature into the drum; the outlet-air state is latent.
        tgo_prev = 0.5 * (batch.t2[:, 0] + tb)

        for t in range(steps):
            use_teacher = teacher_forced or t < warmup_steps
            if use_teacher:
                trp = batch.tc[:, t]
                if self.fixed_initial_bean_temp_c is None:
                    tb = batch.tc[:, t] + self.init_tb_offset
                tgo_prev = 0.5 * (batch.t2[:, t] + tb)
            tb, xb, he_state, trp, tgo = self.step(
                tb,
                xb,
                he_state,
                trp,
                tgo_prev,
                batch.t2[:, t],
                batch.air_speed[:, t],
                batch.drum_speed[:, t],
                batch.dt_seconds[:, t],
                batch.batch_mass_kg,
            )
            preds[:, t] = trp
            tgo_series[:, t] = tgo
            he_series[:, t] = he_state
            tb_series[:, t] = tb
            tgo_prev = tgo
            if detach_state:
                tb = tb.detach()
                xb = xb.detach()
                he_state = he_state.detach()
                trp = trp.detach()
                tgo_prev = tgo_prev.detach()
        return preds, tgo_series, he_series, tb_series

    @torch.no_grad()
    def predict_rollout(self, seq: FullRoastSequence) -> List[float]:
        batch = _make_batch([seq], next(self.parameters()).device)
        preds, _, _, _ = self.rollout_batch(batch, teacher_forced=False)
        return preds[0, batch.mask[0]].detach().cpu().tolist()

    @torch.no_grad()
    def predict_rollout_with_warmup(self, seq: FullRoastSequence, warmup_steps: int = 0) -> List[float]:
        batch = _make_batch([seq], next(self.parameters()).device)
        preds, _, _, _ = self.rollout_batch(batch, teacher_forced=False, warmup_steps=warmup_steps)
        series = preds[0, batch.mask[0]].detach().cpu().tolist()
        return series[warmup_steps:]

    @torch.no_grad()
    def predict_one_step(self, seq: FullRoastSequence) -> List[float]:
        batch = _make_batch([seq], next(self.parameters()).device)
        preds, _, _, _ = self.rollout_batch(batch, teacher_forced=True)
        return preds[0, batch.mask[0]].detach().cpu().tolist()


class ConstantHeFullStateModel(FullStateRoasterModel):
    model_name = "whitebox_constant_he_fullstate"

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.log_he = nn.Parameter(torch.log(torch.tensor(35.0)))

    def he(self, tg_c: torch.Tensor, vg: torch.Tensor, tt: torch.Tensor) -> torch.Tensor:
        return torch.exp(self.log_he)


class LearnedHeFullStateModel(FullStateRoasterModel):
    model_name = "greybox_learned_he_fullstate"

    def __init__(
        self,
        *,
        hidden_widths: Sequence[int] = (128, 64, 32),
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        self.log_base_he = nn.Parameter(torch.log(torch.tensor(35.0)))
        widths = tuple(int(w) for w in hidden_widths)
        if not widths:
            raise ValueError("hidden_widths must contain at least one layer")
        layers: list[nn.Module] = []
        prev = 3
        for width in widths:
            layers.append(nn.Linear(prev, width))
            layers.append(nn.ReLU())
            prev = width
        layers.append(nn.Linear(prev, 1))
        self.he_net = nn.Sequential(*layers)
        self.hidden_widths = widths
        final_linear = self.he_net[-1]
        if isinstance(final_linear, nn.Linear):
            nn.init.zeros_(final_linear.weight)
            nn.init.zeros_(final_linear.bias)

    def he(self, tg_c: torch.Tensor, vg: torch.Tensor, tt: torch.Tensor) -> torch.Tensor:
        features = torch.stack(
            [
                (tg_c - 250.0) / 120.0,
                vg / 10.0,
                tt / 35.0,
            ],
            dim=-1,
        )
        raw = self.he_net(features)
        base = torch.exp(self.log_base_he)
        correction = torch.exp(1.5 * torch.tanh(raw).squeeze(-1))
        return torch.clamp(base * correction, min=1.0, max=250.0)


class MultiClosureFullStateModel(FullStateRoasterModel):
    """Position 4: physics-informed model with three learned closure terms.

    Replaces three physical relations with bounded learned MLPs while preserving
    the rest of the mechanistic state evolution:

        * h_e (heat-transfer closure)          — same parameterization as
                                                  LearnedHeFullStateModel
        * dx_b/dt (moisture-loss rate)         — learned, non-positive bounded
        * dH_e/dt (exothermic reaction rate)   — learned, non-negative bounded

    Each learned closure is initialized so the zero-init network produces a
    rate at roughly the literature-default magnitude — i.e. the multi-closure
    model behaves like the single-closure PI model at the start of training.
    """

    model_name = "multi_closure_pi"

    def __init__(
        self,
        *,
        he_hidden_widths: Sequence[int] = (128, 64, 32),
        moisture_hidden_widths: Sequence[int] = (32, 16),
        reaction_hidden_widths: Sequence[int] = (32, 16),
        moisture_scale: float = 2.74e-5,
        reaction_scale: float = 75.0,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        # --- heat-transfer closure (same as LearnedHeFullStateModel) ---
        self.log_base_he = nn.Parameter(torch.log(torch.tensor(35.0)))
        self.he_hidden_widths = tuple(int(w) for w in he_hidden_widths)
        self.he_net = self._build_closure_mlp(input_dim=3, hidden_widths=self.he_hidden_widths)

        # --- moisture-loss rate closure (output magnitude ~1e-5 at zero init) ---
        self.moisture_hidden_widths = tuple(int(w) for w in moisture_hidden_widths)
        self.moisture_net = self._build_closure_mlp(input_dim=2, hidden_widths=self.moisture_hidden_widths)
        self.moisture_scale = float(moisture_scale)

        # --- exothermic reaction rate closure (output magnitude ~50 J/(kg s) at zero init) ---
        self.reaction_hidden_widths = tuple(int(w) for w in reaction_hidden_widths)
        self.reaction_net = self._build_closure_mlp(input_dim=2, hidden_widths=self.reaction_hidden_widths)
        self.reaction_scale = float(reaction_scale)

    @staticmethod
    def _build_closure_mlp(input_dim: int, hidden_widths: Sequence[int]) -> nn.Sequential:
        widths = tuple(int(w) for w in hidden_widths)
        if not widths:
            raise ValueError("hidden_widths must have at least one entry")
        layers: list[nn.Module] = []
        prev = int(input_dim)
        for width in widths:
            layers.append(nn.Linear(prev, width))
            layers.append(nn.ReLU())
            prev = width
        layers.append(nn.Linear(prev, 1))
        net = nn.Sequential(*layers)
        # Zero-init last layer so untrained closure produces a small, sensible default
        # (softplus(0) ≈ 0.693, so initial output ≈ 0.693 × scale).
        final = net[-1]
        if isinstance(final, nn.Linear):
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)
        return net

    def he(self, tg_c: torch.Tensor, vg: torch.Tensor, tt: torch.Tensor) -> torch.Tensor:
        features = torch.stack(
            [
                (tg_c - 250.0) / 120.0,
                vg / 10.0,
                tt / 35.0,
            ],
            dim=-1,
        )
        raw = self.he_net(features)
        base = torch.exp(self.log_base_he)
        correction = torch.exp(1.5 * torch.tanh(raw).squeeze(-1))
        return torch.clamp(base * correction, min=1.0, max=250.0)

    def _compute_dxb_dt(self, xb: torch.Tensor, tb_k: torch.Tensor, eps: float) -> torch.Tensor:
        # Inputs normalized to roughly mean 0, std ~1 over the expected range.
        features = torch.stack(
            [
                (xb - 0.1) / 0.1,
                (tb_k - 400.0) / 100.0,
            ],
            dim=-1,
        )
        raw = self.moisture_net(features).squeeze(-1)
        # Output is strictly non-positive: dxb/dt <= 0 in roasting.
        # softplus(raw) is non-negative; we negate it and scale to literature magnitude.
        return -self.moisture_scale * torch.nn.functional.softplus(raw)

    def _compute_dhe_dt(
        self,
        he_state: torch.Tensor,
        tb_k: torch.Tensor,
        het_total: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        # Reaction state cannot exceed het_total. Use normalized he progress.
        he_frac = torch.clamp(he_state / (het_total + eps), min=0.0, max=1.0)
        features = torch.stack(
            [
                he_frac - 0.5,
                (tb_k - 400.0) / 100.0,
            ],
            dim=-1,
        )
        raw = self.reaction_net(features).squeeze(-1)
        # Non-negative rate; modulated by remaining capacity so the state cannot grow
        # past het_total in the limit.
        remaining_frac = torch.clamp(1.0 - he_frac, min=0.0)
        return self.reaction_scale * torch.nn.functional.softplus(raw) * remaining_frac


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid = values[mask]
    if valid.numel() == 0:
        return torch.zeros((), dtype=torch.float32, device=values.device)
    return torch.mean(valid)


def _batch_loss(
    model: FullStateRoasterModel,
    batch: FullRoastBatch,
    *,
    tgo_weight: float,
    turning_point_weight: float,
    detach_state: bool,
    warmup_steps: int,
) -> torch.Tensor:
    preds, tgo_series, _, tb_series = model.rollout_batch(
        batch,
        teacher_forced=False,
        detach_state=detach_state,
        warmup_steps=warmup_steps,
    )
    mask = batch.mask
    tc_target = batch.tc[:, 1:]
    if warmup_steps > 0:
        warmup_mask = torch.zeros_like(mask)
        warmup_mask[:, : min(warmup_steps, mask.shape[1])] = True
        mask = mask & ~warmup_mask
    tc_loss = _masked_mean((preds - tc_target) ** 2, mask)
    total_loss = tc_loss

    if turning_point_weight > 0.0:
        turn_penalties: List[torch.Tensor] = []
        for row_idx in range(batch.tc.shape[0]):
            valid_steps = int(batch.mask[row_idx].sum().item())
            if valid_steps <= 0:
                continue
            target_slice = batch.tc[row_idx, 1 : valid_steps + 1]
            if target_slice.numel() == 0:
                continue
            turn_index = int(torch.argmin(target_slice).item())
            turn_penalties.append((tb_series[row_idx, turn_index] - preds[row_idx, turn_index]) ** 2)
        if turn_penalties:
            total_loss = total_loss + turning_point_weight * torch.mean(torch.stack(turn_penalties))

    if tgo_weight > 0.0:
        inlet_air = batch.t2[:, :-1]
        bean_proxy = batch.tc[:, :-1]
        lower = torch.minimum(inlet_air, bean_proxy)
        upper = torch.maximum(inlet_air, bean_proxy)
        low_violation = torch.relu(lower - tgo_series)
        high_violation = torch.relu(tgo_series - upper)
        tgo_regularization = _masked_mean(low_violation**2 + high_violation**2, mask)
        total_loss = total_loss + tgo_weight * tgo_regularization

    return total_loss


def train_model(
    model: FullStateRoasterModel,
    train_sequences: Sequence[FullRoastSequence],
    val_sequences: Sequence[FullRoastSequence],
    *,
    epochs: int,
    lr: float,
    weight_decay: float,
    batch_size: int,
    device: str,
    grad_clip: float = 1.0,
    tgo_weight: float = 0.25,
    turning_point_weight: float = 0.0,
    detach_state_steps: bool = True,
    warmup_steps: int = 0,
    patience: int | None = None,
) -> dict[str, object]:
    torch_device = torch.device(device)
    model.to(torch_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val = float("inf")
    best_state = None
    history: List[Dict[str, float]] = []
    epochs_no_improve = 0
    epochs_run = 0
    best_epoch = 0

    for epoch in range(epochs):
        start = time.perf_counter()
        model.train()
        train_batches = make_batches(train_sequences, batch_size=batch_size, device=torch_device, shuffle=True)
        train_losses: List[float] = []
        for batch in train_batches:
            optimizer.zero_grad()
            loss = _batch_loss(
                model,
                batch,
                tgo_weight=tgo_weight,
                turning_point_weight=turning_point_weight,
                detach_state=detach_state_steps,
                warmup_steps=warmup_steps,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        model.eval()
        with torch.no_grad():
            val_batches = make_batches(val_sequences, batch_size=batch_size, device=torch_device, shuffle=False) if val_sequences else []
            val_losses = [
                float(
                    _batch_loss(
                        model,
                        batch,
                        tgo_weight=tgo_weight,
                        turning_point_weight=turning_point_weight,
                        detach_state=detach_state_steps,
                        warmup_steps=warmup_steps,
                    ).detach().cpu()
                )
                for batch in val_batches
            ]

        train_mean = sum(train_losses) / max(len(train_losses), 1)
        val_mean = sum(val_losses) / len(val_losses) if val_losses else train_mean
        history.append({"epoch": float(epoch + 1), "train_loss": train_mean, "val_loss": val_mean, "epoch_time_sec": time.perf_counter() - start})

        epochs_run = epoch + 1
        if val_mean < best_val - 1e-9:
            best_val = val_mean
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch + 1
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if patience is not None and epochs_no_improve >= int(patience):
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return {
        "best_val_loss": best_val,
        "best_epoch": int(best_epoch),
        "history": history,
        "batch_size": batch_size,
        "tgo_weight": tgo_weight,
        "turning_point_weight": turning_point_weight,
        "detach_state_steps": detach_state_steps,
        "warmup_steps": warmup_steps,
        "patience": int(patience) if patience is not None else None,
        "epochs_run": int(epochs_run),
    }


def evaluate_model(model: FullStateRoasterModel, sequences: Sequence[FullRoastSequence], *, warmup_steps: int = 0) -> Dict[str, Metrics]:
    one_true: List[float] = []
    one_pred: List[float] = []
    roll_true: List[float] = []
    roll_pred: List[float] = []
    for seq in sequences:
        if seq.length < 2:
            continue
        start_index = min(max(warmup_steps, 0), seq.length - 1)
        target = seq.tc[start_index + 1 :]
        one = model.predict_one_step(seq)
        roll = model.predict_rollout_with_warmup(seq, warmup_steps=warmup_steps)
        if start_index > 0:
            one = one[start_index:]
        one_true.extend(target)
        one_pred.extend(one)
        roll_true.extend(target)
        roll_pred.extend(roll)
    return {"one_step": compute_metrics(one_true, one_pred), "rollout": compute_metrics(roll_true, roll_pred)}


def build_run_result(
    models: Dict[str, FullStateRoasterModel],
    test_sequences: Sequence[FullRoastSequence],
    *,
    warmup_steps: int = 0,
) -> FullStateRunResult:
    metrics = {name: evaluate_model(model, test_sequences, warmup_steps=warmup_steps) for name, model in models.items()}
    representative = max(test_sequences, key=lambda seq: seq.length) if test_sequences else None

    predictions: Dict[str, Dict[str, List[float]]] = {}
    representative_roast_id = representative.roast_id if representative else ""
    for name, model in models.items():
        predictions[name] = {
            "one_step": model.predict_one_step(representative)[warmup_steps:] if representative else [],
            "rollout": model.predict_rollout_with_warmup(representative, warmup_steps=warmup_steps) if representative else [],
        }
    if representative:
        predictions["actual"] = {
            "one_step": representative.tc[1 + warmup_steps :],
            "rollout": representative.tc[1 + warmup_steps :],
        }

    return FullStateRunResult(
        metrics=metrics,
        representative_roast_id=representative_roast_id,
        predictions=predictions,
    )


def metrics_to_serializable(metrics: Dict[str, Dict[str, Metrics]]) -> Dict[str, Dict[str, Dict[str, float]]]:
    return {
        model_name: {
            mode: {"r2": float(metric.r2), "rmse": float(metric.rmse), "mae": float(metric.mae), "n": int(metric.n)}
            for mode, metric in model_metrics.items()
        }
        for model_name, model_metrics in metrics.items()
    }


def save_checkpoint(path: Path, model: FullStateRoasterModel, metadata: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_name": model.model_name, "state_dict": model.state_dict(), "metadata": metadata}, path)


def load_checkpoint(path: Path, model: FullStateRoasterModel, device: str = "cpu") -> Dict[str, object]:
    payload = torch.load(path, map_location=device)
    state_dict = payload.get("state_dict", {})
    model.load_state_dict(state_dict)
    model.to(torch.device(device))
    model.eval()
    metadata = payload.get("metadata", {})
    return metadata if isinstance(metadata, dict) else {}


def write_metrics_report(out_dir: Path, metrics: Dict[str, Dict[str, Metrics]], metadata: Dict[str, object]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"metrics": metrics_to_serializable(metrics), "metadata": metadata}
    (out_dir / "thesis_fullstate_metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    rows = [
        "| Model | One-step R2 | Rollout R2 | One-step RMSE | Rollout RMSE |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in ["whitebox_constant_he_fullstate", "greybox_learned_he_fullstate"]:
        m = metrics[name]
        rows.append(f"| {name} | {m['one_step'].r2:.4f} | {m['rollout'].r2:.4f} | {m['one_step'].rmse:.4f} | {m['rollout'].rmse:.4f} |")
    (out_dir / "thesis_fullstate_metrics.md").write_text("\n".join(rows) + "\n", encoding="utf-8")
