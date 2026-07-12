#!/usr/bin/env python
"""Val-selected random-search HPO sweep over manuscript model classes.

This script replaces the legacy test-set-selected tune loops with a clean,
symmetric sweep on the manuscript's 221-roast p2/P10/P13/P16 production
cohort. Trial selection uses rollout R^2 on the validation split only; the
test split is touched exclusively during the final per-seed retraining step.

Model classes covered (one in scope per the current manuscript):
    - blackbox: matched-input AutoRegressiveMLP (prev Tc + T2 + v_g + TT)
    - greybox:  LearnedHeFullStateModel with configurable closure widths
    - residual: ResidualLSTMModel stacked on the seed-matched grey-box base

Outputs (under --outdir, default reports/manuscript_hpo):
    <model_class>/all_trials.jsonl       per-trial config + per-seed val R^2
    <model_class>/best_config.json       winning config + summary
    <model_class>/sweep_summary.md       markdown table sorted by mean val R^2
    final_test_metrics.json              test metrics for best configs over seeds
    final_test_metrics.md                markdown summary of the test metrics
    environment.json                     versions + git SHA + CUDA info
    search_state.json                    search seed and search-space spec
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import math
import platform
import random
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from roaster_piml.blackbox import (  # noqa: E402
    AutoRegressiveMLP,
    count_parameters as bb_count_parameters,
    one_step_blackbox_trial,
    rollout_blackbox_trial,
    train_blackbox,
)
from roaster_piml.io_utils import stable_hash_bucket  # noqa: E402
from roaster_piml.model import Metrics, compute_metrics  # noqa: E402
from roaster_piml.roast_filters import build_roast_metadata_index, filter_sequences  # noqa: E402
from roaster_piml.roast_window import detect_modeled_window_bounds  # noqa: E402
from roaster_piml.thesis_full_state import (  # noqa: E402
    ConstantHeFullStateModel,
    FullRoastSequence,
    LearnedHeFullStateModel,
    MultiClosureFullStateModel,
    evaluate_model as evaluate_fullstate_model,
    load_full_roast_sequences,
    set_seed as set_torch_seed,
    split_sequences_train_val_test,
    train_model as train_fullstate_model,
)
from roaster_piml.thesis_residual import (  # noqa: E402
    ResidualFeedForwardModel,
    ResidualLSTMModel,
    evaluate_residual_model,
    train_residual_model,
)


# ---------------------------------------------------------------------------
# Manuscript-cohort constants
# ---------------------------------------------------------------------------

CORE_FEATURES = ["t2", "air_speed", "drum_speed"]
DEFAULT_P_CODES = ["P10", "P13", "P16"]
DEFAULT_SOURCE_BUCKETS = ["p2"]
PER_TRIAL_SEEDS = (11, 23)
FINAL_SEEDS = (11, 23, 37)
WARMUP_STEPS = 0  # manuscript reports warmup-free rollouts on this cohort
RESIDUAL_BASE_SEED = 11

MODEL_CLASSES = ("blackbox", "greybox", "residual")
MODEL_NAME_BY_CLASS = {
    "blackbox": "blackbox_core",
    "greybox": "greybox_learned_he_fullstate",
    "residual": "residual_lstm_on_greybox",
}
WHITEBOX_MODEL_NAME = "whitebox_constant_he_fullstate"


# ---------------------------------------------------------------------------
# Search-space samplers (random search; deterministic given numpy Generator)
# ---------------------------------------------------------------------------

BLACKBOX_HIDDEN_CHOICES: tuple[tuple[int, ...], ...] = (
    (32, 16),
    (64, 32),
    (128, 64),
    (64, 32, 16),
    (128, 64, 32),
    (256, 128, 64),
)
GREYBOX_HIDDEN_CHOICES: tuple[tuple[int, ...], ...] = (
    (64, 32),
    (128, 64, 32),
    (256, 128, 64, 32),
    (128, 64),
    (64, 64, 64),
)


# Sweep-time epoch budgets are capped below the manuscript's canonical training
# budget so the random search fits in an overnight window. The winner of each
# sweep is retrained at the canonical budget below for the final test eval.
SWEEP_EPOCH_CAPS = {"blackbox": 100, "greybox": 100, "residual": 100, "whitebox": 100, "multi_closure": 100, "residual_ff": 100}
SWEEP_PATIENCE = {"blackbox": 15, "greybox": 15, "residual": 15, "whitebox": 15, "multi_closure": 15, "residual_ff": 15}

# Manuscript-canonical (final-eval) training budgets — applied when the winning
# config is retrained across FINAL_SEEDS for the test-set report.
MANUSCRIPT_FINAL_EPOCHS = {"blackbox": 200, "greybox": 300, "residual": 200, "whitebox": 300, "multi_closure": 300, "residual_ff": 200}
MANUSCRIPT_FINAL_PATIENCE = {"blackbox": 25, "greybox": 30, "residual": 25, "whitebox": 30, "multi_closure": 30, "residual_ff": 25}


def sample_blackbox_config(rng: np.random.Generator) -> dict[str, Any]:
    hidden = BLACKBOX_HIDDEN_CHOICES[int(rng.integers(len(BLACKBOX_HIDDEN_CHOICES)))]
    lr = float(math.exp(rng.uniform(math.log(1e-4), math.log(3e-3))))
    weight_decay = float(rng.choice([0.0, 1e-5, 1e-4]))
    bs_token = rng.choice(["16", "32", "64", "full"])
    batch_size: int | str = "full" if bs_token == "full" else int(bs_token)
    dropout = float(rng.choice([0.0, 0.1]))
    return {
        "hidden_widths": list(hidden),
        "lr": lr,
        "weight_decay": weight_decay,
        "batch_size": batch_size,
        "dropout": dropout,
        "epochs": SWEEP_EPOCH_CAPS["blackbox"],
        "patience": SWEEP_PATIENCE["blackbox"],
    }


def sample_greybox_config(rng: np.random.Generator) -> dict[str, Any]:
    hidden = GREYBOX_HIDDEN_CHOICES[int(rng.integers(len(GREYBOX_HIDDEN_CHOICES)))]
    lr = float(math.exp(rng.uniform(math.log(1e-4), math.log(2e-3))))
    weight_decay = float(rng.choice([1e-6, 1e-5, 1e-4]))
    batch_size = int(rng.choice([16, 32]))  # dropped bs=8 (too slow on this hardware)
    return {
        "hidden_widths": list(hidden),
        "lr": lr,
        "weight_decay": weight_decay,
        "batch_size": batch_size,
        "epochs": SWEEP_EPOCH_CAPS["greybox"],
        "patience": SWEEP_PATIENCE["greybox"],
        "tgo_weight": 0.0,
        "detach_state_steps": False,
        "warmup_steps": WARMUP_STEPS,
    }


MULTI_CLOSURE_HE_CHOICES: tuple[tuple[int, ...], ...] = (
    (64, 32),
    (128, 64, 32),
    (128, 64),
    (64, 64, 64),
)
MULTI_CLOSURE_RATE_CHOICES: tuple[tuple[int, ...], ...] = (
    (16, 8),
    (32, 16),
    (64, 32),
)


def sample_multi_closure_config(rng: np.random.Generator) -> dict[str, Any]:
    """Position 4 sweep: multi-closure PI as a repair for the broken scaffold.

    The paper's central question is how PIML repairs the rollout failure that
    arises from latent states and missing metadata. Position 4 uses the SAME
    broken-scaffold setup as the manuscript's PI baseline — no literature
    priors, init_net learnable — but replaces three physical relations
    (h_e, dx_b/dt, dH_e/dt) with bounded learned MLPs. Tests whether more
    physics-side learning improves the repair beyond a single h_e closure.
    """
    he_widths = MULTI_CLOSURE_HE_CHOICES[int(rng.integers(len(MULTI_CLOSURE_HE_CHOICES)))]
    moisture_widths = MULTI_CLOSURE_RATE_CHOICES[int(rng.integers(len(MULTI_CLOSURE_RATE_CHOICES)))]
    reaction_widths = MULTI_CLOSURE_RATE_CHOICES[int(rng.integers(len(MULTI_CLOSURE_RATE_CHOICES)))]
    lr = float(math.exp(rng.uniform(math.log(1e-4), math.log(2e-3))))
    weight_decay = float(rng.choice([1e-6, 1e-5, 1e-4]))
    batch_size = int(rng.choice([16, 32]))
    return {
        "he_hidden_widths": list(he_widths),
        "moisture_hidden_widths": list(moisture_widths),
        "reaction_hidden_widths": list(reaction_widths),
        "lr": lr,
        "weight_decay": weight_decay,
        "batch_size": batch_size,
        "epochs": SWEEP_EPOCH_CAPS["greybox"],
        "patience": SWEEP_PATIENCE["greybox"],
        "tgo_weight": 0.0,
        "detach_state_steps": False,
        "warmup_steps": WARMUP_STEPS,
        # No literature priors — out of scope for the "repair the broken scaffold" framing.
        "fixed_initial_bean_temp_c": None,
        "fixed_initial_moisture_ratio": None,
        "freeze_init_net": False,
    }


def sample_greybox_bs8_config(rng: np.random.Generator) -> dict[str, Any]:
    """Focused bs=8 grey-box search. Sweep-time epoch cap is bumped to 200 so
    bs=8 trials get a real shot at convergence (the bs=8 probe shows best-val
    epoch can be around 190). Only the larger architectures are sampled — at
    bs=8 the per-trial cost is high, so we focus on configurations the
    exploratory probe suggested could matter."""
    hidden_choices = [(128, 64, 32), (256, 128, 64, 32), (64, 64, 64), (128, 64)]
    hidden = hidden_choices[int(rng.integers(len(hidden_choices)))]
    lr = float(math.exp(rng.uniform(math.log(3e-4), math.log(2e-3))))
    weight_decay = float(rng.choice([1e-6, 1e-5, 1e-4]))
    return {
        "hidden_widths": list(hidden),
        "lr": lr,
        "weight_decay": weight_decay,
        "batch_size": 8,
        "epochs": 200,
        "patience": 30,
        "tgo_weight": 0.0,
        "detach_state_steps": False,
        "warmup_steps": WARMUP_STEPS,
    }


def sample_whitebox_config(rng: np.random.Generator) -> dict[str, Any]:
    """Sample a white-box (constant-He) training schedule.

    The white-box has no closure MLP — only scalar physical parameters and the
    small shared initialization network. So the sweep varies only the training
    schedule (lr, weight_decay, batch_size).
    """
    lr = float(math.exp(rng.uniform(math.log(1e-4), math.log(2e-3))))
    weight_decay = float(rng.choice([1e-6, 1e-5, 1e-4]))
    batch_size = int(rng.choice([16, 32]))
    return {
        "lr": lr,
        "weight_decay": weight_decay,
        "batch_size": batch_size,
        "epochs": SWEEP_EPOCH_CAPS["whitebox"],
        "patience": SWEEP_PATIENCE["whitebox"],
        "tgo_weight": 0.0,
        "detach_state_steps": False,
        "warmup_steps": WARMUP_STEPS,
    }


def sample_residual_config(rng: np.random.Generator) -> dict[str, Any]:
    hidden_size = int(rng.choice([32, 48, 64, 96]))
    residual_weight = float(math.exp(rng.uniform(math.log(3e-4), math.log(3e-3))))
    max_delta = float(rng.choice([12.0, 18.0, 24.0]))
    lr = float(math.exp(rng.uniform(math.log(2e-4), math.log(1e-3))))
    return {
        "hidden_size": hidden_size,
        "residual_weight": residual_weight,
        "max_delta": max_delta,
        "lr": lr,
        "weight_decay": 1e-5,
        "batch_size": 16,  # promoted from 8 to align with grey-box change
        "epochs": SWEEP_EPOCH_CAPS["residual"],
        "patience": SWEEP_PATIENCE["residual"],
        "warmup_steps": WARMUP_STEPS,
    }


RESIDUAL_FF_HIDDEN_CHOICES: tuple[tuple[int, ...], ...] = (
    (16, 8),
    (32, 16),
    (64, 32),
    (32, 16, 8),
    (64, 32, 16),
    (128, 64),
)


def sample_residual_ff_config(rng: np.random.Generator) -> dict[str, Any]:
    """Bounded feedforward residual sweep (position 4 in the FF-only spectrum).

    Search space mirrors the LSTM residual sweep where applicable so the two
    residual variants are directly comparable, except hidden architecture is
    a feedforward MLP rather than an LSTM hidden size.
    """
    hidden = RESIDUAL_FF_HIDDEN_CHOICES[int(rng.integers(len(RESIDUAL_FF_HIDDEN_CHOICES)))]
    residual_weight = float(math.exp(rng.uniform(math.log(3e-4), math.log(3e-3))))
    max_delta = float(rng.choice([12.0, 18.0, 24.0]))
    lr = float(math.exp(rng.uniform(math.log(2e-4), math.log(1e-3))))
    weight_decay = float(rng.choice([1e-6, 1e-5, 1e-4]))
    batch_size = int(rng.choice([16, 32]))
    return {
        "hidden_widths": list(hidden),
        "residual_weight": residual_weight,
        "max_delta": max_delta,
        "lr": lr,
        "weight_decay": weight_decay,
        "batch_size": batch_size,
        "epochs": SWEEP_EPOCH_CAPS["residual_ff"],
        "patience": SWEEP_PATIENCE["residual_ff"],
        "warmup_steps": WARMUP_STEPS,
    }


def sample_residual_ff_unbounded_config(rng: np.random.Generator) -> dict[str, Any]:
    """Unbounded feedforward residual sweep (position 5 in the FF-only spectrum).

    Identical to ``sample_residual_ff_config`` except ``max_delta`` is fixed at
    a large value (1000 K) so the tanh saturation never effectively bounds the
    correction — the FF MLP can produce any prediction. This separates the
    'bounded physics constraint' position from the 'physics-as-base + free
    learning' position on the same architecture family.
    """
    hidden = RESIDUAL_FF_HIDDEN_CHOICES[int(rng.integers(len(RESIDUAL_FF_HIDDEN_CHOICES)))]
    residual_weight = float(math.exp(rng.uniform(math.log(3e-4), math.log(3e-3))))
    lr = float(math.exp(rng.uniform(math.log(2e-4), math.log(1e-3))))
    weight_decay = float(rng.choice([1e-6, 1e-5, 1e-4]))
    batch_size = int(rng.choice([16, 32]))
    return {
        "hidden_widths": list(hidden),
        "residual_weight": residual_weight,
        "max_delta": 1000.0,  # effectively unbounded — tanh saturates beyond physical range
        "lr": lr,
        "weight_decay": weight_decay,
        "batch_size": batch_size,
        "epochs": SWEEP_EPOCH_CAPS["residual_ff"],
        "patience": SWEEP_PATIENCE["residual_ff"],
        "warmup_steps": WARMUP_STEPS,
    }


SEARCH_SPACE_SPEC: dict[str, Any] = {
    "whitebox": {
        "lr": {"distribution": "log_uniform", "low": 1e-4, "high": 2e-3},
        "weight_decay_choices": [1e-6, 1e-5, 1e-4],
        "batch_size_choices": [16, 32],
        "sweep_epochs": SWEEP_EPOCH_CAPS["whitebox"],
        "sweep_patience": SWEEP_PATIENCE["whitebox"],
        "final_epochs": MANUSCRIPT_FINAL_EPOCHS["whitebox"],
        "final_patience": MANUSCRIPT_FINAL_PATIENCE["whitebox"],
        "tgo_weight": 0.0,
        "detach_state_steps": False,
        "warmup_steps": WARMUP_STEPS,
        "note": "white-box has no closure MLP; only training schedule is varied.",
    },
    "blackbox": {
        "hidden_widths_choices": [list(w) for w in BLACKBOX_HIDDEN_CHOICES],
        "lr": {"distribution": "log_uniform", "low": 1e-4, "high": 3e-3},
        "weight_decay_choices": [0.0, 1e-5, 1e-4],
        "batch_size_choices": [16, 32, 64, "full"],
        "dropout_choices": [0.0, 0.1],
        "sweep_epochs": SWEEP_EPOCH_CAPS["blackbox"],
        "sweep_patience": SWEEP_PATIENCE["blackbox"],
        "final_epochs": MANUSCRIPT_FINAL_EPOCHS["blackbox"],
        "final_patience": MANUSCRIPT_FINAL_PATIENCE["blackbox"],
    },
    "greybox": {
        "hidden_widths_choices": [list(w) for w in GREYBOX_HIDDEN_CHOICES],
        "lr": {"distribution": "log_uniform", "low": 1e-4, "high": 2e-3},
        "weight_decay_choices": [1e-6, 1e-5, 1e-4],
        "batch_size_choices": [16, 32],
        "sweep_epochs": SWEEP_EPOCH_CAPS["greybox"],
        "sweep_patience": SWEEP_PATIENCE["greybox"],
        "final_epochs": MANUSCRIPT_FINAL_EPOCHS["greybox"],
        "final_patience": MANUSCRIPT_FINAL_PATIENCE["greybox"],
        "tgo_weight": 0.0,
        "detach_state_steps": False,
        "warmup_steps": WARMUP_STEPS,
    },
    "residual": {
        "hidden_size_choices": [32, 48, 64, 96],
        "residual_weight": {"distribution": "log_uniform", "low": 3e-4, "high": 3e-3},
        "max_delta_choices": [12.0, 18.0, 24.0],
        "lr": {"distribution": "log_uniform", "low": 2e-4, "high": 1e-3},
        "weight_decay": 1e-5,
        "batch_size": 16,
        "sweep_epochs": SWEEP_EPOCH_CAPS["residual"],
        "sweep_patience": SWEEP_PATIENCE["residual"],
        "final_epochs": MANUSCRIPT_FINAL_EPOCHS["residual"],
        "final_patience": MANUSCRIPT_FINAL_PATIENCE["residual"],
        "warmup_steps": WARMUP_STEPS,
    },
}


# ---------------------------------------------------------------------------
# Cohort loading
# ---------------------------------------------------------------------------


def load_manuscript_cohort(
    input_path: Path,
    metadata_path: Path | None,
    include_p_codes: Sequence[str],
    include_source_buckets: Sequence[str],
) -> list[FullRoastSequence]:
    """Load and filter the manuscript cohort sequences.

    The default input is ``data/processed/roast_timeseries_p2_only.csv``, the
    pre-filtered p2-only timeseries that reproduces the manuscript's exact
    221-roast cohort (150/35/36 under md5 split). When that file is used we
    skip the source-bucket filter entirely because the file is already
    bucket-scoped, which also side-steps a stem-collision bug in
    ``build_roast_metadata_index`` exposed by the broader
    ``normalized_files.csv``.

    If a caller points ``input_path`` at the full ``roast_timeseries.csv`` and
    supplies ``metadata_path``, the source-bucket filter is applied via
    ``build_roast_metadata_index`` + ``filter_sequences`` as before.
    """
    sequences = load_full_roast_sequences(
        input_path,
        gas_temp_unit="celsius",
        trim_mode="none",
        align_charge_start=False,
        modeled_window=True,
    )
    # When the input is already a pre-filtered bucket-specific timeseries, the
    # source-bucket filter is a no-op AND would trigger the metadata-index
    # collision bug described above. Skip it; rely on the p_code filter only.
    is_prefiltered = input_path.name.endswith("_p2_only.csv")
    metadata_index = (
        {}
        if (is_prefiltered or metadata_path is None)
        else build_roast_metadata_index(metadata_path)
    )
    return filter_sequences(
        sequences,
        metadata_index=metadata_index,
        include_p_codes=include_p_codes,
        include_source_buckets=[] if is_prefiltered else include_source_buckets,
    )


def split_cohort(
    sequences: Sequence[FullRoastSequence],
) -> tuple[list[FullRoastSequence], list[FullRoastSequence], list[FullRoastSequence]]:
    return split_sequences_train_val_test(sequences, strategy="md5")


def load_grouped_dataframe(
    input_path: Path,
    roast_ids: Sequence[str],
    feature_columns: Sequence[str],
) -> dict[str, pd.DataFrame]:
    roast_id_set = set(roast_ids)
    df = pd.read_csv(input_path)
    df = df[df["roast_id"].isin(roast_id_set)].copy()
    required = ["roast_id", "timestamp", "tc", *feature_columns]
    df = df[required].dropna()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["roast_id", "timestamp"])
    grouped: dict[str, pd.DataFrame] = {}
    for roast_id, roast_df in df.groupby("roast_id", sort=True):
        tc_values = roast_df["tc"].to_numpy(dtype=float).tolist()
        start_idx, end_idx = detect_modeled_window_bounds(tc_values)
        roast_df = roast_df.iloc[start_idx:end_idx].copy()
        if len(roast_df) < 20:
            continue
        grouped[str(roast_id)] = roast_df.reset_index(drop=True)
    return grouped


# ---------------------------------------------------------------------------
# Test-set leakage guard
# ---------------------------------------------------------------------------


class CohortIds:
    """Frozen sets of train/val/test roast IDs with explicit leakage guard."""

    def __init__(
        self,
        train_ids: Sequence[str],
        val_ids: Sequence[str],
        test_ids: Sequence[str],
    ) -> None:
        self.train: tuple[str, ...] = tuple(train_ids)
        self.val: tuple[str, ...] = tuple(val_ids)
        self.test: tuple[str, ...] = tuple(test_ids)
        self._test_set = frozenset(self.test)

    def assert_no_test_leakage(self, roast_ids: Sequence[str], context: str) -> None:
        offenders = [rid for rid in roast_ids if rid in self._test_set]
        if offenders:
            raise AssertionError(
                f"Test-set leakage detected in {context}: {offenders[:5]}"
                + (f" ... (+{len(offenders) - 5} more)" if len(offenders) > 5 else "")
            )


def filter_by_id(
    sequences: Sequence[FullRoastSequence], ids: Sequence[str]
) -> list[FullRoastSequence]:
    wanted = set(ids)
    return [seq for seq in sequences if seq.roast_id in wanted]


# ---------------------------------------------------------------------------
# Pooled rollout evaluation helpers
# ---------------------------------------------------------------------------


def pooled_rollout_metrics_fullstate(
    model: LearnedHeFullStateModel | ConstantHeFullStateModel,
    sequences: Sequence[FullRoastSequence],
) -> tuple[Metrics, Metrics, dict[str, dict[str, list[float]]]]:
    one_true: list[float] = []
    one_pred: list[float] = []
    roll_true: list[float] = []
    roll_pred: list[float] = []
    per_roast: dict[str, dict[str, list[float]]] = {}
    for seq in sequences:
        if seq.length < 2:
            continue
        target = seq.tc[1:]
        one = model.predict_one_step(seq)
        roll = model.predict_rollout_with_warmup(seq, warmup_steps=WARMUP_STEPS)
        one_true.extend(target)
        one_pred.extend(one)
        roll_true.extend(target)
        roll_pred.extend(roll)
        per_roast[seq.roast_id] = {
            "actual": list(target),
            "rollout_pred": list(roll),
            "one_step_pred": list(one),
        }
    return (
        compute_metrics(one_true, one_pred),
        compute_metrics(roll_true, roll_pred),
        per_roast,
    )


def pooled_rollout_metrics_residual(
    residual_model: ResidualLSTMModel,
    base_model: LearnedHeFullStateModel,
    sequences: Sequence[FullRoastSequence],
) -> tuple[Metrics, Metrics, dict[str, dict[str, list[float]]]]:
    one_true: list[float] = []
    one_pred: list[float] = []
    roll_true: list[float] = []
    roll_pred: list[float] = []
    per_roast: dict[str, dict[str, list[float]]] = {}
    for seq in sequences:
        if seq.length < 2:
            continue
        target = seq.tc[1:]
        one = residual_model.predict_one_step(seq, base_model)
        roll = residual_model.predict_rollout_with_warmup(
            seq, base_model, warmup_steps=WARMUP_STEPS
        )
        one_true.extend(target)
        one_pred.extend(one)
        roll_true.extend(target)
        roll_pred.extend(roll)
        per_roast[seq.roast_id] = {
            "actual": list(target),
            "rollout_pred": list(roll),
            "one_step_pred": list(one),
        }
    return (
        compute_metrics(one_true, one_pred),
        compute_metrics(roll_true, roll_pred),
        per_roast,
    )


def pooled_rollout_metrics_blackbox(
    model: AutoRegressiveMLP,
    scalers: dict[str, np.ndarray],
    grouped: dict[str, pd.DataFrame],
    roast_ids: Sequence[str],
    feature_columns: Sequence[str],
    device: str,
) -> tuple[Metrics, Metrics, dict[str, dict[str, list[float]]]]:
    one_true: list[float] = []
    one_pred: list[float] = []
    roll_true: list[float] = []
    roll_pred: list[float] = []
    per_roast: dict[str, dict[str, list[float]]] = {}
    for roast_id in roast_ids:
        roast_df = grouped.get(roast_id)
        if roast_df is None or len(roast_df) < 2:
            continue
        actual, rollout = rollout_blackbox_trial(
            model, scalers, roast_df, feature_columns=feature_columns, device=device
        )
        one_step = one_step_blackbox_trial(
            model, scalers, roast_df, feature_columns=feature_columns, device=device
        )
        target = actual[1:].tolist()
        rollout_pred_after_init = rollout[1:].tolist()
        one_true.extend(target)
        one_pred.extend(one_step.tolist())
        roll_true.extend(target)
        roll_pred.extend(rollout_pred_after_init)
        per_roast[roast_id] = {
            "actual": target,
            "rollout_pred": rollout_pred_after_init,
            "one_step_pred": one_step.tolist(),
        }
    return (
        compute_metrics(one_true, one_pred),
        compute_metrics(roll_true, roll_pred),
        per_roast,
    )


# ---------------------------------------------------------------------------
# Roast-bootstrap CI on pooled rollout R^2
# ---------------------------------------------------------------------------


def roast_bootstrap_rollout_r2(
    per_roast: dict[str, dict[str, list[float]]],
    n_boot: int = 1000,
    seed: int = 0,
) -> tuple[float, float]:
    if not per_roast:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    roast_ids = list(per_roast.keys())
    actuals = {rid: np.asarray(per_roast[rid]["actual"], dtype=float) for rid in roast_ids}
    preds = {rid: np.asarray(per_roast[rid]["rollout_pred"], dtype=float) for rid in roast_ids}
    n_roasts = len(roast_ids)
    boot_r2: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n_roasts, size=n_roasts)
        y_true = np.concatenate([actuals[roast_ids[i]] for i in idx])
        y_pred = np.concatenate([preds[roast_ids[i]] for i in idx])
        denom = np.sum((y_true - np.mean(y_true)) ** 2)
        if denom == 0:
            boot_r2.append(float("nan"))
            continue
        boot_r2.append(1.0 - float(np.sum((y_true - y_pred) ** 2) / denom))
    lo, hi = np.quantile(np.asarray(boot_r2, dtype=float), [0.025, 0.975])
    return float(lo), float(hi)


def per_roast_r2(per_roast: dict[str, dict[str, list[float]]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for rid, payload in per_roast.items():
        y_true = np.asarray(payload["actual"], dtype=float)
        y_pred = np.asarray(payload["rollout_pred"], dtype=float)
        denom = float(np.sum((y_true - np.mean(y_true)) ** 2))
        if denom == 0:
            out[rid] = float("nan")
        else:
            out[rid] = 1.0 - float(np.sum((y_true - y_pred) ** 2) / denom)
    return out


# ---------------------------------------------------------------------------
# Per-model-class trial runner
# ---------------------------------------------------------------------------


@dataclass
class TrialRecord:
    trial_idx: int
    config: dict[str, Any]
    # Headline selection metric — rollout R^2 on the val split.
    per_seed_val_rollout_r2: dict[str, float]
    mean_val_rollout_r2: float
    per_seed_val_rollout_rmse: dict[str, float]
    mean_val_rollout_rmse: float
    # Interpretive fields — support post-hoc comparison of physics-informed vs
    # neural-baseline behavior: convergence speed (history + best_epoch),
    # per-roast failure modes (per_roast_val_r2), and parameter efficiency
    # (param_count vs mean R^2 across trials).
    per_seed_per_roast_val_r2: dict[str, dict[str, float]] = field(default_factory=dict)
    per_seed_history: dict[str, list[dict[str, float]]] = field(default_factory=dict)
    per_seed_best_epoch: dict[str, int] = field(default_factory=dict)
    # Other persistent metadata.
    train_loss_final: float = float("nan")
    val_loss_final: float = float("nan")
    param_count: int = 0
    epochs_run_per_seed: dict[str, int] = field(default_factory=dict)
    wall_time_sec: float = 0.0


def _run_blackbox_trial(
    config: dict[str, Any],
    seeds: Sequence[int],
    cohort_ids: CohortIds,
    grouped: dict[str, pd.DataFrame],
    feature_columns: Sequence[str],
    device: str,
) -> dict[str, Any]:
    cohort_ids.assert_no_test_leakage(cohort_ids.train, "blackbox train")
    cohort_ids.assert_no_test_leakage(cohort_ids.val, "blackbox val")
    per_seed_r2: dict[str, float] = {}
    per_seed_rmse: dict[str, float] = {}
    per_seed_per_roast: dict[str, dict[str, float]] = {}
    per_seed_history: dict[str, list[dict[str, float]]] = {}
    per_seed_best_epoch: dict[str, int] = {}
    per_seed_epochs: dict[str, int] = {}
    last_train_loss = float("nan")
    last_val_loss = float("nan")
    last_param_count = 0
    for seed in seeds:
        model, scalers, meta = train_blackbox(
            grouped,
            cohort_ids.train,
            cohort_ids.val,
            feature_columns=feature_columns,
            hidden_widths=tuple(config["hidden_widths"]),
            dropout=float(config["dropout"]),
            lr=float(config["lr"]),
            weight_decay=float(config["weight_decay"]),
            batch_size=config["batch_size"],
            epochs=int(config["epochs"]),
            patience=int(config["patience"]),
            device=device,
            seed=int(seed),
        )
        _, val_rollout, val_per_roast = pooled_rollout_metrics_blackbox(
            model, scalers, grouped, cohort_ids.val, feature_columns, device
        )
        per_seed_r2[str(seed)] = float(val_rollout.r2)
        per_seed_rmse[str(seed)] = float(val_rollout.rmse)
        per_seed_per_roast[str(seed)] = per_roast_r2(val_per_roast)
        per_seed_history[str(seed)] = meta["history"]
        per_seed_best_epoch[str(seed)] = int(meta.get("best_epoch", 0))
        per_seed_epochs[str(seed)] = int(meta["epochs_run"])
        last_train_loss = (
            float(meta["history"][-1]["train_loss"]) if meta["history"] else float("nan")
        )
        last_val_loss = float(meta["best_val_loss"])
        last_param_count = int(bb_count_parameters(model))
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    return {
        "per_seed_r2": per_seed_r2,
        "per_seed_rmse": per_seed_rmse,
        "per_seed_per_roast": per_seed_per_roast,
        "per_seed_history": per_seed_history,
        "per_seed_best_epoch": per_seed_best_epoch,
        "per_seed_epochs": per_seed_epochs,
        "train_loss_final": last_train_loss,
        "val_loss_final": last_val_loss,
        "param_count": last_param_count,
    }


def _run_greybox_trial(
    config: dict[str, Any],
    seeds: Sequence[int],
    cohort_ids: CohortIds,
    train_sequences: Sequence[FullRoastSequence],
    val_sequences: Sequence[FullRoastSequence],
    device: str,
) -> dict[str, Any]:
    cohort_ids.assert_no_test_leakage([s.roast_id for s in train_sequences], "greybox train")
    cohort_ids.assert_no_test_leakage([s.roast_id for s in val_sequences], "greybox val")
    per_seed_r2: dict[str, float] = {}
    per_seed_rmse: dict[str, float] = {}
    per_seed_per_roast: dict[str, dict[str, float]] = {}
    per_seed_history: dict[str, list[dict[str, float]]] = {}
    per_seed_best_epoch: dict[str, int] = {}
    per_seed_epochs: dict[str, int] = {}
    last_train_loss = float("nan")
    last_val_loss = float("nan")
    last_param_count = 0
    for seed in seeds:
        set_torch_seed(int(seed))
        model = LearnedHeFullStateModel(hidden_widths=tuple(config["hidden_widths"]))
        meta = train_fullstate_model(
            model,
            train_sequences,
            val_sequences,
            epochs=int(config["epochs"]),
            lr=float(config["lr"]),
            weight_decay=float(config["weight_decay"]),
            batch_size=int(config["batch_size"]),
            device=device,
            tgo_weight=float(config["tgo_weight"]),
            detach_state_steps=bool(config["detach_state_steps"]),
            warmup_steps=int(config["warmup_steps"]),
            patience=int(config["patience"]),
        )
        _, val_rollout, val_per_roast = pooled_rollout_metrics_fullstate(model, val_sequences)
        per_seed_r2[str(seed)] = float(val_rollout.r2)
        per_seed_rmse[str(seed)] = float(val_rollout.rmse)
        per_seed_per_roast[str(seed)] = per_roast_r2(val_per_roast)
        per_seed_history[str(seed)] = meta["history"]
        per_seed_best_epoch[str(seed)] = int(meta.get("best_epoch", 0))
        per_seed_epochs[str(seed)] = int(meta["epochs_run"])
        last_train_loss = (
            float(meta["history"][-1]["train_loss"]) if meta["history"] else float("nan")
        )
        last_val_loss = float(meta["best_val_loss"])
        last_param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    return {
        "per_seed_r2": per_seed_r2,
        "per_seed_rmse": per_seed_rmse,
        "per_seed_per_roast": per_seed_per_roast,
        "per_seed_history": per_seed_history,
        "per_seed_best_epoch": per_seed_best_epoch,
        "per_seed_epochs": per_seed_epochs,
        "train_loss_final": last_train_loss,
        "val_loss_final": last_val_loss,
        "param_count": last_param_count,
    }


def _run_multi_closure_trial(
    config: dict[str, Any],
    seeds: Sequence[int],
    cohort_ids: CohortIds,
    train_sequences: Sequence[FullRoastSequence],
    val_sequences: Sequence[FullRoastSequence],
    device: str,
) -> dict[str, Any]:
    """Position 4: multi-closure PI model with learned h_e + moisture + reaction.

    Uses fixed literature priors for the initial unmeasured states and a frozen
    init_net (matching the priors-corrected PI setup).
    """
    cohort_ids.assert_no_test_leakage([s.roast_id for s in train_sequences], "multi_closure train")
    cohort_ids.assert_no_test_leakage([s.roast_id for s in val_sequences], "multi_closure val")
    per_seed_r2: dict[str, float] = {}
    per_seed_rmse: dict[str, float] = {}
    per_seed_per_roast: dict[str, dict[str, float]] = {}
    per_seed_history: dict[str, list[dict[str, float]]] = {}
    per_seed_best_epoch: dict[str, int] = {}
    per_seed_epochs: dict[str, int] = {}
    last_train_loss = float("nan")
    last_val_loss = float("nan")
    last_param_count = 0
    for seed in seeds:
        set_torch_seed(int(seed))
        model_kwargs: dict[str, Any] = {
            "he_hidden_widths": tuple(config["he_hidden_widths"]),
            "moisture_hidden_widths": tuple(config["moisture_hidden_widths"]),
            "reaction_hidden_widths": tuple(config["reaction_hidden_widths"]),
        }
        if config.get("fixed_initial_bean_temp_c") is not None:
            model_kwargs["fixed_initial_bean_temp_c"] = float(config["fixed_initial_bean_temp_c"])
        if config.get("fixed_initial_moisture_ratio") is not None:
            model_kwargs["fixed_initial_moisture_ratio"] = float(config["fixed_initial_moisture_ratio"])
        model = MultiClosureFullStateModel(**model_kwargs)
        if config.get("freeze_init_net", False):
            for p in model.init_net.parameters():
                p.requires_grad = False
        meta = train_fullstate_model(
            model,
            train_sequences,
            val_sequences,
            epochs=int(config["epochs"]),
            lr=float(config["lr"]),
            weight_decay=float(config["weight_decay"]),
            batch_size=int(config["batch_size"]),
            device=device,
            tgo_weight=float(config["tgo_weight"]),
            detach_state_steps=bool(config["detach_state_steps"]),
            warmup_steps=int(config["warmup_steps"]),
            patience=int(config["patience"]),
        )
        _, val_rollout, val_per_roast = pooled_rollout_metrics_fullstate(model, val_sequences)
        per_seed_r2[str(seed)] = float(val_rollout.r2)
        per_seed_rmse[str(seed)] = float(val_rollout.rmse)
        per_seed_per_roast[str(seed)] = per_roast_r2(val_per_roast)
        per_seed_history[str(seed)] = meta["history"]
        per_seed_best_epoch[str(seed)] = int(meta.get("best_epoch", 0))
        per_seed_epochs[str(seed)] = int(meta["epochs_run"])
        last_train_loss = (
            float(meta["history"][-1]["train_loss"]) if meta["history"] else float("nan")
        )
        last_val_loss = float(meta["best_val_loss"])
        last_param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    return {
        "per_seed_r2": per_seed_r2,
        "per_seed_rmse": per_seed_rmse,
        "per_seed_per_roast": per_seed_per_roast,
        "per_seed_history": per_seed_history,
        "per_seed_best_epoch": per_seed_best_epoch,
        "per_seed_epochs": per_seed_epochs,
        "train_loss_final": last_train_loss,
        "val_loss_final": last_val_loss,
        "param_count": last_param_count,
    }


def _run_whitebox_trial(
    config: dict[str, Any],
    seeds: Sequence[int],
    cohort_ids: CohortIds,
    train_sequences: Sequence[FullRoastSequence],
    val_sequences: Sequence[FullRoastSequence],
    device: str,
) -> dict[str, Any]:
    """White-box (constant-He) trial. Shares the grey-box trainer, but the
    model is the ConstantHeFullStateModel which has no closure MLP — only
    scalar physical parameters and the shared init network."""
    cohort_ids.assert_no_test_leakage([s.roast_id for s in train_sequences], "whitebox train")
    cohort_ids.assert_no_test_leakage([s.roast_id for s in val_sequences], "whitebox val")
    per_seed_r2: dict[str, float] = {}
    per_seed_rmse: dict[str, float] = {}
    per_seed_per_roast: dict[str, dict[str, float]] = {}
    per_seed_history: dict[str, list[dict[str, float]]] = {}
    per_seed_best_epoch: dict[str, int] = {}
    per_seed_epochs: dict[str, int] = {}
    last_train_loss = float("nan")
    last_val_loss = float("nan")
    last_param_count = 0
    for seed in seeds:
        set_torch_seed(int(seed))
        model = ConstantHeFullStateModel()
        meta = train_fullstate_model(
            model,
            train_sequences,
            val_sequences,
            epochs=int(config["epochs"]),
            lr=float(config["lr"]),
            weight_decay=float(config["weight_decay"]),
            batch_size=int(config["batch_size"]),
            device=device,
            tgo_weight=float(config["tgo_weight"]),
            detach_state_steps=bool(config["detach_state_steps"]),
            warmup_steps=int(config["warmup_steps"]),
            patience=int(config["patience"]),
        )
        _, val_rollout, val_per_roast = pooled_rollout_metrics_fullstate(model, val_sequences)
        per_seed_r2[str(seed)] = float(val_rollout.r2)
        per_seed_rmse[str(seed)] = float(val_rollout.rmse)
        per_seed_per_roast[str(seed)] = per_roast_r2(val_per_roast)
        per_seed_history[str(seed)] = meta["history"]
        per_seed_best_epoch[str(seed)] = int(meta.get("best_epoch", 0))
        per_seed_epochs[str(seed)] = int(meta["epochs_run"])
        last_train_loss = (
            float(meta["history"][-1]["train_loss"]) if meta["history"] else float("nan")
        )
        last_val_loss = float(meta["best_val_loss"])
        last_param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    return {
        "per_seed_r2": per_seed_r2,
        "per_seed_rmse": per_seed_rmse,
        "per_seed_per_roast": per_seed_per_roast,
        "per_seed_history": per_seed_history,
        "per_seed_best_epoch": per_seed_best_epoch,
        "per_seed_epochs": per_seed_epochs,
        "train_loss_final": last_train_loss,
        "val_loss_final": last_val_loss,
        "param_count": last_param_count,
    }


def _run_residual_ff_trial(
    config: dict[str, Any],
    seeds: Sequence[int],
    cohort_ids: CohortIds,
    train_sequences: Sequence[FullRoastSequence],
    val_sequences: Sequence[FullRoastSequence],
    base_model: LearnedHeFullStateModel,
    device: str,
) -> dict[str, Any]:
    """Bounded feedforward residual trial. Mirrors _run_residual_trial but uses
    a stateless FF MLP architecture instead of the LSTM."""
    cohort_ids.assert_no_test_leakage([s.roast_id for s in train_sequences], "residual_ff train")
    cohort_ids.assert_no_test_leakage([s.roast_id for s in val_sequences], "residual_ff val")
    per_seed_r2: dict[str, float] = {}
    per_seed_rmse: dict[str, float] = {}
    per_seed_per_roast: dict[str, dict[str, float]] = {}
    per_seed_history: dict[str, list[dict[str, float]]] = {}
    per_seed_best_epoch: dict[str, int] = {}
    per_seed_epochs: dict[str, int] = {}
    last_train_loss = float("nan")
    last_val_loss = float("nan")
    last_param_count = 0
    for seed in seeds:
        set_torch_seed(int(seed))
        residual = ResidualFeedForwardModel(
            hidden_widths=tuple(config["hidden_widths"]),
            max_delta=float(config["max_delta"]),
        )
        meta = train_residual_model(
            residual,
            base_model,
            train_sequences,
            val_sequences,
            epochs=int(config["epochs"]),
            lr=float(config["lr"]),
            weight_decay=float(config["weight_decay"]),
            batch_size=int(config["batch_size"]),
            device=device,
            warmup_steps=int(config["warmup_steps"]),
            residual_weight=float(config["residual_weight"]),
            patience=int(config["patience"]),
        )
        _, val_rollout, val_per_roast = pooled_rollout_metrics_residual(
            residual, base_model, val_sequences
        )
        per_seed_r2[str(seed)] = float(val_rollout.r2)
        per_seed_rmse[str(seed)] = float(val_rollout.rmse)
        per_seed_per_roast[str(seed)] = per_roast_r2(val_per_roast)
        per_seed_history[str(seed)] = meta["history"]
        per_seed_best_epoch[str(seed)] = int(meta.get("best_epoch", 0))
        per_seed_epochs[str(seed)] = int(meta["epochs_run"])
        last_train_loss = (
            float(meta["history"][-1]["train_loss"]) if meta["history"] else float("nan")
        )
        last_val_loss = float(meta["best_val_loss"])
        last_param_count = sum(p.numel() for p in residual.parameters() if p.requires_grad)
        del residual
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    return {
        "per_seed_r2": per_seed_r2,
        "per_seed_rmse": per_seed_rmse,
        "per_seed_per_roast": per_seed_per_roast,
        "per_seed_history": per_seed_history,
        "per_seed_best_epoch": per_seed_best_epoch,
        "per_seed_epochs": per_seed_epochs,
        "train_loss_final": last_train_loss,
        "val_loss_final": last_val_loss,
        "param_count": last_param_count,
    }


def _run_residual_trial(
    config: dict[str, Any],
    seeds: Sequence[int],
    cohort_ids: CohortIds,
    train_sequences: Sequence[FullRoastSequence],
    val_sequences: Sequence[FullRoastSequence],
    base_model: LearnedHeFullStateModel,
    device: str,
) -> dict[str, Any]:
    cohort_ids.assert_no_test_leakage([s.roast_id for s in train_sequences], "residual train")
    cohort_ids.assert_no_test_leakage([s.roast_id for s in val_sequences], "residual val")
    per_seed_r2: dict[str, float] = {}
    per_seed_rmse: dict[str, float] = {}
    per_seed_per_roast: dict[str, dict[str, float]] = {}
    per_seed_history: dict[str, list[dict[str, float]]] = {}
    per_seed_best_epoch: dict[str, int] = {}
    per_seed_epochs: dict[str, int] = {}
    last_train_loss = float("nan")
    last_val_loss = float("nan")
    last_param_count = 0
    for seed in seeds:
        set_torch_seed(int(seed))
        residual = ResidualLSTMModel(
            hidden_size=int(config["hidden_size"]),
            num_layers=1,
            dropout=0.0,
            max_delta=float(config["max_delta"]),
        )
        meta = train_residual_model(
            residual,
            base_model,
            train_sequences,
            val_sequences,
            epochs=int(config["epochs"]),
            lr=float(config["lr"]),
            weight_decay=float(config["weight_decay"]),
            batch_size=int(config["batch_size"]),
            device=device,
            warmup_steps=int(config["warmup_steps"]),
            residual_weight=float(config["residual_weight"]),
            patience=int(config["patience"]),
        )
        _, val_rollout, val_per_roast = pooled_rollout_metrics_residual(
            residual, base_model, val_sequences
        )
        per_seed_r2[str(seed)] = float(val_rollout.r2)
        per_seed_rmse[str(seed)] = float(val_rollout.rmse)
        per_seed_per_roast[str(seed)] = per_roast_r2(val_per_roast)
        per_seed_history[str(seed)] = meta["history"]
        per_seed_best_epoch[str(seed)] = int(meta.get("best_epoch", 0))
        per_seed_epochs[str(seed)] = int(meta["epochs_run"])
        last_train_loss = (
            float(meta["history"][-1]["train_loss"]) if meta["history"] else float("nan")
        )
        last_val_loss = float(meta["best_val_loss"])
        last_param_count = sum(p.numel() for p in residual.parameters() if p.requires_grad)
        del residual
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    return {
        "per_seed_r2": per_seed_r2,
        "per_seed_rmse": per_seed_rmse,
        "per_seed_per_roast": per_seed_per_roast,
        "per_seed_history": per_seed_history,
        "per_seed_best_epoch": per_seed_best_epoch,
        "per_seed_epochs": per_seed_epochs,
        "train_loss_final": last_train_loss,
        "val_loss_final": last_val_loss,
        "param_count": last_param_count,
    }


# ---------------------------------------------------------------------------
# Sweep loop
# ---------------------------------------------------------------------------


def run_sweep(
    model_class: str,
    n_trials: int,
    search_seed: int,
    per_trial_seeds: Sequence[int],
    trial_fn: Callable[[dict[str, Any], Sequence[int]], dict[str, Any]],
    sample_fn: Callable[[np.random.Generator], dict[str, Any]],
    outdir: Path,
) -> TrialRecord:
    rng = np.random.default_rng(search_seed)
    class_dir = outdir / model_class
    class_dir.mkdir(parents=True, exist_ok=True)
    trials_jsonl = class_dir / "all_trials.jsonl"
    trials_jsonl.write_text("", encoding="utf-8")

    records: list[TrialRecord] = []
    for trial_idx in range(n_trials):
        config = sample_fn(rng)
        start = time.perf_counter()
        try:
            result = trial_fn(config, per_trial_seeds)
        except Exception as exc:  # log and skip
            elapsed = time.perf_counter() - start
            failure_record = {
                "trial_idx": trial_idx,
                "config": config,
                "error": repr(exc),
                "wall_time_sec": elapsed,
            }
            with trials_jsonl.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(failure_record) + "\n")
            print(f"[{model_class}] trial {trial_idx} failed: {exc}", flush=True)
            continue
        elapsed = time.perf_counter() - start

        r2_values = [v for v in result["per_seed_r2"].values() if math.isfinite(v)]
        rmse_values = [v for v in result["per_seed_rmse"].values() if math.isfinite(v)]
        mean_r2 = float(np.mean(r2_values)) if r2_values else float("-inf")
        mean_rmse = float(np.mean(rmse_values)) if rmse_values else float("nan")
        record = TrialRecord(
            trial_idx=trial_idx,
            config=config,
            per_seed_val_rollout_r2=result["per_seed_r2"],
            mean_val_rollout_r2=mean_r2,
            per_seed_val_rollout_rmse=result["per_seed_rmse"],
            mean_val_rollout_rmse=mean_rmse,
            per_seed_per_roast_val_r2=result.get("per_seed_per_roast", {}),
            per_seed_history=result.get("per_seed_history", {}),
            per_seed_best_epoch=result.get("per_seed_best_epoch", {}),
            train_loss_final=float(result["train_loss_final"]),
            val_loss_final=float(result["val_loss_final"]),
            param_count=int(result["param_count"]),
            epochs_run_per_seed=result["per_seed_epochs"],
            wall_time_sec=float(elapsed),
        )
        records.append(record)
        with trials_jsonl.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(dataclasses.asdict(record)) + "\n")
        print(
            f"[{model_class}] trial {trial_idx:>3}: "
            f"mean_val_R2={mean_r2:+.4f} per_seed={result['per_seed_r2']} "
            f"params={record.param_count} elapsed={elapsed:.1f}s",
            flush=True,
        )

    if not records:
        raise RuntimeError(f"No successful trials for model class {model_class}.")

    finite_records = [r for r in records if math.isfinite(r.mean_val_rollout_r2)]
    if not finite_records:
        raise RuntimeError(
            f"All trials produced non-finite val R^2 for model class {model_class}."
        )
    best = max(finite_records, key=lambda r: r.mean_val_rollout_r2)

    # Best-config payload, sorted summary.
    (class_dir / "best_config.json").write_text(
        json.dumps(dataclasses.asdict(best), indent=2), encoding="utf-8"
    )
    ranked = sorted(records, key=lambda r: r.mean_val_rollout_r2, reverse=True)
    md_lines = [
        f"# Sweep summary — {model_class}",
        "",
        f"Best trial: `#{best.trial_idx}`  mean val rollout R^2 = `{best.mean_val_rollout_r2:.4f}`",
        "",
        "| Rank | Trial | Mean val R^2 | Per-seed val R^2 | Mean val RMSE | Params | Epochs (per seed) | Wall time (s) |",
        "|---:|---:|---:|---|---:|---:|---|---:|",
    ]
    for rank, rec in enumerate(ranked, start=1):
        md_lines.append(
            f"| {rank} | {rec.trial_idx} | {rec.mean_val_rollout_r2:.4f} | "
            f"{rec.per_seed_val_rollout_r2} | {rec.mean_val_rollout_rmse:.4f} | "
            f"{rec.param_count} | {rec.epochs_run_per_seed} | {rec.wall_time_sec:.1f} |"
        )
    (class_dir / "sweep_summary.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return best


# ---------------------------------------------------------------------------
# Final test-set retraining + evaluation
# ---------------------------------------------------------------------------


def _promote_to_final_budget(config: dict[str, Any], model_class: str) -> dict[str, Any]:
    """Return a copy of config with sweep-time epoch/patience replaced by the
    manuscript-canonical training budget for ``model_class``."""
    promoted = dict(config)
    promoted["epochs"] = MANUSCRIPT_FINAL_EPOCHS[model_class]
    promoted["patience"] = MANUSCRIPT_FINAL_PATIENCE[model_class]
    return promoted


def _greybox_checkpoint_path(outdir: Path, seed: int) -> Path:
    return outdir / "checkpoints" / f"greybox_seed{seed}.pt"


def _save_greybox_checkpoint(
    path: Path,
    model: LearnedHeFullStateModel,
    config: dict[str, Any],
    seed: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "hidden_widths": list(model.hidden_widths),
            "config": config,
            "seed": int(seed),
        },
        path,
    )


def _load_greybox_checkpoint(
    path: Path,
    device: str,
) -> LearnedHeFullStateModel:
    payload = torch.load(path, map_location=device)
    model = LearnedHeFullStateModel(hidden_widths=tuple(payload["hidden_widths"]))
    model.load_state_dict(payload["state_dict"])
    model.to(torch.device(device))
    model.eval()
    return model


def retrain_and_evaluate_blackbox(
    config: dict[str, Any],
    seed: int,
    cohort_ids: CohortIds,
    grouped: dict[str, pd.DataFrame],
    feature_columns: Sequence[str],
    device: str,
) -> dict[str, Any]:
    cohort_ids.assert_no_test_leakage(cohort_ids.train, "blackbox final train")
    cohort_ids.assert_no_test_leakage(cohort_ids.val, "blackbox final val")
    model, scalers, meta = train_blackbox(
        grouped,
        cohort_ids.train,
        cohort_ids.val,
        feature_columns=feature_columns,
        hidden_widths=tuple(config["hidden_widths"]),
        dropout=float(config["dropout"]),
        lr=float(config["lr"]),
        weight_decay=float(config["weight_decay"]),
        batch_size=config["batch_size"],
        epochs=int(config["epochs"]),
        patience=int(config["patience"]),
        device=device,
        seed=int(seed),
    )
    one_step, rollout, per_roast = pooled_rollout_metrics_blackbox(
        model, scalers, grouped, cohort_ids.test, feature_columns, device
    )
    ci_lo, ci_hi = roast_bootstrap_rollout_r2(per_roast, n_boot=1000, seed=seed)
    payload = {
        "one_step_metrics": {
            "r2": float(one_step.r2),
            "rmse": float(one_step.rmse),
            "mae": float(one_step.mae),
            "n": int(one_step.n),
        },
        "rollout_metrics": {
            "r2": float(rollout.r2),
            "rmse": float(rollout.rmse),
            "mae": float(rollout.mae),
            "n": int(rollout.n),
        },
        "rollout_r2_ci95": [ci_lo, ci_hi],
        "param_count": int(bb_count_parameters(model)),
        "per_roast": per_roast,
        "per_roast_r2": per_roast_r2(per_roast),
        "training": {"best_val_loss": meta["best_val_loss"], "epochs_run": meta["epochs_run"], "best_epoch": int(meta.get("best_epoch", 0)), "history": meta.get("history", [])},
        "seed": int(seed),
        "config": config,
    }
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return payload


def retrain_and_evaluate_fullstate(
    *,
    model_factory: Callable[[], LearnedHeFullStateModel | ConstantHeFullStateModel],
    config: dict[str, Any],
    seed: int,
    cohort_ids: CohortIds,
    train_sequences: Sequence[FullRoastSequence],
    val_sequences: Sequence[FullRoastSequence],
    test_sequences: Sequence[FullRoastSequence],
    device: str,
    save_checkpoint_path: Path | None = None,
) -> tuple[dict[str, Any], LearnedHeFullStateModel | ConstantHeFullStateModel]:
    cohort_ids.assert_no_test_leakage([s.roast_id for s in train_sequences], "fullstate final train")
    cohort_ids.assert_no_test_leakage([s.roast_id for s in val_sequences], "fullstate final val")
    set_torch_seed(int(seed))
    model = model_factory()
    meta = train_fullstate_model(
        model,
        train_sequences,
        val_sequences,
        epochs=int(config["epochs"]),
        lr=float(config["lr"]),
        weight_decay=float(config["weight_decay"]),
        batch_size=int(config["batch_size"]),
        device=device,
        tgo_weight=float(config["tgo_weight"]),
        detach_state_steps=bool(config["detach_state_steps"]),
        warmup_steps=int(config["warmup_steps"]),
        patience=int(config["patience"]),
    )
    one_step, rollout, per_roast = pooled_rollout_metrics_fullstate(model, test_sequences)
    ci_lo, ci_hi = roast_bootstrap_rollout_r2(per_roast, n_boot=1000, seed=seed)
    if save_checkpoint_path is not None and isinstance(model, LearnedHeFullStateModel):
        _save_greybox_checkpoint(save_checkpoint_path, model, config, seed)
    payload = {
        "one_step_metrics": {
            "r2": float(one_step.r2),
            "rmse": float(one_step.rmse),
            "mae": float(one_step.mae),
            "n": int(one_step.n),
        },
        "rollout_metrics": {
            "r2": float(rollout.r2),
            "rmse": float(rollout.rmse),
            "mae": float(rollout.mae),
            "n": int(rollout.n),
        },
        "rollout_r2_ci95": [ci_lo, ci_hi],
        "param_count": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "per_roast": per_roast,
        "per_roast_r2": per_roast_r2(per_roast),
        "training": {"best_val_loss": meta["best_val_loss"], "epochs_run": meta["epochs_run"], "best_epoch": int(meta.get("best_epoch", 0)), "history": meta.get("history", [])},
        "seed": int(seed),
        "config": config,
    }
    return payload, model


def retrain_and_evaluate_residual(
    config: dict[str, Any],
    seed: int,
    cohort_ids: CohortIds,
    train_sequences: Sequence[FullRoastSequence],
    val_sequences: Sequence[FullRoastSequence],
    test_sequences: Sequence[FullRoastSequence],
    base_model: LearnedHeFullStateModel,
    device: str,
) -> dict[str, Any]:
    cohort_ids.assert_no_test_leakage([s.roast_id for s in train_sequences], "residual final train")
    cohort_ids.assert_no_test_leakage([s.roast_id for s in val_sequences], "residual final val")
    set_torch_seed(int(seed))
    residual = ResidualLSTMModel(
        hidden_size=int(config["hidden_size"]),
        num_layers=1,
        dropout=0.0,
        max_delta=float(config["max_delta"]),
    )
    meta = train_residual_model(
        residual,
        base_model,
        train_sequences,
        val_sequences,
        epochs=int(config["epochs"]),
        lr=float(config["lr"]),
        weight_decay=float(config["weight_decay"]),
        batch_size=int(config["batch_size"]),
        device=device,
        warmup_steps=int(config["warmup_steps"]),
        residual_weight=float(config["residual_weight"]),
        patience=int(config["patience"]),
    )
    one_step, rollout, per_roast = pooled_rollout_metrics_residual(
        residual, base_model, test_sequences
    )
    ci_lo, ci_hi = roast_bootstrap_rollout_r2(per_roast, n_boot=1000, seed=seed)
    payload = {
        "one_step_metrics": {
            "r2": float(one_step.r2),
            "rmse": float(one_step.rmse),
            "mae": float(one_step.mae),
            "n": int(one_step.n),
        },
        "rollout_metrics": {
            "r2": float(rollout.r2),
            "rmse": float(rollout.rmse),
            "mae": float(rollout.mae),
            "n": int(rollout.n),
        },
        "rollout_r2_ci95": [ci_lo, ci_hi],
        "param_count": int(sum(p.numel() for p in residual.parameters() if p.requires_grad)),
        "per_roast": per_roast,
        "per_roast_r2": per_roast_r2(per_roast),
        "training": {"best_val_loss": meta["best_val_loss"], "epochs_run": meta["epochs_run"], "best_epoch": int(meta.get("best_epoch", 0)), "history": meta.get("history", [])},
        "seed": int(seed),
        "config": config,
    }
    del residual
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return payload


# ---------------------------------------------------------------------------
# Environment capture
# ---------------------------------------------------------------------------


def capture_environment(outdir: Path) -> dict[str, Any]:
    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        git_sha = "unknown"
    try:
        git_status_lines = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True
        ).strip().splitlines()
        git_dirty = bool(git_status_lines)
    except Exception:
        git_dirty = True

    env = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
        "git_sha": git_sha,
        "git_dirty": git_dirty,
    }
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "environment.json").write_text(json.dumps(env, indent=2), encoding="utf-8")
    return env


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _comma_int_list(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def _comma_str_list(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "data" / "processed" / "roast_timeseries_p2_only.csv",
        help="Timeseries CSV. Default is the pre-filtered p2-only cohort that "
        "reproduces the manuscript's exact 221-roast split.",
    )
    parser.add_argument("--metadata-path", type=Path, default=ROOT / "data" / "metadata" / "normalized_files.csv")
    parser.add_argument("--include-p-codes", type=str, default=",".join(DEFAULT_P_CODES))
    parser.add_argument("--include-source-buckets", type=str, default=",".join(DEFAULT_SOURCE_BUCKETS))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n-trials", type=int, default=40)
    parser.add_argument("--search-seed", type=int, default=0)
    parser.add_argument(
        "--per-trial-seeds",
        type=str,
        default="11",
        help="Seeds used to evaluate each trial during the sweep. Single seed "
        "by default for overnight budget; final retrain still uses --final-seeds.",
    )
    parser.add_argument("--final-seeds", type=str, default="11,23,37")
    parser.add_argument(
        "--phase",
        type=str,
        default="all",
        choices=[
            "all",
            "sweep_blackbox",
            "sweep_greybox",
            "sweep_greybox_bs8",
            "sweep_residual",
            "sweep_residual_ff",
            "sweep_residual_ff_unbounded",
            "sweep_whitebox",
            "sweep_multi_closure",
            "final_eval",
            "final_eval_whitebox",
        ],
        help="all = run every phase in sequence (excludes sweep_whitebox / "
        "final_eval_whitebox, which are run on demand after the main sweep).",
    )
    parser.add_argument(
        "--dry-run-trials",
        type=int,
        default=0,
        help="If > 0, override --n-trials for timing measurement only.",
    )
    parser.add_argument("--outdir", type=Path, default=ROOT / "reports" / "manuscript_hpo")
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Dataset not found: {args.input}")
    # Metadata is only required when the input is not the pre-filtered p2_only
    # timeseries; the prefiltered file is already bucket-scoped and the cohort
    # loader skips the source-bucket filter in that case.
    is_prefiltered_input = args.input.name.endswith("_p2_only.csv")
    if not is_prefiltered_input and not args.metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {args.metadata_path}")

    outdir: Path = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    env = capture_environment(outdir)
    print(f"Environment captured: torch={env['torch']} cuda={env['cuda_available']} device={args.device}", flush=True)

    include_p_codes = _comma_str_list(args.include_p_codes)
    include_source_buckets = _comma_str_list(args.include_source_buckets)
    per_trial_seeds = _comma_int_list(args.per_trial_seeds)
    final_seeds = _comma_int_list(args.final_seeds)
    n_trials = args.dry_run_trials if args.dry_run_trials > 0 else args.n_trials

    print("Loading cohort...", flush=True)
    sequences = load_manuscript_cohort(
        args.input,
        args.metadata_path,
        include_p_codes,
        include_source_buckets,
    )
    train_sequences, val_sequences, test_sequences = split_cohort(sequences)
    cohort_ids = CohortIds(
        train_ids=[s.roast_id for s in train_sequences],
        val_ids=[s.roast_id for s in val_sequences],
        test_ids=[s.roast_id for s in test_sequences],
    )
    print(
        f"Cohort: total={len(sequences)} train={len(train_sequences)} val={len(val_sequences)} test={len(test_sequences)}",
        flush=True,
    )

    grouped = load_grouped_dataframe(args.input, [s.roast_id for s in sequences], CORE_FEATURES)

    # Persist search-state for reproducibility.
    (outdir / "search_state.json").write_text(
        json.dumps(
            {
                "search_seed": args.search_seed,
                "n_trials": n_trials,
                "per_trial_seeds": per_trial_seeds,
                "final_seeds": final_seeds,
                "include_p_codes": include_p_codes,
                "include_source_buckets": include_source_buckets,
                "split_strategy": "md5",
                "warmup_steps": WARMUP_STEPS,
                "residual_base_seed": RESIDUAL_BASE_SEED,
                "search_space": SEARCH_SPACE_SPEC,
                "cohort_sizes": {
                    "train": len(train_sequences),
                    "val": len(val_sequences),
                    "test": len(test_sequences),
                    "total": len(sequences),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # ------------------------------------------------------------- Sweep phase
    best_configs: dict[str, dict[str, Any]] = {}

    def _load_best_config(model_class: str) -> dict[str, Any]:
        path = outdir / model_class / "best_config.json"
        if not path.exists():
            raise FileNotFoundError(
                f"Best config for {model_class} not found at {path}. Run --phase sweep_{model_class} first."
            )
        return json.loads(path.read_text(encoding="utf-8"))

    run_blackbox_sweep = args.phase in ("all", "sweep_blackbox")
    run_greybox_sweep = args.phase in ("all", "sweep_greybox")
    run_residual_sweep = args.phase in ("all", "sweep_residual")
    run_whitebox_sweep = args.phase == "sweep_whitebox"
    run_greybox_bs8_sweep = args.phase == "sweep_greybox_bs8"
    run_multi_closure_sweep = args.phase == "sweep_multi_closure"
    run_residual_ff_sweep = args.phase == "sweep_residual_ff"
    run_residual_ff_unbounded_sweep = args.phase == "sweep_residual_ff_unbounded"
    run_final = args.phase in ("all", "final_eval")
    run_final_whitebox_only = args.phase == "final_eval_whitebox"

    if run_blackbox_sweep:
        print("\n=== Black-box sweep ===", flush=True)
        best = run_sweep(
            model_class="blackbox",
            n_trials=n_trials,
            search_seed=args.search_seed,
            per_trial_seeds=per_trial_seeds,
            trial_fn=lambda cfg, seeds: _run_blackbox_trial(
                cfg, seeds, cohort_ids, grouped, CORE_FEATURES, args.device
            ),
            sample_fn=sample_blackbox_config,
            outdir=outdir,
        )
        best_configs["blackbox"] = dataclasses.asdict(best)
    if run_greybox_sweep:
        print("\n=== Grey-box sweep ===", flush=True)
        best = run_sweep(
            model_class="greybox",
            n_trials=n_trials,
            search_seed=args.search_seed + 1,  # decorrelate the per-class search RNGs
            per_trial_seeds=per_trial_seeds,
            trial_fn=lambda cfg, seeds: _run_greybox_trial(
                cfg, seeds, cohort_ids, train_sequences, val_sequences, args.device
            ),
            sample_fn=sample_greybox_config,
            outdir=outdir,
        )
        best_configs["greybox"] = dataclasses.asdict(best)

    # Residual sweep requires a frozen seed-11 grey-box base.
    # We materialize that base eagerly when the residual sweep is requested.
    if run_residual_sweep:
        if "greybox" not in best_configs:
            best_configs["greybox"] = _load_best_config("greybox")
        gb_cfg = best_configs["greybox"]["config"]
        # The base used by the residual sweep should match the base used at
        # final test time — i.e. retrained at the manuscript-canonical budget,
        # not the sweep-time cap.
        gb_cfg_final = _promote_to_final_budget(gb_cfg, "greybox")
        base_ckpt_path = _greybox_checkpoint_path(outdir, RESIDUAL_BASE_SEED)
        if not base_ckpt_path.exists():
            print(
                f"\n=== Training residual-sweep base (grey-box seed {RESIDUAL_BASE_SEED}, "
                f"epochs={gb_cfg_final['epochs']}) ===",
                flush=True,
            )
            _, base_model = retrain_and_evaluate_fullstate(
                model_factory=lambda: LearnedHeFullStateModel(
                    hidden_widths=tuple(gb_cfg_final["hidden_widths"])
                ),
                config=gb_cfg_final,
                seed=RESIDUAL_BASE_SEED,
                cohort_ids=cohort_ids,
                train_sequences=train_sequences,
                val_sequences=val_sequences,
                test_sequences=test_sequences,
                device=args.device,
                save_checkpoint_path=base_ckpt_path,
            )
            del base_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        base_model = _load_greybox_checkpoint(base_ckpt_path, args.device)
        print("\n=== Residual sweep ===", flush=True)
        best = run_sweep(
            model_class="residual",
            n_trials=n_trials,
            search_seed=args.search_seed + 2,
            per_trial_seeds=per_trial_seeds,
            trial_fn=lambda cfg, seeds: _run_residual_trial(
                cfg, seeds, cohort_ids, train_sequences, val_sequences, base_model, args.device
            ),
            sample_fn=sample_residual_config,
            outdir=outdir,
        )
        best_configs["residual"] = dataclasses.asdict(best)
        del base_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    # ----------------------------------------------- Bounded FF residual sweep
    if run_residual_ff_sweep:
        base_ckpt_path = _greybox_checkpoint_path(outdir, RESIDUAL_BASE_SEED)
        if not base_ckpt_path.exists():
            raise FileNotFoundError(
                f"Seed-{RESIDUAL_BASE_SEED} PI base checkpoint missing at {base_ckpt_path}. "
                "Run the main HPO sweep's final_eval phase first to materialize it."
            )
        base_model = _load_greybox_checkpoint(base_ckpt_path, args.device)
        print("\n=== Bounded FF residual sweep (position 4) ===", flush=True)
        best = run_sweep(
            model_class="residual_ff",
            n_trials=n_trials,
            search_seed=args.search_seed + 23,
            per_trial_seeds=per_trial_seeds,
            trial_fn=lambda cfg, seeds: _run_residual_ff_trial(
                cfg, seeds, cohort_ids, train_sequences, val_sequences, base_model, args.device
            ),
            sample_fn=sample_residual_ff_config,
            outdir=outdir,
        )
        best_configs["residual_ff"] = dataclasses.asdict(best)
        del base_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    # --------------------------------------- Unbounded FF residual sweep
    if run_residual_ff_unbounded_sweep:
        base_ckpt_path = _greybox_checkpoint_path(outdir, RESIDUAL_BASE_SEED)
        if not base_ckpt_path.exists():
            raise FileNotFoundError(
                f"Seed-{RESIDUAL_BASE_SEED} PI base checkpoint missing at {base_ckpt_path}. "
                "Run the main HPO sweep's final_eval phase first to materialize it."
            )
        base_model = _load_greybox_checkpoint(base_ckpt_path, args.device)
        print("\n=== Unbounded FF residual sweep (position 5) ===", flush=True)
        best = run_sweep(
            model_class="residual_ff_unbounded",
            n_trials=n_trials,
            search_seed=args.search_seed + 29,
            per_trial_seeds=per_trial_seeds,
            trial_fn=lambda cfg, seeds: _run_residual_ff_trial(
                cfg, seeds, cohort_ids, train_sequences, val_sequences, base_model, args.device
            ),
            sample_fn=sample_residual_ff_unbounded_config,
            outdir=outdir,
        )
        best_configs["residual_ff_unbounded"] = dataclasses.asdict(best)
        del base_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    # ------------------------------------------------------------- bs=8 PI sweep
    if run_greybox_bs8_sweep:
        print("\n=== Grey-box bs=8 focused sweep ===", flush=True)
        best = run_sweep(
            model_class="greybox_bs8",
            n_trials=n_trials,
            search_seed=args.search_seed + 11,
            per_trial_seeds=per_trial_seeds,
            trial_fn=lambda cfg, seeds: _run_greybox_trial(
                cfg, seeds, cohort_ids, train_sequences, val_sequences, args.device
            ),
            sample_fn=sample_greybox_bs8_config,
            outdir=outdir,
        )
        best_configs["greybox_bs8"] = dataclasses.asdict(best)

    # ------------------------------------------------------------- Multi-closure PI sweep
    if run_multi_closure_sweep:
        print("\n=== Multi-closure PI sweep (position 4) ===", flush=True)
        best = run_sweep(
            model_class="multi_closure",
            n_trials=n_trials,
            search_seed=args.search_seed + 19,
            per_trial_seeds=per_trial_seeds,
            trial_fn=lambda cfg, seeds: _run_multi_closure_trial(
                cfg, seeds, cohort_ids, train_sequences, val_sequences, args.device
            ),
            sample_fn=sample_multi_closure_config,
            outdir=outdir,
        )
        best_configs["multi_closure"] = dataclasses.asdict(best)

    # ------------------------------------------------------------- White-box sweep
    if run_whitebox_sweep:
        print("\n=== White-box sweep ===", flush=True)
        best = run_sweep(
            model_class="whitebox",
            n_trials=n_trials,
            search_seed=args.search_seed + 3,
            per_trial_seeds=per_trial_seeds,
            trial_fn=lambda cfg, seeds: _run_whitebox_trial(
                cfg, seeds, cohort_ids, train_sequences, val_sequences, args.device
            ),
            sample_fn=sample_whitebox_config,
            outdir=outdir,
        )
        best_configs["whitebox"] = dataclasses.asdict(best)

    # ------------------------------------------------------------- Final eval
    if run_final:
        for cls in ("blackbox", "greybox", "residual"):
            if cls not in best_configs:
                best_configs[cls] = _load_best_config(cls)

        # Promote each winning config from sweep-time epoch caps to the
        # manuscript-canonical training budget for the final test eval.
        bb_cfg = _promote_to_final_budget(best_configs["blackbox"]["config"], "blackbox")
        gb_cfg = _promote_to_final_budget(best_configs["greybox"]["config"], "greybox")
        res_cfg = _promote_to_final_budget(best_configs["residual"]["config"], "residual")

        final_payload: dict[str, Any] = {
            "cohort": {
                "include_p_codes": include_p_codes,
                "include_source_buckets": include_source_buckets,
                "sizes": {
                    "train": len(train_sequences),
                    "val": len(val_sequences),
                    "test": len(test_sequences),
                    "total": len(sequences),
                },
                "split_strategy": "md5",
                "warmup_steps": WARMUP_STEPS,
            },
            "best_configs": {
                "blackbox_core": bb_cfg,
                "greybox_learned_he_fullstate": gb_cfg,
                "residual_lstm_on_greybox": res_cfg,
            },
            "per_seed": {},
            "environment": env,
        }

        for seed in final_seeds:
            print(f"\n=== Final eval seed {seed} ===", flush=True)
            seed_payload: dict[str, Any] = {}

            # Black-box
            print(f"[seed {seed}] blackbox_core...", flush=True)
            seed_payload["blackbox_core"] = retrain_and_evaluate_blackbox(
                bb_cfg, seed, cohort_ids, grouped, CORE_FEATURES, args.device
            )

            # White-box: use grey-box's training schedule, no closure MLP to vary.
            print(f"[seed {seed}] whitebox_constant_he_fullstate...", flush=True)
            wb_payload, _ = retrain_and_evaluate_fullstate(
                model_factory=lambda: ConstantHeFullStateModel(),
                config=gb_cfg,
                seed=seed,
                cohort_ids=cohort_ids,
                train_sequences=train_sequences,
                val_sequences=val_sequences,
                test_sequences=test_sequences,
                device=args.device,
            )
            seed_payload[WHITEBOX_MODEL_NAME] = wb_payload

            # Grey-box: save checkpoint for residual stacking.
            print(f"[seed {seed}] greybox_learned_he_fullstate...", flush=True)
            ckpt_path = _greybox_checkpoint_path(outdir, seed)
            gb_payload, gb_model = retrain_and_evaluate_fullstate(
                model_factory=lambda: LearnedHeFullStateModel(
                    hidden_widths=tuple(gb_cfg["hidden_widths"])
                ),
                config=gb_cfg,
                seed=seed,
                cohort_ids=cohort_ids,
                train_sequences=train_sequences,
                val_sequences=val_sequences,
                test_sequences=test_sequences,
                device=args.device,
                save_checkpoint_path=ckpt_path,
            )
            seed_payload["greybox_learned_he_fullstate"] = gb_payload

            # Residual on the seed-matched grey-box base.
            print(f"[seed {seed}] residual_lstm_on_greybox...", flush=True)
            seed_payload["residual_lstm_on_greybox"] = retrain_and_evaluate_residual(
                res_cfg, seed, cohort_ids, train_sequences, val_sequences,
                test_sequences, gb_model, args.device,
            )

            del gb_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            final_payload["per_seed"][str(seed)] = seed_payload

        out_json = outdir / "final_test_metrics.json"
        out_json.write_text(json.dumps(final_payload, indent=2), encoding="utf-8")
        _write_final_markdown(outdir, final_payload, final_seeds)
        print(f"\nFinal test metrics saved to {out_json}", flush=True)

    # ------------------------------------------------- Targeted white-box final eval
    # Mutates an existing final_test_metrics.json: replaces only the
    # whitebox_constant_he_fullstate entry for each FINAL_SEED. Avoids
    # re-running the ~2-hour multi-model final eval.
    if run_final_whitebox_only:
        wb_cfg = _promote_to_final_budget(_load_best_config("whitebox")["config"], "whitebox")
        final_json_path = outdir / "final_test_metrics.json"
        if not final_json_path.exists():
            raise FileNotFoundError(
                f"{final_json_path} not found. Run --phase final_eval first so the "
                "other models are present; this phase only replaces the white-box entry."
            )
        final_payload = json.loads(final_json_path.read_text(encoding="utf-8"))
        final_payload.setdefault("best_configs", {})["whitebox_constant_he_fullstate"] = wb_cfg
        for seed in final_seeds:
            print(f"\n=== Final eval (whitebox-only) seed {seed} ===", flush=True)
            wb_payload, _ = retrain_and_evaluate_fullstate(
                model_factory=lambda: ConstantHeFullStateModel(),
                config=wb_cfg,
                seed=seed,
                cohort_ids=cohort_ids,
                train_sequences=train_sequences,
                val_sequences=val_sequences,
                test_sequences=test_sequences,
                device=args.device,
            )
            final_payload.setdefault("per_seed", {}).setdefault(str(seed), {})[WHITEBOX_MODEL_NAME] = wb_payload
        final_json_path.write_text(json.dumps(final_payload, indent=2), encoding="utf-8")
        _write_final_markdown(outdir, final_payload, final_seeds)
        print(f"\nUpdated white-box entries in {final_json_path}", flush=True)


def _write_final_markdown(
    outdir: Path, payload: dict[str, Any], final_seeds: Sequence[int]
) -> None:
    model_order = [
        ("whitebox_constant_he_fullstate", "Mechanistic"),
        ("greybox_learned_he_fullstate", "Physics Informed Model"),
        ("residual_lstm_on_greybox", "Residual LSTM"),
        ("blackbox_core", "Neural Net Baseline"),
    ]
    lines = ["# Final test metrics (HPO sweep)", ""]
    for seed in final_seeds:
        lines.append(f"## Seed {seed}")
        lines.append("")
        lines.append(
            "| Model | One-step R^2 | Rollout R^2 | Rollout R^2 95% CI | Rollout RMSE | Params |"
        )
        lines.append("|---|---:|---:|---|---:|---:|")
        seed_payload = payload["per_seed"].get(str(seed), {})
        for key, label in model_order:
            entry = seed_payload.get(key)
            if entry is None:
                continue
            one = entry["one_step_metrics"]
            roll = entry["rollout_metrics"]
            ci = entry["rollout_r2_ci95"]
            lines.append(
                f"| {label} | {one['r2']:.4f} | {roll['r2']:.4f} | "
                f"[{ci[0]:.4f}, {ci[1]:.4f}] | {roll['rmse']:.4f} | {entry['param_count']} |"
            )
        lines.append("")
    # Seed-stability ranges
    lines.append("## Seed stability (rollout R^2)")
    lines.append("")
    lines.append("| Model | min | max | range |")
    lines.append("|---|---:|---:|---:|")
    for key, label in model_order:
        values = [
            payload["per_seed"][str(seed)][key]["rollout_metrics"]["r2"]
            for seed in final_seeds
            if str(seed) in payload["per_seed"] and key in payload["per_seed"][str(seed)]
        ]
        if not values:
            continue
        lo = min(values)
        hi = max(values)
        lines.append(f"| {label} | {lo:.4f} | {hi:.4f} | {hi - lo:.4f} |")
    (outdir / "final_test_metrics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
