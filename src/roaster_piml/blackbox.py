"""Configurable autoregressive black-box MLP with val-loss early stopping.

This module is a generalization of the hand-tuned baseline in
``scripts/run_main_cohort_experiments.py``. The original baseline used a fixed
two-layer MLP with full-batch training and no early stopping. The variant here
accepts arbitrary hidden widths, weight decay, dropout, mini-batching, and
patience-based early stopping while preserving the same rollout semantics
(predict next ``Tc`` from previous ``Tc`` + per-step exogenous controls).
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn


class AutoRegressiveMLP(nn.Module):
    """Feed-forward MLP with configurable depth, width, and dropout."""

    def __init__(
        self,
        input_dim: int,
        hidden_widths: Sequence[int] = (64, 32),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        widths = tuple(int(w) for w in hidden_widths)
        layers: list[nn.Module] = []
        prev = int(input_dim)
        for width in widths:
            layers.append(nn.Linear(prev, width))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(float(dropout)))
            prev = width
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)
        self.input_dim = int(input_dim)
        self.hidden_widths = widths
        self.dropout = float(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _build_arrays(
    grouped: dict[str, pd.DataFrame],
    roast_ids: Sequence[str],
    feature_columns: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    x_rows: list[np.ndarray] = []
    y_rows: list[np.ndarray] = []
    for roast_id in roast_ids:
        df = grouped.get(roast_id)
        if df is None or len(df) < 2:
            continue
        tc_prev = df["tc"].to_numpy(dtype=float)[:-1]
        controls = df[list(feature_columns)].to_numpy(dtype=float)[:-1]
        x_rows.append(np.column_stack([tc_prev, controls]))
        y_rows.append(df["tc"].to_numpy(dtype=float)[1:].reshape(-1, 1))
    if not x_rows:
        raise RuntimeError("No training rows available for black-box baseline.")
    return np.vstack(x_rows), np.vstack(y_rows)


def _fit_scalers(x: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
    x_mean = x.mean(axis=0)
    x_std = x.std(axis=0)
    x_std[x_std == 0] = 1.0
    y_mean = y.mean(axis=0)
    y_std = y.std(axis=0)
    y_std[y_std == 0] = 1.0
    return {"x_mean": x_mean, "x_std": x_std, "y_mean": y_mean, "y_std": y_std}


def set_blackbox_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_blackbox(
    grouped: dict[str, pd.DataFrame],
    train_ids: Sequence[str],
    val_ids: Sequence[str],
    *,
    feature_columns: Sequence[str],
    hidden_widths: Sequence[int] = (64, 32),
    dropout: float = 0.0,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    batch_size: int | str = "full",
    epochs: int = 200,
    patience: int | None = 20,
    grad_clip: float = 1.0,
    device: str = "cpu",
    seed: int = 11,
) -> tuple[AutoRegressiveMLP, dict[str, np.ndarray], dict[str, Any]]:
    """Train an AR-MLP, selecting weights by best validation loss.

    Returns (model, scalers, metadata). The returned model has its weights
    reset to the best-val checkpoint before return.
    """
    set_blackbox_seed(seed)

    x_train, y_train = _build_arrays(grouped, train_ids, feature_columns)
    scalers = _fit_scalers(x_train, y_train)
    x_train_t = torch.tensor(
        (x_train - scalers["x_mean"]) / scalers["x_std"],
        dtype=torch.float32,
        device=device,
    )
    y_train_t = torch.tensor(
        (y_train - scalers["y_mean"]) / scalers["y_std"],
        dtype=torch.float32,
        device=device,
    )

    x_val_t = None
    y_val_t = None
    if val_ids:
        x_val, y_val = _build_arrays(grouped, val_ids, feature_columns)
        x_val_t = torch.tensor(
            (x_val - scalers["x_mean"]) / scalers["x_std"],
            dtype=torch.float32,
            device=device,
        )
        y_val_t = torch.tensor(
            (y_val - scalers["y_mean"]) / scalers["y_std"],
            dtype=torch.float32,
            device=device,
        )

    model = AutoRegressiveMLP(
        input_dim=int(x_train_t.shape[1]),
        hidden_widths=hidden_widths,
        dropout=dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    n_train = int(x_train_t.shape[0])
    use_full_batch = batch_size == "full" or batch_size is None or int(batch_size) >= n_train
    batch_int = None if use_full_batch else int(batch_size)

    best_val = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    epochs_no_improve = 0
    history: list[dict[str, float]] = []
    epochs_run = 0
    best_epoch = 0

    for epoch in range(int(epochs)):
        model.train()
        if use_full_batch:
            optimizer.zero_grad()
            pred = model(x_train_t)
            loss = loss_fn(pred, y_train_t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            train_loss_value = float(loss.detach().cpu())
        else:
            perm = torch.randperm(n_train, device=device)
            x_shuf = x_train_t[perm]
            y_shuf = y_train_t[perm]
            batch_losses: list[float] = []
            for start in range(0, n_train, batch_int):
                xb = x_shuf[start : start + batch_int]
                yb = y_shuf[start : start + batch_int]
                optimizer.zero_grad()
                loss = loss_fn(model(xb), yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                batch_losses.append(float(loss.detach().cpu()))
            train_loss_value = sum(batch_losses) / max(len(batch_losses), 1)

        model.eval()
        with torch.no_grad():
            if x_val_t is not None and y_val_t is not None:
                val_loss_value = float(loss_fn(model(x_val_t), y_val_t).detach().cpu())
            else:
                val_loss_value = train_loss_value

        history.append(
            {
                "epoch": float(epoch + 1),
                "train_loss": train_loss_value,
                "val_loss": val_loss_value,
            }
        )
        epochs_run = epoch + 1

        if val_loss_value < best_val - 1e-9:
            best_val = val_loss_value
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            best_epoch = epoch + 1
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if patience is not None and epochs_no_improve >= int(patience):
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    metadata = {
        "best_val_loss": best_val,
        "best_epoch": int(best_epoch),
        "epochs_run": int(epochs_run),
        "history": history,
        "hidden_widths": list(model.hidden_widths),
        "dropout": float(model.dropout),
        "lr": float(lr),
        "weight_decay": float(weight_decay),
        "batch_size": "full" if use_full_batch else int(batch_int),
        "patience": int(patience) if patience is not None else None,
        "feature_columns": list(feature_columns),
        "seed": int(seed),
    }
    return model, scalers, metadata


def rollout_blackbox_trial(
    model: AutoRegressiveMLP,
    scalers: dict[str, np.ndarray],
    roast_df: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    controls = roast_df[list(feature_columns)].to_numpy(dtype=float)
    actual = roast_df["tc"].to_numpy(dtype=float)
    pred = np.zeros_like(actual)
    pred[0] = actual[0]
    model.eval()
    with torch.no_grad():
        for step in range(1, len(roast_df)):
            x_row = np.concatenate([[pred[step - 1]], controls[step - 1]])
            x_scaled = (x_row - scalers["x_mean"]) / scalers["x_std"]
            x_tensor = torch.tensor(x_scaled, dtype=torch.float32, device=device).unsqueeze(0)
            y_scaled = model(x_tensor).cpu().numpy()[0, 0]
            pred[step] = y_scaled * scalers["y_std"][0] + scalers["y_mean"][0]
    return actual, pred


def one_step_blackbox_trial(
    model: AutoRegressiveMLP,
    scalers: dict[str, np.ndarray],
    roast_df: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    device: str,
) -> np.ndarray:
    controls = roast_df[list(feature_columns)].to_numpy(dtype=float)
    actual = roast_df["tc"].to_numpy(dtype=float)
    preds = np.zeros(len(actual) - 1, dtype=float)
    model.eval()
    with torch.no_grad():
        for step in range(len(actual) - 1):
            x_row = np.concatenate([[actual[step]], controls[step]])
            x_scaled = (x_row - scalers["x_mean"]) / scalers["x_std"]
            x_tensor = torch.tensor(x_scaled, dtype=torch.float32, device=device).unsqueeze(0)
            y_scaled = model(x_tensor).cpu().numpy()[0, 0]
            preds[step] = y_scaled * scalers["y_std"][0] + scalers["y_mean"][0]
    return preds


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
