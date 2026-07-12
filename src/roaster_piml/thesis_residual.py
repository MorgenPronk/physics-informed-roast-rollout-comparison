"""Residual sequence model on top of the tuned grey-box full-state rollout."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import torch
from torch import nn

from .model import Metrics, compute_metrics
from .thesis_full_state import FullRoastBatch, FullRoastSequence, LearnedHeFullStateModel, make_batches


@dataclass
class ResidualRunResult:
    metrics: Dict[str, Dict[str, Metrics]]
    representative_roast_id: str
    predictions: Dict[str, Dict[str, List[float]]]


class ResidualLSTMModel(nn.Module):
    model_name = "residual_lstm_on_greybox"

    def __init__(self, hidden_size: int = 48, num_layers: int = 1, dropout: float = 0.0, max_delta: float = 18.0) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.max_delta = max_delta
        self.input_size = 7
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(self.input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=lstm_dropout)
        self.output = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )
        final_linear = self.output[-1]
        if isinstance(final_linear, nn.Linear):
            nn.init.zeros_(final_linear.weight)
            nn.init.zeros_(final_linear.bias)

    def _feature_vector(
        self,
        *,
        current_temp: torch.Tensor,
        base_next: torch.Tensor,
        batch: FullRoastBatch,
        step_index: int,
    ) -> torch.Tensor:
        return torch.stack(
            [
                (current_temp - 180.0) / 60.0,
                (base_next - 180.0) / 60.0,
                (base_next - current_temp) / 30.0,
                (batch.t2[:, step_index] - 150.0) / 100.0,
                batch.air_speed[:, step_index] / 20.0,
                batch.drum_speed[:, step_index] / 40.0,
                batch.flow_gas[:, step_index] / 100.0,
            ],
            dim=-1,
        )

    def rollout_batch(
        self,
        batch: FullRoastBatch,
        base_preds: torch.Tensor,
        *,
        warmup_steps: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, steps = base_preds.shape
        device = batch.tc.device
        corrected = torch.zeros((batch_size, steps), dtype=torch.float32, device=device)
        deltas = torch.zeros((batch_size, steps), dtype=torch.float32, device=device)

        hidden = None
        current_temp = batch.tc[:, 0]
        for t in range(steps):
            features = self._feature_vector(current_temp=current_temp, base_next=base_preds[:, t], batch=batch, step_index=t)
            output, hidden = self.lstm(features.unsqueeze(1), hidden)
            delta = self.max_delta * torch.tanh(self.output(output[:, 0]).squeeze(-1))
            next_temp = torch.clamp(base_preds[:, t] + delta, min=0.0, max=300.0)

            corrected[:, t] = next_temp
            deltas[:, t] = delta

            if t < warmup_steps - 1:
                current_temp = batch.tc[:, t + 1]
            else:
                current_temp = next_temp
        return corrected, deltas

    @torch.no_grad()
    def predict_rollout_with_warmup(
        self,
        seq: FullRoastSequence,
        base_model: LearnedHeFullStateModel,
        *,
        warmup_steps: int = 0,
    ) -> List[float]:
        device = next(self.parameters()).device
        batch = make_batches([seq], batch_size=1, device=device, shuffle=False)[0]
        base_preds, _, _, _ = base_model.rollout_batch(batch, teacher_forced=False, warmup_steps=warmup_steps)
        corrected, _ = self.rollout_batch(batch, base_preds, warmup_steps=warmup_steps)
        series = corrected[0, batch.mask[0]].detach().cpu().tolist()
        return series[warmup_steps:]

    @torch.no_grad()
    def predict_one_step(
        self,
        seq: FullRoastSequence,
        base_model: LearnedHeFullStateModel,
    ) -> List[float]:
        device = next(self.parameters()).device
        batch = make_batches([seq], batch_size=1, device=device, shuffle=False)[0]
        base_one_step, _, _, _ = base_model.rollout_batch(batch, teacher_forced=True)

        steps = base_one_step.shape[1]
        batch_size = 1
        corrected = torch.zeros((batch_size, steps), dtype=torch.float32, device=device)
        hidden = None
        current_temp = batch.tc[:, 0]
        for t in range(steps):
            features = self._feature_vector(current_temp=current_temp, base_next=base_one_step[:, t], batch=batch, step_index=t)
            output, hidden = self.lstm(features.unsqueeze(1), hidden)
            delta = self.max_delta * torch.tanh(self.output(output[:, 0]).squeeze(-1))
            next_temp = torch.clamp(base_one_step[:, t] + delta, min=0.0, max=300.0)
            corrected[:, t] = next_temp
            if t + 1 < batch.tc.shape[1]:
                current_temp = batch.tc[:, t + 1]
        return corrected[0, batch.mask[0]].detach().cpu().tolist()


class ResidualFeedForwardModel(nn.Module):
    """Bounded feedforward residual correction on top of a frozen physics-informed base.

    Mirrors :class:`ResidualLSTMModel` but uses a stateless feedforward MLP and
    the matched-input neural-baseline input set (current corrected temperature
    plus the three core drivers ``T_2, v_g, TT``). With ``hidden_widths=(32, 16)``
    its parameter count exactly matches the matched-input baseline (705
    parameters), which lets the same architecture be evaluated at two points on
    the physics-content spectrum: standalone (no physics) and as a bounded
    correction on top of a frozen physics-informed rollout.
    """

    model_name = "residual_ff_on_greybox"

    def __init__(
        self,
        hidden_widths: Sequence[int] = (32, 16),
        max_delta: float = 24.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        widths = tuple(int(w) for w in hidden_widths)
        if not widths:
            raise ValueError("hidden_widths must contain at least one layer")
        self.hidden_widths = widths
        self.max_delta = float(max_delta)
        self.input_size = 4  # current_Tc, T_2, v_g, TT — matches NN baseline
        # Metadata-compat attributes so the existing train_residual_model
        # trainer can introspect either LSTM or FF residual variants.
        self.hidden_size = int(max(widths))
        self.num_layers = int(len(widths))

        layers: list[nn.Module] = []
        prev = self.input_size
        for width in widths:
            layers.append(nn.Linear(prev, width))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(float(dropout)))
            prev = width
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)
        final = self.net[-1]
        if isinstance(final, nn.Linear):
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)

    def _feature_vector(
        self,
        *,
        current_temp: torch.Tensor,
        batch: FullRoastBatch,
        step_index: int,
    ) -> torch.Tensor:
        return torch.stack(
            [
                (current_temp - 180.0) / 60.0,
                (batch.t2[:, step_index] - 150.0) / 100.0,
                batch.air_speed[:, step_index] / 20.0,
                batch.drum_speed[:, step_index] / 40.0,
            ],
            dim=-1,
        )

    def rollout_batch(
        self,
        batch: FullRoastBatch,
        base_preds: torch.Tensor,
        *,
        warmup_steps: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, steps = base_preds.shape
        device = batch.tc.device
        corrected = torch.zeros((batch_size, steps), dtype=torch.float32, device=device)
        deltas = torch.zeros((batch_size, steps), dtype=torch.float32, device=device)

        current_temp = batch.tc[:, 0]
        for t in range(steps):
            features = self._feature_vector(current_temp=current_temp, batch=batch, step_index=t)
            delta = self.max_delta * torch.tanh(self.net(features).squeeze(-1))
            next_temp = torch.clamp(base_preds[:, t] + delta, min=0.0, max=300.0)

            corrected[:, t] = next_temp
            deltas[:, t] = delta

            if t < warmup_steps - 1:
                current_temp = batch.tc[:, t + 1]
            else:
                current_temp = next_temp
        return corrected, deltas

    @torch.no_grad()
    def predict_rollout_with_warmup(
        self,
        seq: FullRoastSequence,
        base_model: LearnedHeFullStateModel,
        *,
        warmup_steps: int = 0,
    ) -> List[float]:
        device = next(self.parameters()).device
        batch = make_batches([seq], batch_size=1, device=device, shuffle=False)[0]
        base_preds, _, _, _ = base_model.rollout_batch(batch, teacher_forced=False, warmup_steps=warmup_steps)
        corrected, _ = self.rollout_batch(batch, base_preds, warmup_steps=warmup_steps)
        series = corrected[0, batch.mask[0]].detach().cpu().tolist()
        return series[warmup_steps:]

    @torch.no_grad()
    def predict_one_step(
        self,
        seq: FullRoastSequence,
        base_model: LearnedHeFullStateModel,
    ) -> List[float]:
        device = next(self.parameters()).device
        batch = make_batches([seq], batch_size=1, device=device, shuffle=False)[0]
        base_one_step, _, _, _ = base_model.rollout_batch(batch, teacher_forced=True)

        steps = base_one_step.shape[1]
        corrected = torch.zeros((1, steps), dtype=torch.float32, device=device)
        current_temp = batch.tc[:, 0]
        for t in range(steps):
            features = self._feature_vector(current_temp=current_temp, batch=batch, step_index=t)
            delta = self.max_delta * torch.tanh(self.net(features).squeeze(-1))
            next_temp = torch.clamp(base_one_step[:, t] + delta, min=0.0, max=300.0)
            corrected[:, t] = next_temp
            if t + 1 < batch.tc.shape[1]:
                current_temp = batch.tc[:, t + 1]
        return corrected[0, batch.mask[0]].detach().cpu().tolist()


class NNWithPIFeaturesModel(nn.Module):
    """Neural-baseline architecture + physics-derived predictions as input features.

    Position 7.5 on the physics-content spectrum. Same feed-forward MLP shape
    and parameter count as the matched-input neural baseline, but augmented
    with 2 additional features per timestep: the frozen physics-informed
    model's next-step prediction and predicted increment. The model output is
    parameterized as ``base_pred + unbounded_delta`` so that zero-initialized
    weights leave the model trivially equal to the PI base at the start of
    training — same trick as the residual variants — but with no bound on the
    delta. This isolates the value of *seeing* PI predictions as features
    (compared to position 5b which only uses PI as a base offset).

    Notes:
        * Inputs use the same measured channels as the matched-input neural
          baseline (current_Tc, T_2, v_g, TT) plus 2 physics-derived channels
          (PI_next_Tc, PI_delta). No new measurement signals are introduced.
        * delta is unbounded.
    """

    model_name = "nn_with_pi_features"

    def __init__(
        self,
        hidden_widths: Sequence[int] = (32, 16),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        widths = tuple(int(w) for w in hidden_widths)
        if not widths:
            raise ValueError("hidden_widths must contain at least one layer")
        self.hidden_widths = widths
        self.input_size = 6  # current_Tc, T_2, v_g, TT, PI_next_Tc, PI_delta

        layers: list[nn.Module] = []
        prev = self.input_size
        for width in widths:
            layers.append(nn.Linear(prev, width))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(float(dropout)))
            prev = width
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)
        final = self.net[-1]
        if isinstance(final, nn.Linear):
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)
        # Compat attributes so train_residual_model can introspect this class.
        self.hidden_size = int(max(widths))
        self.num_layers = int(len(widths))
        self.max_delta = float("inf")

    def _feature_vector(
        self,
        *,
        current_temp: torch.Tensor,
        base_next: torch.Tensor,
        batch: FullRoastBatch,
        step_index: int,
    ) -> torch.Tensor:
        base_delta = base_next - current_temp
        return torch.stack(
            [
                (current_temp - 180.0) / 60.0,
                (batch.t2[:, step_index] - 150.0) / 100.0,
                batch.air_speed[:, step_index] / 20.0,
                batch.drum_speed[:, step_index] / 40.0,
                (base_next - 180.0) / 60.0,
                base_delta / 30.0,
            ],
            dim=-1,
        )

    def rollout_batch(
        self,
        batch: FullRoastBatch,
        base_preds: torch.Tensor,
        *,
        warmup_steps: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, steps = base_preds.shape
        device = batch.tc.device
        corrected = torch.zeros((batch_size, steps), dtype=torch.float32, device=device)
        deltas = torch.zeros((batch_size, steps), dtype=torch.float32, device=device)

        current_temp = batch.tc[:, 0]
        for t in range(steps):
            features = self._feature_vector(
                current_temp=current_temp, base_next=base_preds[:, t],
                batch=batch, step_index=t,
            )
            delta = self.net(features).squeeze(-1)  # unbounded
            next_temp = torch.clamp(base_preds[:, t] + delta, min=0.0, max=300.0)
            corrected[:, t] = next_temp
            deltas[:, t] = delta
            if t < warmup_steps - 1:
                current_temp = batch.tc[:, t + 1]
            else:
                current_temp = next_temp
        return corrected, deltas

    @torch.no_grad()
    def predict_rollout_with_warmup(
        self,
        seq: FullRoastSequence,
        base_model: LearnedHeFullStateModel,
        *,
        warmup_steps: int = 0,
    ) -> List[float]:
        device = next(self.parameters()).device
        batch = make_batches([seq], batch_size=1, device=device, shuffle=False)[0]
        base_preds, _, _, _ = base_model.rollout_batch(batch, teacher_forced=False, warmup_steps=warmup_steps)
        corrected, _ = self.rollout_batch(batch, base_preds, warmup_steps=warmup_steps)
        series = corrected[0, batch.mask[0]].detach().cpu().tolist()
        return series[warmup_steps:]

    @torch.no_grad()
    def predict_one_step(
        self,
        seq: FullRoastSequence,
        base_model: LearnedHeFullStateModel,
    ) -> List[float]:
        device = next(self.parameters()).device
        batch = make_batches([seq], batch_size=1, device=device, shuffle=False)[0]
        base_one_step, _, _, _ = base_model.rollout_batch(batch, teacher_forced=True)
        steps = base_one_step.shape[1]
        corrected = torch.zeros((1, steps), dtype=torch.float32, device=device)
        current_temp = batch.tc[:, 0]
        for t in range(steps):
            features = self._feature_vector(
                current_temp=current_temp, base_next=base_one_step[:, t],
                batch=batch, step_index=t,
            )
            delta = self.net(features).squeeze(-1)
            next_temp = torch.clamp(base_one_step[:, t] + delta, min=0.0, max=300.0)
            corrected[:, t] = next_temp
            if t + 1 < batch.tc.shape[1]:
                current_temp = batch.tc[:, t + 1]
        return corrected[0, batch.mask[0]].detach().cpu().tolist()


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid = values[mask]
    if valid.numel() == 0:
        return torch.zeros((), dtype=torch.float32, device=values.device)
    return torch.mean(valid)


def _residual_loss(
    residual_model: ResidualLSTMModel,
    base_model: LearnedHeFullStateModel,
    batch: FullRoastBatch,
    *,
    warmup_steps: int,
    residual_weight: float,
) -> torch.Tensor:
    with torch.no_grad():
        base_preds, _, _, _ = base_model.rollout_batch(batch, teacher_forced=False, warmup_steps=warmup_steps)
    corrected, deltas = residual_model.rollout_batch(batch, base_preds, warmup_steps=warmup_steps)

    mask = batch.mask.clone()
    if warmup_steps > 0:
        warmup_mask = torch.zeros_like(mask)
        warmup_mask[:, : min(warmup_steps, mask.shape[1])] = True
        mask = mask & ~warmup_mask
    target = batch.tc[:, 1:]
    tc_loss = _masked_mean((corrected - target) ** 2, mask)
    if residual_weight <= 0.0:
        return tc_loss
    reg_loss = _masked_mean(deltas**2, mask)
    return tc_loss + residual_weight * reg_loss


def train_residual_model(
    residual_model: ResidualLSTMModel,
    base_model: LearnedHeFullStateModel,
    train_sequences: Sequence[FullRoastSequence],
    val_sequences: Sequence[FullRoastSequence],
    *,
    epochs: int,
    lr: float,
    weight_decay: float,
    batch_size: int,
    device: str,
    warmup_steps: int,
    residual_weight: float,
    grad_clip: float = 1.0,
    patience: int | None = None,
) -> Dict[str, object]:
    torch_device = torch.device(device)
    residual_model.to(torch_device)
    base_model.to(torch_device)
    base_model.eval()
    optimizer = torch.optim.Adam(residual_model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val = float("inf")
    best_state = None
    history: List[Dict[str, float]] = []
    epochs_no_improve = 0
    epochs_run = 0
    best_epoch = 0

    for epoch in range(epochs):
        start = time.perf_counter()
        residual_model.train()
        train_batches = make_batches(train_sequences, batch_size=batch_size, device=torch_device, shuffle=True)
        train_losses: List[float] = []
        for batch in train_batches:
            optimizer.zero_grad()
            loss = _residual_loss(
                residual_model,
                base_model,
                batch,
                warmup_steps=warmup_steps,
                residual_weight=residual_weight,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(residual_model.parameters(), grad_clip)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        residual_model.eval()
        with torch.no_grad():
            val_batches = make_batches(val_sequences, batch_size=batch_size, device=torch_device, shuffle=False) if val_sequences else []
            val_losses = [
                float(
                    _residual_loss(
                        residual_model,
                        base_model,
                        batch,
                        warmup_steps=warmup_steps,
                        residual_weight=residual_weight,
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
            best_state = {k: v.detach().cpu().clone() for k, v in residual_model.state_dict().items()}
            best_epoch = epoch + 1
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if patience is not None and epochs_no_improve >= int(patience):
                break

    if best_state is not None:
        residual_model.load_state_dict(best_state)
    return {
        "best_val_loss": best_val,
        "best_epoch": int(best_epoch),
        "history": history,
        "batch_size": batch_size,
        "warmup_steps": warmup_steps,
        "residual_weight": residual_weight,
        "hidden_size": residual_model.hidden_size,
        "num_layers": residual_model.num_layers,
        "max_delta": residual_model.max_delta,
        "patience": int(patience) if patience is not None else None,
        "epochs_run": int(epochs_run),
    }


def evaluate_residual_model(
    residual_model: ResidualLSTMModel,
    base_model: LearnedHeFullStateModel,
    sequences: Sequence[FullRoastSequence],
    *,
    warmup_steps: int,
) -> Dict[str, Metrics]:
    one_true: List[float] = []
    one_pred: List[float] = []
    roll_true: List[float] = []
    roll_pred: List[float] = []
    for seq in sequences:
        if seq.length < 2:
            continue
        start_index = min(max(warmup_steps, 0), seq.length - 1)
        target = seq.tc[start_index + 1 :]
        one = residual_model.predict_one_step(seq, base_model)
        roll = residual_model.predict_rollout_with_warmup(seq, base_model, warmup_steps=warmup_steps)
        if start_index > 0:
            one = one[start_index:]
        one_true.extend(target)
        one_pred.extend(one)
        roll_true.extend(target)
        roll_pred.extend(roll)
    return {"one_step": compute_metrics(one_true, one_pred), "rollout": compute_metrics(roll_true, roll_pred)}


def build_residual_run_result(
    *,
    base_model: LearnedHeFullStateModel,
    residual_model: ResidualLSTMModel,
    test_sequences: Sequence[FullRoastSequence],
    warmup_steps: int,
) -> ResidualRunResult:
    greybox_metrics = {
        "one_step": compute_metrics([], []),
        "rollout": compute_metrics([], []),
    }
    one_true: List[float] = []
    one_pred: List[float] = []
    roll_true: List[float] = []
    roll_pred: List[float] = []
    for seq in test_sequences:
        if seq.length < 2:
            continue
        start_index = min(max(warmup_steps, 0), seq.length - 1)
        target = seq.tc[start_index + 1 :]
        one = base_model.predict_one_step(seq)
        roll = base_model.predict_rollout_with_warmup(seq, warmup_steps=warmup_steps)
        if start_index > 0:
            one = one[start_index:]
        one_true.extend(target)
        one_pred.extend(one)
        roll_true.extend(target)
        roll_pred.extend(roll)
    greybox_metrics = {"one_step": compute_metrics(one_true, one_pred), "rollout": compute_metrics(roll_true, roll_pred)}
    residual_metrics = evaluate_residual_model(residual_model, base_model, test_sequences, warmup_steps=warmup_steps)

    representative = max(test_sequences, key=lambda seq: seq.length) if test_sequences else None
    predictions: Dict[str, Dict[str, List[float]]] = {}
    representative_roast_id = representative.roast_id if representative else ""
    if representative:
        predictions["greybox_learned_he_fullstate"] = {
            "one_step": base_model.predict_one_step(representative)[warmup_steps:],
            "rollout": base_model.predict_rollout_with_warmup(representative, warmup_steps=warmup_steps),
        }
        predictions[residual_model.model_name] = {
            "one_step": residual_model.predict_one_step(representative, base_model)[warmup_steps:],
            "rollout": residual_model.predict_rollout_with_warmup(representative, base_model, warmup_steps=warmup_steps),
        }
        predictions["actual"] = {
            "one_step": representative.tc[1 + warmup_steps :],
            "rollout": representative.tc[1 + warmup_steps :],
        }

    return ResidualRunResult(
        metrics={
            "greybox_learned_he_fullstate": greybox_metrics,
            residual_model.model_name: residual_metrics,
        },
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


def save_checkpoint(path: Path, model: ResidualLSTMModel, metadata: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_name": model.model_name,
            "state_dict": model.state_dict(),
            "metadata": metadata,
            "hidden_size": model.hidden_size,
            "num_layers": model.num_layers,
            "max_delta": model.max_delta,
        },
        path,
    )


def load_checkpoint(path: Path, device: str = "cpu") -> tuple[ResidualLSTMModel, Dict[str, object]]:
    payload = torch.load(path, map_location=device)
    model = ResidualLSTMModel(
        hidden_size=int(payload.get("hidden_size", 48)),
        num_layers=int(payload.get("num_layers", 1)),
        max_delta=float(payload.get("max_delta", 18.0)),
    )
    model.load_state_dict(payload.get("state_dict", {}))
    model.to(torch.device(device))
    model.eval()
    metadata = payload.get("metadata", {})
    return model, metadata if isinstance(metadata, dict) else {}


def write_metrics_report(out_dir: Path, run_result: ResidualRunResult, metadata: Dict[str, object]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"metrics": metrics_to_serializable(run_result.metrics), "metadata": metadata}
    (out_dir / "thesis_residual_metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    rows = [
        "| Model | One-step R2 | Rollout R2 | One-step RMSE | Rollout RMSE |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in ["greybox_learned_he_fullstate", "residual_lstm_on_greybox"]:
        m = run_result.metrics[name]
        rows.append(f"| {name} | {m['one_step'].r2:.4f} | {m['rollout'].r2:.4f} | {m['one_step'].rmse:.4f} | {m['rollout'].rmse:.4f} |")
    (out_dir / "thesis_residual_metrics.md").write_text("\n".join(rows) + "\n", encoding="utf-8")
