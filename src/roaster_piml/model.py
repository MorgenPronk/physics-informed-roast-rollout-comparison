"""Simple physics-informed baseline model for roast temperature prediction."""

from __future__ import annotations

import csv
import datetime as dt
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .io_utils import parse_float, parse_timestamp, stable_split_key


@dataclass
class Metrics:
    r2: float
    rmse: float
    mae: float
    n: int


def compute_metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> Metrics:
    if not y_true:
        return Metrics(r2=float("nan"), rmse=float("nan"), mae=float("nan"), n=0)

    n = len(y_true)
    mean_true = sum(y_true) / n
    ss_res = sum((a - b) ** 2 for a, b in zip(y_true, y_pred))
    ss_tot = sum((a - mean_true) ** 2 for a in y_true)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else float("nan")
    rmse = math.sqrt(ss_res / n)
    mae = sum(abs(a - b) for a, b in zip(y_true, y_pred)) / n
    return Metrics(r2=r2, rmse=rmse, mae=mae, n=n)


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: Sequence[float], mean_value: float) -> float:
    if not values:
        return 1.0
    variance = sum((x - mean_value) ** 2 for x in values) / len(values)
    return max(math.sqrt(variance), 1e-8)


class PhysicsInformedDeltaModel:
    """Predict dTc/dt from current state with a constrained, interpretable feature map."""

    model_name = "physics_informed_linear_delta"

    def __init__(self, learning_rate: float = 5e-3, epochs: int = 20, grad_clip: float = 1.0):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.grad_clip = grad_clip
        self.weights: List[float] = []
        self.feature_means: List[float] = []
        self.feature_stds: List[float] = []
        self.target_mean = 0.0
        self.target_std = 1.0

    @staticmethod
    def features(row: Dict[str, float]) -> List[float]:
        tc = row.get("tc") or 0.0
        t1 = row.get("t1") or 0.0
        t2 = row.get("t2") or 0.0
        flow = row.get("flow_gas") or 0.0
        air = row.get("air_speed") or 0.0
        drum = row.get("drum_speed") or 0.0
        gas = row.get("gas_pressure") or 0.0
        set_bf = row.get("set_bf") or 0.0
        set_tt = row.get("set_tt") or 0.0

        # Physics-inspired heat-flow terms and controllable inputs.
        return [
            1.0,
            t1 - tc,
            t2 - tc,
            flow,
            air,
            drum,
            gas,
            set_bf,
            set_tt,
            tc,
        ]

    def _fit_normalization(self, feature_rows: Sequence[List[float]], y_delta: Sequence[float]) -> None:
        width = len(feature_rows[0])
        self.feature_means = []
        self.feature_stds = []
        for i in range(width):
            column = [row[i] for row in feature_rows]
            m = _mean(column)
            s = _std(column, m)
            self.feature_means.append(m)
            self.feature_stds.append(s)

        self.target_mean = _mean(y_delta)
        self.target_std = _std(y_delta, self.target_mean)

    def _normalize_feature(self, feature: List[float]) -> List[float]:
        return [
            (value - mean) / std
            for value, mean, std in zip(feature, self.feature_means, self.feature_stds)
        ]

    def _normalize_target(self, y: float) -> float:
        return (y - self.target_mean) / self.target_std

    def _denormalize_target(self, y_norm: float) -> float:
        return (y_norm * self.target_std) + self.target_mean

    def fit(self, x_rows: Sequence[Dict[str, float]], y_delta: Sequence[float]) -> None:
        if not x_rows:
            self.weights = []
            return

        feature_rows = [self.features(row) for row in x_rows]
        self._fit_normalization(feature_rows, y_delta)
        feature_rows = [self._normalize_feature(row) for row in feature_rows]
        y_norm = [self._normalize_target(value) for value in y_delta]

        width = len(feature_rows[0])
        self.weights = [0.0] * width

        for _ in range(self.epochs):
            for feats, target in zip(feature_rows, y_norm):
                pred = sum(w * f for w, f in zip(self.weights, feats))
                err = pred - target
                for i, value in enumerate(feats):
                    grad = 2.0 * err * value
                    grad = max(min(grad, self.grad_clip), -self.grad_clip)
                    self.weights[i] -= self.learning_rate * grad

    def predict_delta(self, rows: Sequence[Dict[str, float]]) -> List[float]:
        predictions: List[float] = []
        for row in rows:
            feats = self._normalize_feature(self.features(row))
            pred_norm = sum(w * f for w, f in zip(self.weights, feats))
            pred = self._denormalize_target(pred_norm)
            if not math.isfinite(pred):
                pred = 0.0
            predictions.append(pred)
        return predictions

    def predict_next_tc(self, rows: Sequence[Dict[str, float]]) -> List[float]:
        delta = self.predict_delta(rows)
        output: List[float] = []
        for row, d_tc in zip(rows, delta):
            tc = row.get("tc") or 0.0
            output.append(tc + d_tc)
        return output

    def to_dict(self) -> Dict[str, object]:
        return {
            "model_name": self.model_name,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "grad_clip": self.grad_clip,
            "weights": self.weights,
            "feature_means": self.feature_means,
            "feature_stds": self.feature_stds,
            "target_mean": self.target_mean,
            "target_std": self.target_std,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "PhysicsInformedDeltaModel":
        model = cls(
            learning_rate=float(payload.get("learning_rate", 5e-3)),
            epochs=int(payload.get("epochs", 20)),
            grad_clip=float(payload.get("grad_clip", 1.0)),
        )
        model.weights = [float(x) for x in payload.get("weights", [])]
        model.feature_means = [float(x) for x in payload.get("feature_means", [])]
        model.feature_stds = [float(x) for x in payload.get("feature_stds", [])]
        model.target_mean = float(payload.get("target_mean", 0.0))
        model.target_std = max(float(payload.get("target_std", 1.0)), 1e-8)
        return model

    def save_json(self, path: Path, metadata: Optional[Dict[str, object]] = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": self.to_dict(),
            "metadata": metadata or {},
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load_json(cls, path: Path) -> Tuple["PhysicsInformedDeltaModel", Dict[str, object]]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if "model" in payload:
            model_payload = payload.get("model", {})
            metadata = payload.get("metadata", {})
        else:
            model_payload = payload
            metadata = {}
        return cls.from_dict(model_payload), metadata


def _parse_dataset_row(row: Dict[str, str]) -> Dict[str, float]:
    parsed: Dict[str, float] = {}
    for key, value in row.items():
        if key in ("roast_id", "timestamp"):
            continue
        number = parse_float(value)
        if number is not None:
            parsed[key] = number
    return parsed


def load_supervised_rows(path: Path) -> List[Tuple[str, dt.datetime, Dict[str, float]]]:
    records: List[Tuple[str, dt.datetime, Dict[str, float]]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            roast_id = row.get("roast_id", "unknown")
            ts = parse_timestamp(row.get("timestamp", ""))
            if ts is None:
                continue
            parsed = _parse_dataset_row(row)
            if "tc" not in parsed:
                continue
            records.append((roast_id, ts, parsed))
    records.sort(key=lambda x: (x[0], x[1]))
    return records


def build_next_step_dataset(records: Sequence[Tuple[str, dt.datetime, Dict[str, float]]]) -> Tuple[List[Dict[str, float]], List[float], List[str]]:
    x_rows: List[Dict[str, float]] = []
    y_next_tc: List[float] = []
    roast_ids: List[str] = []

    for idx in range(len(records) - 1):
        roast_a, ts_a, row_a = records[idx]
        roast_b, ts_b, row_b = records[idx + 1]
        if roast_a != roast_b:
            continue
        delta_t = (ts_b - ts_a).total_seconds()
        if delta_t <= 0 or delta_t > 120:
            continue
        if "tc" not in row_b:
            continue
        x_rows.append(row_a)
        y_next_tc.append(row_b["tc"])
        roast_ids.append(roast_a)

    return x_rows, y_next_tc, roast_ids


def split_by_roast(
    x_rows: Sequence[Dict[str, float]],
    y_next_tc: Sequence[float],
    roast_ids: Sequence[str],
    train_ratio: float = 0.8,
) -> Tuple[List[Dict[str, float]], List[float], List[Dict[str, float]], List[float]]:
    x_train: List[Dict[str, float]] = []
    y_train: List[float] = []
    x_test: List[Dict[str, float]] = []
    y_test: List[float] = []

    threshold = int(train_ratio * 100)
    for row, target, roast_id in zip(x_rows, y_next_tc, roast_ids):
        bucket = stable_split_key(roast_id)
        if bucket < threshold:
            x_train.append(row)
            y_train.append(target)
        else:
            x_test.append(row)
            y_test.append(target)

    return x_train, y_train, x_test, y_test


def summarize_and_write(
    out_dir: Path,
    y_test: Sequence[float],
    pred_naive: Sequence[float],
    pred_model: Sequence[float],
    x_test: Sequence[Dict[str, float]],
) -> Dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)

    naive_metrics = compute_metrics(y_test, pred_naive)
    model_metrics = compute_metrics(y_test, pred_model)

    summary = {
        "naive": asdict(naive_metrics),
        "physics_informed_linear": asdict(model_metrics),
    }

    (out_dir / "baseline_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    table_md = "\n".join(
        [
            "| Model | R2 | RMSE | MAE | N |",
            "|---|---:|---:|---:|---:|",
            f"| Naive persistence | {naive_metrics.r2:.4f} | {naive_metrics.rmse:.4f} | {naive_metrics.mae:.4f} | {naive_metrics.n} |",
            f"| Physics-informed linear delta | {model_metrics.r2:.4f} | {model_metrics.rmse:.4f} | {model_metrics.mae:.4f} | {model_metrics.n} |",
        ]
    )
    (out_dir / "baseline_metrics.md").write_text(table_md + "\n", encoding="utf-8")

    full_rows: List[Dict[str, object]] = []
    for idx, (truth, naive, model, row) in enumerate(zip(y_test, pred_naive, pred_model, x_test)):
        full_rows.append(
            {
                "index": idx,
                "tc_current": row.get("tc", ""),
                "tc_next_actual": truth,
                "tc_next_naive": naive,
                "tc_next_model": model,
                "error_naive": naive - truth,
                "error_model": model - truth,
            }
        )

    with (out_dir / "predictions_full.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(full_rows[0].keys()) if full_rows else ["index"])
        writer.writeheader()
        writer.writerows(full_rows)

    preview_rows = full_rows[:2000]
    with (out_dir / "predictions_preview.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(preview_rows[0].keys()) if preview_rows else ["index"])
        writer.writeheader()
        writer.writerows(preview_rows)

    return summary
