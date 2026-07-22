import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from arogyaflow.exceptions import ConfigurationError, DataQualityError
from arogyaflow.time import utc_now


class MonitoringThresholds(BaseModel):
    model_config = ConfigDict(frozen=True)

    max_missing_rate: float = Field(default=0.05, ge=0, le=1)
    max_duplicate_rate: float = Field(default=0.01, ge=0, le=1)
    max_drift_score: float = Field(default=0.2, ge=0)
    max_relative_metric_degradation: float = Field(default=0.1, ge=0)


class MonitoringAlert(BaseModel):
    component: Literal["data_quality", "drift", "performance"]
    metric: str
    observed: float
    threshold: float
    message: str


class MonitoringReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    generated_at: datetime
    status: Literal["healthy", "review_required"]
    synthetic_data_only: bool = True
    reference_rows: int
    current_rows: int
    missing_rate: float
    duplicate_rate: float
    drift_scores: dict[str, float]
    metric_degradation: dict[str, float]
    alerts: list[MonitoringAlert]
    retraining_recommended: bool = False
    retraining_blocked: bool = False


def _numeric_drift(reference: pd.Series, current: pd.Series, bins: int) -> float:
    reference_values = reference.dropna().to_numpy(dtype=float)
    current_values = current.dropna().to_numpy(dtype=float)
    if not len(reference_values) or not len(current_values):
        return 1.0
    edges = np.unique(np.quantile(reference_values, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        return 0.0 if np.allclose(current_values, reference_values[0]) else 1.0
    edges[0], edges[-1] = -np.inf, np.inf
    reference_share = np.histogram(reference_values, bins=edges)[0] / len(reference_values)
    current_share = np.histogram(current_values, bins=edges)[0] / len(current_values)
    reference_share = np.clip(reference_share, 1e-6, None)
    current_share = np.clip(current_share, 1e-6, None)
    return float(
        np.sum((current_share - reference_share) * np.log(current_share / reference_share))
    )


def _categorical_drift(reference: pd.Series, current: pd.Series) -> float:
    reference_share = reference.fillna("<missing>").astype(str).value_counts(normalize=True)
    current_share = current.fillna("<missing>").astype(str).value_counts(normalize=True)
    categories = reference_share.index.union(current_share.index)
    difference = reference_share.reindex(categories, fill_value=0) - current_share.reindex(
        categories, fill_value=0
    )
    return float(difference.abs().sum() / 2)


def feature_drift(
    reference: pd.DataFrame, current: pd.DataFrame, *, bins: int = 10
) -> dict[str, float]:
    if bins < 2:
        raise ValueError("bins must be at least 2")
    scores: dict[str, float] = {}
    for column in sorted(set(reference.columns) & set(current.columns)):
        if pd.api.types.is_numeric_dtype(reference[column]) and pd.api.types.is_numeric_dtype(
            current[column]
        ):
            scores[column] = _numeric_drift(reference[column], current[column], bins)
        else:
            scores[column] = _categorical_drift(reference[column], current[column])
    return scores


def metric_degradation(reference: dict[str, float], current: dict[str, float]) -> dict[str, float]:
    higher_is_better = {
        "f1",
        "interval_coverage",
        "pr_auc",
        "precision",
        "recall",
        "roc_auc",
    }
    degradation: dict[str, float] = {}
    for metric in sorted(reference):
        if metric not in current:
            degradation[metric] = 1.0
            continue
        baseline = reference[metric]
        observed = current[metric]
        denominator = max(abs(baseline), 1e-9)
        change = baseline - observed if metric in higher_is_better else observed - baseline
        degradation[metric] = max(0.0, change / denominator)
    return degradation


def build_monitoring_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    reference_metrics: dict[str, float],
    current_metrics: dict[str, float],
    required_columns: tuple[str, ...],
    thresholds: MonitoringThresholds,
    *,
    synthetic_data_only: bool = True,
) -> MonitoringReport:
    if reference.empty or current.empty:
        raise DataQualityError("Monitoring datasets cannot be empty")
    if missing := set(required_columns) - set(reference.columns):
        raise DataQualityError(f"Reference data is missing columns: {sorted(missing)}")
    current_missing_columns = set(required_columns) - set(current.columns)
    present = [column for column in required_columns if column in current.columns]
    missing_cells = int(current[present].isna().sum().sum()) if present else 0
    expected_cells = len(current) * max(len(required_columns), 1)
    missing_cells += len(current) * len(current_missing_columns)
    missing_rate = missing_cells / expected_cells
    duplicate_rate = float(current.duplicated().mean())
    drift = feature_drift(reference[present], current[present]) if present else {}
    degradation = metric_degradation(reference_metrics, current_metrics)
    alerts: list[MonitoringAlert] = []

    checks: list[tuple[str, str, float, float, str]] = [
        (
            "data_quality",
            "missing_rate",
            missing_rate,
            thresholds.max_missing_rate,
            "Required-field missing rate exceeded its threshold.",
        ),
        (
            "data_quality",
            "duplicate_rate",
            duplicate_rate,
            thresholds.max_duplicate_rate,
            "Duplicate-row rate exceeded its threshold.",
        ),
    ]
    checks.extend(
        ("drift", name, score, thresholds.max_drift_score, "Feature drift requires review.")
        for name, score in drift.items()
    )
    checks.extend(
        (
            "performance",
            name,
            score,
            thresholds.max_relative_metric_degradation,
            "Model metric degradation requires review.",
        )
        for name, score in degradation.items()
    )
    for component, metric, observed, threshold, message in checks:
        if observed > threshold:
            alerts.append(
                MonitoringAlert(
                    component=cast_component(component),
                    metric=metric,
                    observed=observed,
                    threshold=threshold,
                    message=message,
                )
            )
    quality_alert = any(alert.component == "data_quality" for alert in alerts)
    model_alert = any(alert.component in {"drift", "performance"} for alert in alerts)
    return MonitoringReport(
        generated_at=utc_now(),
        status="review_required" if alerts else "healthy",
        synthetic_data_only=synthetic_data_only,
        reference_rows=len(reference),
        current_rows=len(current),
        missing_rate=missing_rate,
        duplicate_rate=duplicate_rate,
        drift_scores=drift,
        metric_degradation=degradation,
        alerts=alerts,
        retraining_recommended=model_alert and not quality_alert,
        retraining_blocked=quality_alert,
    )


def cast_component(value: str) -> Literal["data_quality", "drift", "performance"]:
    if value not in {"data_quality", "drift", "performance"}:
        raise ValueError(f"Unknown monitoring component: {value}")
    return value  # type: ignore[return-value]


def load_monitoring_report(path: Path | None) -> MonitoringReport:
    if path is None:
        raise ConfigurationError("Monitoring report path is not configured")
    if not path.is_file():
        raise ConfigurationError("Monitoring report does not exist")
    try:
        return MonitoringReport.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise DataQualityError("Monitoring report is invalid") from exc


def _read_metrics(path: Path) -> dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise DataQualityError("Metrics file must contain a JSON object")
    metrics: dict[str, float] = {}
    for name, value in payload.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise DataQualityError(f"Metric {name} must be numeric")
        number = float(value)
        if not math.isfinite(number):
            raise DataQualityError(f"Metric {name} must be finite")
        metrics[str(name)] = number
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--current", required=True, type=Path)
    parser.add_argument("--reference-metrics", required=True, type=Path)
    parser.add_argument("--current-metrics", required=True, type=Path)
    parser.add_argument("--required-column", action="append", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-missing-rate", type=float, default=0.05)
    parser.add_argument("--max-duplicate-rate", type=float, default=0.01)
    parser.add_argument("--max-drift-score", type=float, default=0.2)
    parser.add_argument("--max-metric-degradation", type=float, default=0.1)
    parser.add_argument("--real-data", action="store_true")
    args = parser.parse_args()
    report = build_monitoring_report(
        pd.read_parquet(args.reference),
        pd.read_parquet(args.current),
        _read_metrics(args.reference_metrics),
        _read_metrics(args.current_metrics),
        tuple(args.required_column),
        MonitoringThresholds(
            max_missing_rate=args.max_missing_rate,
            max_duplicate_rate=args.max_duplicate_rate,
            max_drift_score=args.max_drift_score,
            max_relative_metric_degradation=args.max_metric_degradation,
        ),
        synthetic_data_only=not args.real_data,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report.model_dump_json(indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
