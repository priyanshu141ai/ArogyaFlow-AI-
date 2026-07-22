import argparse
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, cast

import joblib  # type: ignore[import-untyped]
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
from sklearn.compose import ColumnTransformer  # type: ignore[import-untyped]
from sklearn.ensemble import HistGradientBoostingRegressor  # type: ignore[import-untyped]
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # type: ignore[import-untyped]

from arogyaflow.analysis import build_arrival_target
from arogyaflow.baseline_pipeline import load_bronze
from arogyaflow.data.generation import Dataset
from arogyaflow.exceptions import ForecastHorizonError, ModelArtifactError, TrainingDataError
from arogyaflow.time import utc_now
from arogyaflow.tracking import ModelKind, TrackingOptions, flatten_metrics, log_training_run

FORECAST_SCHEMA_VERSION = "1.0"
SUPPORTED_HORIZONS = (6, 12, 24)
FEATURE_COLUMNS = (
    "department_id",
    "hour",
    "weekday",
    "is_weekend",
    "lag_1",
    "lag_24",
    "lag_168",
    "rolling_24",
)


class ArrivalTrainingConfig(TrackingOptions):
    model_config = ConfigDict(frozen=True)

    model_version: str = Field(min_length=1)
    random_seed: int
    backtest_windows: int = Field(default=3, ge=2, le=8)
    experiment_name: str = "arogyaflow-arrivals"
    registered_model_name: str = "arogyaflow-arrivals"


class ForecastPoint(BaseModel):
    interval_start: datetime
    hospital_id: str
    department_id: str | None
    level: Literal["department", "hospital"]
    predicted_arrivals: float
    lower_arrivals: float
    upper_arrivals: float


class ArrivalForecast(BaseModel):
    model_version: str
    schema_version: str
    generated_at: datetime
    horizon_hours: Literal[6, 12, 24]
    reconciliation: Literal["bottom_up"] = "bottom_up"
    points: list[ForecastPoint]


@dataclass
class ArrivalArtifact:
    model_version: str
    schema_version: str
    preprocessor: ColumnTransformer
    point_model: Any
    lower_model: Any
    upper_model: Any
    history: pd.DataFrame


@dataclass(frozen=True)
class ArrivalTrainingResult:
    artifact_path: Path
    report_path: Path
    report: dict[str, Any]
    mlflow_run_id: str | None


def build_forecast_features(target: pd.DataFrame) -> pd.DataFrame:
    required = {"hospital_id", "department_id", "interval_start", "arrivals"}
    if missing := required - set(target.columns):
        raise TrainingDataError(f"Missing arrival target columns: {sorted(missing)}")
    frames: list[pd.DataFrame] = []
    for _, group in target.groupby(["hospital_id", "department_id"], sort=True):
        ordered = group.sort_values("interval_start").copy()
        ordered["hour"] = ordered["interval_start"].dt.hour
        ordered["weekday"] = ordered["interval_start"].dt.weekday
        ordered["is_weekend"] = (ordered["weekday"] >= 5).astype(int)
        ordered["lag_1"] = ordered["arrivals"].shift(1)
        ordered["lag_24"] = ordered["arrivals"].shift(24)
        ordered["lag_168"] = ordered["arrivals"].shift(168)
        ordered["rolling_24"] = ordered["arrivals"].shift(1).rolling(24).mean()
        frames.append(ordered)
    features = pd.concat(frames, ignore_index=True).dropna().reset_index(drop=True)
    if features.empty:
        raise TrainingDataError("At least eight days of hourly arrival history are required")
    return features


def _preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            (
                "department",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ["department_id"],
            ),
            (
                "numeric",
                StandardScaler(),
                [column for column in FEATURE_COLUMNS if column != "department_id"],
            ),
        ],
        verbose_feature_names_out=False,
    )


def _fit_artifact(target: pd.DataFrame, model_version: str, random_seed: int) -> ArrivalArtifact:
    features = build_forecast_features(target)
    processor = _preprocessor()
    transformed = processor.fit_transform(features[list(FEATURE_COLUMNS)])
    point = HistGradientBoostingRegressor(
        max_iter=100, max_leaf_nodes=15, random_state=random_seed
    ).fit(transformed, features["arrivals"])
    lower = HistGradientBoostingRegressor(
        loss="quantile", quantile=0.1, max_iter=100, random_state=random_seed
    ).fit(transformed, features["arrivals"])
    upper = HistGradientBoostingRegressor(
        loss="quantile", quantile=0.9, max_iter=100, random_state=random_seed
    ).fit(transformed, features["arrivals"])
    return ArrivalArtifact(
        model_version=model_version,
        schema_version=FORECAST_SCHEMA_VERSION,
        preprocessor=processor,
        point_model=point,
        lower_model=lower,
        upper_model=upper,
        history=target.copy(),
    )


def _department_forecast(artifact: ArrivalArtifact, horizon: int) -> pd.DataFrame:
    predictions: list[dict[str, object]] = []
    for (hospital_id, department_id), group in artifact.history.groupby(
        ["hospital_id", "department_id"], sort=True
    ):
        values: dict[pd.Timestamp, float] = {}
        for interval, arrivals in group[["interval_start", "arrivals"]].itertuples(
            index=False, name=None
        ):
            values[pd.Timestamp(cast(Any, interval))] = float(cast(Any, arrivals))
        last_time = max(values)
        for step in range(1, horizon + 1):
            timestamp = last_time + pd.Timedelta(hours=step)
            recent = [values[timestamp - pd.Timedelta(hours=offset)] for offset in range(1, 25)]
            row = pd.DataFrame(
                [
                    {
                        "department_id": str(department_id),
                        "hour": timestamp.hour,
                        "weekday": timestamp.weekday(),
                        "is_weekend": int(timestamp.weekday() >= 5),
                        "lag_1": values[timestamp - pd.Timedelta(hours=1)],
                        "lag_24": values[timestamp - pd.Timedelta(hours=24)],
                        "lag_168": values[timestamp - pd.Timedelta(hours=168)],
                        "rolling_24": sum(recent) / 24,
                    }
                ]
            )
            transformed = artifact.preprocessor.transform(row[list(FEATURE_COLUMNS)])
            point = max(0.0, float(artifact.point_model.predict(transformed)[0]))
            lower = max(0.0, min(point, float(artifact.lower_model.predict(transformed)[0])))
            upper = max(point, float(artifact.upper_model.predict(transformed)[0]))
            values[timestamp] = point
            predictions.append(
                {
                    "interval_start": timestamp,
                    "hospital_id": str(hospital_id),
                    "department_id": str(department_id),
                    "predicted_arrivals": point,
                    "lower_arrivals": lower,
                    "upper_arrivals": upper,
                }
            )
    return pd.DataFrame(predictions)


def forecast_arrivals(artifact: ArrivalArtifact, horizon_hours: int) -> ArrivalForecast:
    if horizon_hours not in SUPPORTED_HORIZONS:
        raise ForecastHorizonError(f"Supported horizons: {SUPPORTED_HORIZONS}")
    department = _department_forecast(artifact, horizon_hours)
    department_records = cast(list[dict[str, Any]], department.to_dict(orient="records"))
    points = [ForecastPoint(level="department", **row) for row in department_records]
    hospital = (
        department.groupby(["interval_start", "hospital_id"], as_index=False)[
            ["predicted_arrivals", "lower_arrivals", "upper_arrivals"]
        ]
        .sum()
        .assign(department_id=None)
    )
    hospital_records = cast(list[dict[str, Any]], hospital.to_dict(orient="records"))
    points.extend(ForecastPoint(level="hospital", **row) for row in hospital_records)
    return ArrivalForecast(
        model_version=artifact.model_version,
        schema_version=artifact.schema_version,
        generated_at=utc_now(),
        horizon_hours=cast(Literal[6, 12, 24], horizon_hours),
        points=points,
    )


def _metrics(actual: np.ndarray[Any, Any], predicted: np.ndarray[Any, Any]) -> dict[str, float]:
    error = predicted - actual
    absolute = np.abs(error)
    return {
        "mae": float(absolute.mean()),
        "rmse": math.sqrt(float(np.mean(error**2))),
        "wape": float(absolute.sum()) / max(float(np.abs(actual).sum()), 1.0),
        "bias": float(error.mean()),
    }


def _baseline_predictions(
    history: pd.DataFrame, future: pd.DataFrame
) -> dict[str, np.ndarray[Any, Any]]:
    lookup: dict[tuple[str, str, pd.Timestamp], float] = {}
    for hospital, department, interval, arrivals in history[
        ["hospital_id", "department_id", "interval_start", "arrivals"]
    ].itertuples(index=False, name=None):
        lookup[(str(hospital), str(department), pd.Timestamp(cast(Any, interval)))] = float(
            cast(Any, arrivals)
        )
    seasonal: dict[tuple[str, str, int, int], float] = {}
    seasonal_values = (
        history.assign(
            weekday=history["interval_start"].dt.weekday,
            hour=history["interval_start"].dt.hour,
        )
        .groupby(["hospital_id", "department_id", "weekday", "hour"])["arrivals"]
        .mean()
    )
    for raw_key, value in seasonal_values.items():
        hospital, department, weekday, hour = cast(tuple[Any, Any, Any, Any], raw_key)
        seasonal[(str(hospital), str(department), int(weekday), int(hour))] = float(value)
    previous_day: list[float] = []
    previous_week: list[float] = []
    seasonal_mean: list[float] = []
    for hospital, department, interval in future[
        ["hospital_id", "department_id", "interval_start"]
    ].itertuples(index=False, name=None):
        timestamp = pd.Timestamp(cast(Any, interval))
        group = (str(hospital), str(department))
        fallback = seasonal[(*group, timestamp.weekday(), timestamp.hour)]
        previous_day.append(lookup.get((*group, timestamp - pd.Timedelta(days=1)), fallback))
        previous_week.append(lookup.get((*group, timestamp - pd.Timedelta(days=7)), fallback))
        seasonal_mean.append(fallback)
    return {
        "same_hour_previous_day": np.asarray(previous_day),
        "same_hour_previous_week": np.asarray(previous_week),
        "seasonal_mean": np.asarray(seasonal_mean),
    }


def rolling_backtest(
    target: pd.DataFrame, config: ArrivalTrainingConfig
) -> dict[str, dict[str, Any]]:
    timestamps = pd.Index(target["interval_start"].drop_duplicates().sort_values())
    required = 168 + 24 * config.backtest_windows
    if len(timestamps) < required:
        raise TrainingDataError(f"Rolling backtest requires at least {required} hourly intervals")
    collected: dict[int, dict[str, list[tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]]]] = {
        horizon: {} for horizon in SUPPORTED_HORIZONS
    }
    coverage: dict[int, list[bool]] = {horizon: [] for horizon in SUPPORTED_HORIZONS}
    for window in range(config.backtest_windows, 0, -1):
        cutoff = timestamps[-(24 * window + 1)]
        history = target.loc[target["interval_start"] <= cutoff]
        future = target.loc[
            (target["interval_start"] > cutoff)
            & (target["interval_start"] <= cutoff + pd.Timedelta(hours=24))
        ]
        artifact = _fit_artifact(history, config.model_version, config.random_seed)
        model_forecast = _department_forecast(artifact, 24)
        evaluated = future.merge(
            model_forecast,
            on=["hospital_id", "department_id", "interval_start"],
            validate="one_to_one",
        )
        baselines = _baseline_predictions(history, evaluated)
        baselines["hist_gradient_boosting"] = evaluated["predicted_arrivals"].to_numpy()
        for horizon in SUPPORTED_HORIZONS:
            horizon_end = cutoff + pd.Timedelta(hours=horizon)
            mask = evaluated["interval_start"] <= horizon_end
            actual = evaluated.loc[mask, "arrivals"].to_numpy(dtype=float)
            for name, prediction in baselines.items():
                collected[horizon].setdefault(name, []).append((actual, prediction[mask]))
            coverage[horizon].extend(
                (
                    (actual >= evaluated.loc[mask, "lower_arrivals"].to_numpy())
                    & (actual <= evaluated.loc[mask, "upper_arrivals"].to_numpy())
                ).tolist()
            )
    report: dict[str, dict[str, Any]] = {}
    for horizon, models in collected.items():
        model_metrics: dict[str, dict[str, float]] = {}
        for name, pairs in models.items():
            actual = np.concatenate([pair[0] for pair in pairs])
            predicted = np.concatenate([pair[1] for pair in pairs])
            metrics = _metrics(actual, predicted)
            peak_mask = actual >= np.quantile(actual, 0.9)
            metrics["peak_mae"] = float(np.abs(predicted[peak_mask] - actual[peak_mask]).mean())
            model_metrics[name] = metrics
        report[str(horizon)] = {
            "models": model_metrics,
            "interval_coverage": float(np.mean(coverage[horizon])),
        }
    return report


def train_arrival_forecaster(
    target: pd.DataFrame, config: ArrivalTrainingConfig, output_dir: Path
) -> ArrivalTrainingResult:
    backtest = rolling_backtest(target, config)
    artifact = _fit_artifact(target, config.model_version, config.random_seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = output_dir / "arrival_forecast.joblib"
    joblib.dump(artifact, artifact_path)
    sample = forecast_arrivals(artifact, 24)
    report: dict[str, Any] = {
        "model_version": config.model_version,
        "forecast_schema_version": FORECAST_SCHEMA_VERSION,
        "horizons": list(SUPPORTED_HORIZONS),
        "rolling_backtest": backtest,
        "reconciliation": "bottom_up",
        "interval_coverage_24h": backtest["24"]["interval_coverage"],
        "forecast_rows": len(sample.points),
        "limitations": [
            "Requires at least eight days of complete hourly history.",
            "Summed department intervals ignore cross-department error correlation.",
        ],
    }
    report_path = output_dir / "arrival_evaluation.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    run = log_training_run(
        config,
        model_kind=ModelKind.ARRIVALS,
        output_dir=output_dir,
        artifact_path=artifact_path,
        report_path=report_path,
        parameters={
            "model_version": config.model_version,
            "forecast_schema_version": FORECAST_SCHEMA_VERSION,
            "random_seed": config.random_seed,
        },
        metrics=flatten_metrics(report),
    )
    report["mlflow"] = run.__dict__ if run else {"tracking_enabled": False}
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return ArrivalTrainingResult(artifact_path, report_path, report, run.run_id if run else None)


def train_arrival_from_dataset(
    dataset: Dataset, config: ArrivalTrainingConfig, output_dir: Path
) -> ArrivalTrainingResult:
    target = build_arrival_target(dataset["appointments"], dataset["encounters"])
    return train_arrival_forecaster(target, config, output_dir)


def load_arrival_artifact(path: Path) -> ArrivalArtifact:
    artifact = joblib.load(path)
    if not isinstance(artifact, ArrivalArtifact):
        raise ModelArtifactError(f"Unexpected artifact type: {type(artifact).__name__}")
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bronze", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("models/arrivals"))
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--tracking-uri", default=os.environ.get("MLFLOW_TRACKING_URI"))
    parser.add_argument("--skip-tracking", action="store_true")
    args = parser.parse_args()
    config = ArrivalTrainingConfig(
        model_version=args.model_version,
        random_seed=args.seed,
        mlflow_tracking_uri=args.tracking_uri,
        track_experiment=not args.skip_tracking,
    )
    train_arrival_from_dataset(load_bronze(args.bronze), config, args.output)


if __name__ == "__main__":
    main()
