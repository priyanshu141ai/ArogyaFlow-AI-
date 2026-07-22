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

from arogyaflow.baseline_pipeline import load_bronze
from arogyaflow.data.generation import Dataset
from arogyaflow.exceptions import (
    DataContractError,
    DataQualityError,
    ForecastHorizonError,
    ModelArtifactError,
    TrainingDataError,
)
from arogyaflow.time import utc_now
from arogyaflow.tracking import ModelKind, TrackingOptions, flatten_metrics, log_training_run

OCCUPANCY_SCHEMA_VERSION = "1.0"
SUPPORTED_HORIZONS = (6, 12, 24)
FEATURE_COLUMNS = (
    "ward_id",
    "hour",
    "weekday",
    "staffed_capacity",
    "lag_1",
    "lag_24",
    "lag_168",
    "rolling_24",
    "expected_discharges",
)


class OccupancyTrainingConfig(TrackingOptions):
    model_config = ConfigDict(frozen=True)

    model_version: str = Field(min_length=1)
    random_seed: int
    alert_threshold: float = Field(gt=0, le=1)
    backtest_windows: int = Field(default=3, ge=2, le=8)
    experiment_name: str = "arogyaflow-occupancy"
    registered_model_name: str = "arogyaflow-occupancy"


class OccupancyPoint(BaseModel):
    interval_start: datetime
    ward_id: str
    staffed_capacity: int
    predicted_occupied_beds: float
    lower_occupied_beds: float
    upper_occupied_beds: float
    predicted_occupancy_ratio: float
    capacity_alert: bool


class OccupancyForecast(BaseModel):
    model_version: str
    schema_version: str
    generated_at: datetime
    horizon_hours: Literal[6, 12, 24]
    alert_threshold: float
    points: list[OccupancyPoint]


@dataclass
class OccupancyArtifact:
    model_version: str
    schema_version: str
    alert_threshold: float
    preprocessor: ColumnTransformer
    point_model: Any
    lower_model: Any
    upper_model: Any
    history: pd.DataFrame


@dataclass(frozen=True)
class OccupancyTrainingResult:
    artifact_path: Path
    report_path: Path
    report: dict[str, Any]
    mlflow_run_id: str | None


def build_occupancy_target(dataset: Dataset) -> pd.DataFrame:
    required = {"beds", "bed_events", "admissions"}
    if missing := required - dataset.keys():
        raise DataContractError(f"Missing occupancy tables: {sorted(missing)}")
    beds = dataset["beds"]
    events = dataset["bed_events"].merge(beds, on="bed_id", validate="many_to_one")
    if events.empty:
        raise DataQualityError("No bed events available")
    deltas: list[dict[str, object]] = []
    for bed_id, bed_events in events.groupby("bed_id", sort=False):
        occupied = False
        for event in bed_events.sort_values("event_time").itertuples(index=False):
            if event.event_type == "occupied":
                if occupied:
                    raise DataQualityError(f"Bed {bed_id} has overlapping occupancy")
                occupied = True
                delta = 1
            else:
                if not occupied:
                    raise DataQualityError(f"Bed {bed_id} released while unoccupied")
                occupied = False
                delta = -1
            deltas.append(
                {
                    "ward_id": event.ward_id,
                    "interval_start": pd.Timestamp(cast(Any, event.event_time)).floor("h"),
                    "delta": delta,
                }
            )
    capacity = beds.groupby("ward_id").size().to_dict()
    delta_frame = pd.DataFrame(deltas)
    admissions = dataset["admissions"].copy()
    admissions["expected_interval"] = admissions["expected_discharge_at"].dt.floor("h")
    expected = admissions.groupby(["ward_id", "expected_interval"]).size().to_dict()
    output: list[pd.DataFrame] = []
    for ward_id, ward_events in delta_frame.groupby("ward_id", sort=True):
        timeline = pd.date_range(
            ward_events["interval_start"].min(),
            ward_events["interval_start"].max(),
            freq="h",
        )
        hourly_delta = (
            ward_events.groupby("interval_start")["delta"].sum().reindex(timeline, fill_value=0)
        )
        occupancy_series = hourly_delta.cumsum()
        ward_capacity = int(capacity[ward_id])
        ward = pd.DataFrame(
            {
                "ward_id": str(ward_id),
                "interval_start": timeline,
                "occupied_beds": occupancy_series.to_numpy(dtype=int),
                "staffed_capacity": ward_capacity,
                "expected_discharges": [
                    int(expected.get((ward_id, timestamp), 0)) for timestamp in timeline
                ],
            }
        )
        output.append(ward)
    target = pd.concat(output, ignore_index=True)
    if (target["occupied_beds"] < 0).any() or (
        target["occupied_beds"] > target["staffed_capacity"]
    ).any():
        raise DataQualityError("Occupied beds must remain within staffed capacity")
    target["occupancy_ratio"] = target["occupied_beds"] / target["staffed_capacity"]
    return target


def build_occupancy_features(target: pd.DataFrame) -> pd.DataFrame:
    required = {
        "ward_id",
        "interval_start",
        "occupied_beds",
        "staffed_capacity",
        "expected_discharges",
    }
    if missing := required - set(target.columns):
        raise TrainingDataError(f"Missing occupancy target columns: {sorted(missing)}")
    frames: list[pd.DataFrame] = []
    for _, group in target.groupby("ward_id", sort=True):
        ordered = group.sort_values("interval_start").copy()
        ordered["hour"] = ordered["interval_start"].dt.hour
        ordered["weekday"] = ordered["interval_start"].dt.weekday
        ordered["lag_1"] = ordered["occupied_beds"].shift(1)
        ordered["lag_24"] = ordered["occupied_beds"].shift(24)
        ordered["lag_168"] = ordered["occupied_beds"].shift(168)
        ordered["rolling_24"] = ordered["occupied_beds"].shift(1).rolling(24).mean()
        frames.append(ordered)
    features = pd.concat(frames, ignore_index=True).dropna().reset_index(drop=True)
    if features.empty:
        raise TrainingDataError("At least eight days of hourly occupancy history are required")
    return features


def _preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            (
                "ward",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ["ward_id"],
            ),
            (
                "numeric",
                StandardScaler(),
                [column for column in FEATURE_COLUMNS if column != "ward_id"],
            ),
        ],
        verbose_feature_names_out=False,
    )


def _fit_artifact(target: pd.DataFrame, config: OccupancyTrainingConfig) -> OccupancyArtifact:
    features = build_occupancy_features(target)
    processor = _preprocessor()
    transformed = processor.fit_transform(features[list(FEATURE_COLUMNS)])
    point = HistGradientBoostingRegressor(
        max_iter=100, max_leaf_nodes=15, random_state=config.random_seed
    ).fit(transformed, features["occupied_beds"])
    lower = HistGradientBoostingRegressor(
        loss="quantile", quantile=0.1, max_iter=100, random_state=config.random_seed
    ).fit(transformed, features["occupied_beds"])
    upper = HistGradientBoostingRegressor(
        loss="quantile", quantile=0.9, max_iter=100, random_state=config.random_seed
    ).fit(transformed, features["occupied_beds"])
    return OccupancyArtifact(
        model_version=config.model_version,
        schema_version=OCCUPANCY_SCHEMA_VERSION,
        alert_threshold=config.alert_threshold,
        preprocessor=processor,
        point_model=point,
        lower_model=lower,
        upper_model=upper,
        history=target.copy(),
    )


def _ward_forecast(artifact: OccupancyArtifact, horizon: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for ward_id, group in artifact.history.groupby("ward_id", sort=True):
        values: dict[pd.Timestamp, float] = {}
        for interval, occupied in group[["interval_start", "occupied_beds"]].itertuples(
            index=False, name=None
        ):
            values[pd.Timestamp(cast(Any, interval))] = float(cast(Any, occupied))
        capacity = int(group["staffed_capacity"].iloc[-1])
        last_time = max(values)
        for step in range(1, horizon + 1):
            timestamp = last_time + pd.Timedelta(hours=step)
            recent = [values[timestamp - pd.Timedelta(hours=offset)] for offset in range(1, 25)]
            feature = pd.DataFrame(
                [
                    {
                        "ward_id": str(ward_id),
                        "hour": timestamp.hour,
                        "weekday": timestamp.weekday(),
                        "staffed_capacity": capacity,
                        "lag_1": values[timestamp - pd.Timedelta(hours=1)],
                        "lag_24": values[timestamp - pd.Timedelta(hours=24)],
                        "lag_168": values[timestamp - pd.Timedelta(hours=168)],
                        "rolling_24": sum(recent) / 24,
                        "expected_discharges": 0,
                    }
                ]
            )
            transformed = artifact.preprocessor.transform(feature[list(FEATURE_COLUMNS)])
            point = float(np.clip(artifact.point_model.predict(transformed)[0], 0, capacity))
            lower = float(
                np.clip(min(point, artifact.lower_model.predict(transformed)[0]), 0, capacity)
            )
            upper = float(
                np.clip(max(point, artifact.upper_model.predict(transformed)[0]), 0, capacity)
            )
            values[timestamp] = point
            rows.append(
                {
                    "interval_start": timestamp,
                    "ward_id": str(ward_id),
                    "staffed_capacity": capacity,
                    "predicted_occupied_beds": point,
                    "lower_occupied_beds": lower,
                    "upper_occupied_beds": upper,
                }
            )
    return pd.DataFrame(rows)


def forecast_occupancy(artifact: OccupancyArtifact, horizon_hours: int) -> OccupancyForecast:
    if horizon_hours not in SUPPORTED_HORIZONS:
        raise ForecastHorizonError(f"Supported horizons: {SUPPORTED_HORIZONS}")
    frame = _ward_forecast(artifact, horizon_hours)
    frame["predicted_occupancy_ratio"] = (
        frame["predicted_occupied_beds"] / frame["staffed_capacity"]
    )
    frame["capacity_alert"] = frame["predicted_occupancy_ratio"] >= artifact.alert_threshold
    records = cast(list[dict[str, Any]], frame.to_dict(orient="records"))
    return OccupancyForecast(
        model_version=artifact.model_version,
        schema_version=artifact.schema_version,
        generated_at=utc_now(),
        horizon_hours=cast(Literal[6, 12, 24], horizon_hours),
        alert_threshold=artifact.alert_threshold,
        points=[OccupancyPoint(**record) for record in records],
    )


def _metrics(actual: np.ndarray[Any, Any], predicted: np.ndarray[Any, Any]) -> dict[str, float]:
    error = predicted - actual
    absolute = np.abs(error)
    return {
        "mae_beds": float(absolute.mean()),
        "rmse_beds": math.sqrt(float(np.mean(error**2))),
        "wape": float(absolute.sum()) / max(float(np.abs(actual).sum()), 1.0),
    }


def _alert_metrics(
    actual: np.ndarray[Any, Any],
    predicted: np.ndarray[Any, Any],
    capacity: np.ndarray[Any, Any],
    threshold: float,
) -> dict[str, float]:
    actual_alert = actual / capacity >= threshold
    predicted_alert = predicted / capacity >= threshold
    true_positive = int((actual_alert & predicted_alert).sum())
    false_positive = int((~actual_alert & predicted_alert).sum())
    false_negative = int((actual_alert & ~predicted_alert).sum())
    true_negative = int((~actual_alert & ~predicted_alert).sum())
    return {
        "alert_recall": true_positive / max(true_positive + false_negative, 1),
        "false_alert_rate": false_positive / max(false_positive + true_negative, 1),
    }


def rolling_backtest(
    target: pd.DataFrame, config: OccupancyTrainingConfig
) -> dict[str, dict[str, Any]]:
    timestamps = pd.Index(target["interval_start"].drop_duplicates().sort_values())
    required = 168 + 24 * config.backtest_windows
    if len(timestamps) < required:
        raise TrainingDataError(f"Rolling backtest requires at least {required} hourly intervals")
    collected: dict[int, list[pd.DataFrame]] = {horizon: [] for horizon in SUPPORTED_HORIZONS}
    for window in range(config.backtest_windows, 0, -1):
        cutoff = timestamps[-(24 * window + 1)]
        history = target.loc[target["interval_start"] <= cutoff]
        future = target.loc[
            (target["interval_start"] > cutoff)
            & (target["interval_start"] <= cutoff + pd.Timedelta(hours=24))
        ]
        forecast = _ward_forecast(_fit_artifact(history, config), 24)
        evaluated = future.merge(
            forecast, on=["ward_id", "interval_start", "staffed_capacity"], validate="one_to_one"
        )
        lookup = {
            (row.ward_id, pd.Timestamp(row.interval_start)): float(row.occupied_beds)
            for row in history.itertuples(index=False)
        }
        evaluated["previous_day"] = [
            lookup[(row.ward_id, pd.Timestamp(row.interval_start) - pd.Timedelta(days=1))]
            for row in evaluated.itertuples(index=False)
        ]
        evaluated["previous_week"] = [
            lookup[(row.ward_id, pd.Timestamp(row.interval_start) - pd.Timedelta(days=7))]
            for row in evaluated.itertuples(index=False)
        ]
        for horizon in SUPPORTED_HORIZONS:
            collected[horizon].append(
                evaluated.loc[evaluated["interval_start"] <= cutoff + pd.Timedelta(hours=horizon)]
            )
    report: dict[str, dict[str, Any]] = {}
    for horizon, frames in collected.items():
        evaluated = pd.concat(frames, ignore_index=True)
        actual = evaluated["occupied_beds"].to_numpy(dtype=float)
        point = evaluated["predicted_occupied_beds"].to_numpy(dtype=float)
        capacity = evaluated["staffed_capacity"].to_numpy(dtype=float)
        report[str(horizon)] = {
            "direct_model": _metrics(actual, point),
            "same_hour_previous_day": _metrics(
                actual, evaluated["previous_day"].to_numpy(dtype=float)
            ),
            "same_hour_previous_week": _metrics(
                actual, evaluated["previous_week"].to_numpy(dtype=float)
            ),
            "capacity_alerts": _alert_metrics(actual, point, capacity, config.alert_threshold),
            "interval_coverage": float(
                np.mean(
                    (actual >= evaluated["lower_occupied_beds"].to_numpy())
                    & (actual <= evaluated["upper_occupied_beds"].to_numpy())
                )
            ),
        }
    return report


def train_occupancy_model(
    target: pd.DataFrame, config: OccupancyTrainingConfig, output_dir: Path
) -> OccupancyTrainingResult:
    backtest = rolling_backtest(target, config)
    artifact = _fit_artifact(target, config)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = output_dir / "bed_occupancy.joblib"
    joblib.dump(artifact, artifact_path)
    report: dict[str, Any] = {
        "model_version": config.model_version,
        "forecast_schema_version": OCCUPANCY_SCHEMA_VERSION,
        "rolling_backtest": backtest,
        "occupancy_invariants_passed": bool(
            (target["occupied_beds"] >= 0).all()
            and (target["occupied_beds"] <= target["staffed_capacity"]).all()
        ),
        "expected_discharge_assumption": (
            "Expected discharge timestamps are known after admission; "
            "unknown future admissions are zero."
        ),
        "limitations": [
            "Active bed count is used as staffed capacity until ward staffing data exists.",
            "Alert quality depends on the configured capacity threshold.",
        ],
    }
    report_path = output_dir / "bed_occupancy_evaluation.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    run = log_training_run(
        config,
        model_kind=ModelKind.OCCUPANCY,
        output_dir=output_dir,
        artifact_path=artifact_path,
        report_path=report_path,
        parameters={
            "model_version": config.model_version,
            "forecast_schema_version": OCCUPANCY_SCHEMA_VERSION,
            "random_seed": config.random_seed,
            "alert_threshold": config.alert_threshold,
        },
        metrics=flatten_metrics(report),
    )
    report["mlflow"] = run.__dict__ if run else {"tracking_enabled": False}
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return OccupancyTrainingResult(artifact_path, report_path, report, run.run_id if run else None)


def load_occupancy_artifact(path: Path) -> OccupancyArtifact:
    artifact = joblib.load(path)
    if not isinstance(artifact, OccupancyArtifact):
        raise ModelArtifactError(f"Unexpected artifact type: {type(artifact).__name__}")
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bronze", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("models/occupancy"))
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--alert-threshold", required=True, type=float)
    parser.add_argument("--tracking-uri", default=os.environ.get("MLFLOW_TRACKING_URI"))
    parser.add_argument("--skip-tracking", action="store_true")
    args = parser.parse_args()
    config = OccupancyTrainingConfig(
        model_version=args.model_version,
        random_seed=args.seed,
        alert_threshold=args.alert_threshold,
        mlflow_tracking_uri=args.tracking_uri,
        track_experiment=not args.skip_tracking,
    )
    dataset = load_bronze(args.bronze)
    train_occupancy_model(build_occupancy_target(dataset), config, args.output)


if __name__ == "__main__":
    main()
