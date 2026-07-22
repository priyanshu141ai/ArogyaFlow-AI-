from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from arogyaflow.bed_occupancy import (
    OccupancyTrainingConfig,
    build_occupancy_target,
    forecast_occupancy,
    load_occupancy_artifact,
    train_occupancy_model,
)
from arogyaflow.data.generation import ScenarioConfig, ScenarioName, generate_dataset
from arogyaflow.data.validation import validate_dataset


def _occupancy_target() -> pd.DataFrame:
    intervals = pd.date_range("2026-01-01", periods=24 * 14, freq="h", tz="UTC")
    rows = []
    for ward, capacity, offset in (("ward_a", 20, 7), ("ward_b", 10, 3)):
        for timestamp in intervals:
            peak = 9 if 10 <= timestamp.hour <= 18 else 3
            occupied = min(capacity, offset + peak + timestamp.weekday() % 2)
            rows.append(
                {
                    "ward_id": ward,
                    "interval_start": timestamp,
                    "occupied_beds": occupied,
                    "staffed_capacity": capacity,
                    "expected_discharges": 1 if timestamp.hour == 11 else 0,
                    "occupancy_ratio": occupied / capacity,
                }
            )
    return pd.DataFrame(rows)


def test_generated_bed_events_preserve_capacity() -> None:
    dataset = validate_dataset(
        generate_dataset(
            ScenarioConfig(
                scenario=ScenarioName.BED_CLOSURE,
                seed=23,
                start=datetime(2026, 1, 5, tzinfo=UTC),
                days=4,
            )
        )
    ).clean
    target = build_occupancy_target(dataset)
    assert (target["occupied_beds"] <= target["staffed_capacity"]).all()


def test_occupancy_training_forecast_and_alerts(tmp_path: Path) -> None:
    result = train_occupancy_model(
        _occupancy_target(),
        OccupancyTrainingConfig(
            model_version="occupancy-test-v1",
            random_seed=17,
            alert_threshold=0.8,
            backtest_windows=2,
        ),
        tmp_path,
    )
    assert result.mlflow_run_id
    assert result.report["mlflow"]["registered_model_version"]
    assert result.report["occupancy_invariants_passed"] is True
    assert "alert_recall" in result.report["rolling_backtest"]["24"]["capacity_alerts"]
    artifact = load_occupancy_artifact(result.artifact_path)
    forecast = forecast_occupancy(artifact, 24)
    assert all(
        0 <= point.predicted_occupied_beds <= point.staffed_capacity for point in forecast.points
    )
