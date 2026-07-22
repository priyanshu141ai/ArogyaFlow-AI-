from pathlib import Path

import pandas as pd
import pytest

from arogyaflow.arrival_forecasting import (
    ArrivalTrainingConfig,
    forecast_arrivals,
    load_arrival_artifact,
    train_arrival_forecaster,
)
from arogyaflow.exceptions import ForecastHorizonError


def _arrival_target() -> pd.DataFrame:
    intervals = pd.date_range("2026-01-01", periods=24 * 14, freq="h", tz="UTC")
    rows = []
    for department, offset in (("dep_a", 0), ("dep_b", 2)):
        for timestamp in intervals:
            arrivals = (
                offset + 2 + (4 if 9 <= timestamp.hour <= 17 else 0) + timestamp.weekday() % 2
            )
            rows.append(
                {
                    "hospital_id": "hospital_demo",
                    "department_id": department,
                    "interval_start": timestamp,
                    "arrivals": arrivals,
                }
            )
    return pd.DataFrame(rows)


def test_arrival_training_backtest_and_reconciliation(tmp_path: Path) -> None:
    result = train_arrival_forecaster(
        _arrival_target(),
        ArrivalTrainingConfig(model_version="arrival-test-v1", random_seed=5, backtest_windows=2),
        tmp_path,
    )
    assert result.mlflow_run_id
    assert result.report["mlflow"]["registered_model_version"]
    assert set(result.report["rolling_backtest"]) == {"6", "12", "24"}
    artifact = load_arrival_artifact(result.artifact_path)
    forecast = forecast_arrivals(artifact, 24)
    frame = pd.DataFrame(point.model_dump() for point in forecast.points)
    departments = (
        frame.loc[frame["level"] == "department"]
        .groupby("interval_start")["predicted_arrivals"]
        .sum()
    )
    hospital = frame.loc[frame["level"] == "hospital"].set_index("interval_start")[
        "predicted_arrivals"
    ]
    pd.testing.assert_series_equal(departments, hospital, check_names=False)
    assert (frame["lower_arrivals"] <= frame["predicted_arrivals"]).all()
    assert (frame["predicted_arrivals"] <= frame["upper_arrivals"]).all()

    with pytest.raises(ForecastHorizonError):
        forecast_arrivals(artifact, 5)
