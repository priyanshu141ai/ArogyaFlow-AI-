from datetime import datetime
from importlib import resources
from pathlib import Path
from typing import Any, cast

import joblib  # type: ignore[import-untyped]
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from arogyaflow.api import app
from arogyaflow.arrival_forecasting import ArrivalArtifact
from arogyaflow.bed_occupancy import OccupancyArtifact
from arogyaflow.config import get_settings
from arogyaflow.database import PredictionRecord, PredictionType
from arogyaflow.no_show import FEATURE_COLUMNS as NO_SHOW_COLUMNS
from arogyaflow.no_show import NoShowArtifact
from arogyaflow.no_show import save_artifact as save_no_show_artifact
from arogyaflow.serving import PredictionApplication
from arogyaflow.wait_time import FEATURE_COLUMNS as WAIT_COLUMNS
from arogyaflow.wait_time import WaitTimeArtifact
from arogyaflow.wait_time import save_artifact as save_wait_artifact


class StubPreprocessor:
    def transform(self, frame: pd.DataFrame) -> np.ndarray[Any, Any]:
        return np.zeros((len(frame), 1))


class StubRegressor:
    def __init__(self, value: float) -> None:
        self.value = value

    def predict(self, values: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        return np.full(len(values), self.value)


class StubClassifier:
    def predict_proba(self, values: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        probability = 0.7
        return np.tile([1 - probability, probability], (len(values), 1))


class MemoryStore:
    def __init__(self) -> None:
        self.records: list[PredictionRecord] = []

    def insert_prediction(
        self,
        *,
        request_id: str,
        prediction_type: PredictionType,
        model_version: str,
        schema_version: str,
        request_payload: dict[str, Any],
        response_payload: dict[str, Any],
        created_at: datetime,
    ) -> PredictionRecord:
        record = PredictionRecord(
            id=len(self.records) + 1,
            request_id=request_id,
            prediction_type=prediction_type,
            model_version=model_version,
            schema_version=schema_version,
            response_payload=response_payload,
            created_at=created_at,
        )
        self.records.append(record)
        return record

    def recent_predictions(self, limit: int) -> list[PredictionRecord]:
        return list(reversed(self.records[-limit:]))


def _write_artifacts(root: Path) -> dict[str, Path]:
    preprocessor = cast(Any, StubPreprocessor())
    wait_path = root / "wait.joblib"
    save_wait_artifact(
        WaitTimeArtifact(
            model_version="wait-api-v1",
            schema_version="1.0",
            feature_columns=WAIT_COLUMNS,
            preprocessor=preprocessor,
            point_model=StubRegressor(20),
            lower_model=StubRegressor(10),
            upper_model=StubRegressor(30),
        ),
        wait_path,
    )
    no_show_path = root / "no-show.joblib"
    save_no_show_artifact(
        NoShowArtifact(
            model_version="no-show-api-v1",
            schema_version="1.0",
            feature_columns=NO_SHOW_COLUMNS,
            preprocessor=preprocessor,
            classifier=StubClassifier(),
            reminder_threshold=0.5,
        ),
        no_show_path,
    )
    intervals = pd.date_range("2026-01-01", periods=24 * 8, freq="h", tz="UTC")
    arrival_path = root / "arrivals.joblib"
    joblib.dump(
        ArrivalArtifact(
            model_version="arrival-api-v1",
            schema_version="1.0",
            preprocessor=preprocessor,
            point_model=StubRegressor(5),
            lower_model=StubRegressor(3),
            upper_model=StubRegressor(7),
            history=pd.DataFrame(
                {
                    "hospital_id": "hospital_demo",
                    "department_id": "dep_a",
                    "interval_start": intervals,
                    "arrivals": 4,
                }
            ),
        ),
        arrival_path,
    )
    occupancy_path = root / "occupancy.joblib"
    joblib.dump(
        OccupancyArtifact(
            model_version="occupancy-api-v1",
            schema_version="1.0",
            alert_threshold=0.8,
            preprocessor=preprocessor,
            point_model=StubRegressor(8),
            lower_model=StubRegressor(6),
            upper_model=StubRegressor(10),
            history=pd.DataFrame(
                {
                    "ward_id": "ward_a",
                    "interval_start": intervals,
                    "occupied_beds": 7,
                    "staffed_capacity": 10,
                    "expected_discharges": 0,
                    "occupancy_ratio": 0.7,
                }
            ),
        ),
        occupancy_path,
    )
    return {
        "WAIT": wait_path,
        "ARRIVAL": arrival_path,
        "NO_SHOW": no_show_path,
        "OCCUPANCY": occupancy_path,
    }


def test_prediction_endpoints_and_persistence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name, path in _write_artifacts(tmp_path).items():
        monkeypatch.setenv(f"AROGYAFLOW_{name}_MODEL_PATH", str(path))
    monkeypatch.delenv("AROGYAFLOW_DATABASE_URL", raising=False)
    get_settings.cache_clear()
    store = MemoryStore()
    with TestClient(app) as client:
        app.state.predictions = PredictionApplication(app.state.settings, store)
        wait = client.post(
            "/v1/predictions/wait-time",
            headers={"X-Request-ID": "req-wait"},
            json={
                "department_id": "dep_a",
                "appointment_type": "new",
                "priority_type": "routine",
                "reminder_sent": True,
                "weekday": 1,
                "hour": 10,
                "queue_length": 4,
                "available_doctors": 2,
                "available_rooms": 3,
            },
        )
        arrivals = client.post("/v1/forecasts/arrivals", json={"horizon_hours": 6})
        no_show = client.post(
            "/v1/predictions/no-show",
            json={
                "department_id": "dep_a",
                "appointment_type": "follow_up",
                "priority_type": "routine",
                "booking_lead_hours": 48,
                "scheduled_weekday": 2,
                "scheduled_hour": 11,
                "historical_appointments": 3,
                "historical_no_shows": 1,
                "historical_late_arrivals": 1,
                "historical_reminders": 2,
            },
        )
        occupancy = client.post("/v1/forecasts/occupancy", json={"horizon_hours": 6})
        recent = client.get("/v1/predictions/recent?limit=10")
    get_settings.cache_clear()

    assert wait.status_code == arrivals.status_code == no_show.status_code == 200
    assert occupancy.status_code == recent.status_code == 200
    assert wait.json()["predicted_wait_minutes"] == 20
    assert wait.json()["request_id"] == "req-wait"
    assert arrivals.json()["forecast"]["horizon_hours"] == 6
    assert no_show.json()["automatic_cancellation"] is False
    assert occupancy.json()["forecast"]["points"][0]["capacity_alert"] is True
    assert all(response.json()["persisted"] for response in (wait, arrivals, no_show, occupancy))
    assert len(recent.json()["records"]) == 4


def test_prediction_migration_is_packaged() -> None:
    migration = resources.files("arogyaflow.migrations").joinpath("0001_prediction_records.sql")
    sql = migration.read_text(encoding="utf-8")
    assert "created_at timestamptz" in sql
    assert "prediction_records_type_created_at_idx" in sql
