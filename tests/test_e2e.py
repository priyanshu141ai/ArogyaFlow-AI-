from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from arogyaflow.api import app
from arogyaflow.arrival_forecasting import ArrivalTrainingConfig, train_arrival_from_dataset
from arogyaflow.bed_occupancy import (
    OccupancyTrainingConfig,
    build_occupancy_target,
    train_occupancy_model,
)
from arogyaflow.config import get_settings
from arogyaflow.data.generation import ScenarioConfig, ScenarioName, generate_dataset
from arogyaflow.data.validation import validate_dataset
from arogyaflow.database import PredictionRecord, PredictionType
from arogyaflow.serving import PredictionApplication
from arogyaflow.train_no_show import NoShowTrainingConfig, train_no_show_model
from arogyaflow.train_wait_time import WaitTimeTrainingConfig, train_wait_time_model


class E2EStore:
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
        del request_payload
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


def test_generate_train_serve_and_persist(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dataset = validate_dataset(
        generate_dataset(
            ScenarioConfig(
                scenario=ScenarioName.NORMAL_WEEK,
                seed=31,
                start=datetime(2026, 1, 5, tzinfo=UTC),
                days=10,
            )
        )
    ).clean
    wait = train_wait_time_model(
        dataset,
        WaitTimeTrainingConfig(
            model_version="e2e-wait-v1",
            random_seed=11,
            shap_sample_size=10,
            track_experiment=False,
        ),
        tmp_path / "wait",
    )
    arrivals = train_arrival_from_dataset(
        dataset,
        ArrivalTrainingConfig(
            model_version="e2e-arrival-v1",
            random_seed=5,
            backtest_windows=2,
            track_experiment=False,
        ),
        tmp_path / "arrivals",
    )
    no_show = train_no_show_model(
        dataset,
        NoShowTrainingConfig(
            model_version="e2e-no-show-v1",
            random_seed=13,
            reminder_capacity_fraction=0.2,
            reminder_effectiveness=0.25,
            maximum_ece=0.35,
            track_experiment=False,
        ),
        tmp_path / "no-show",
    )
    occupancy = train_occupancy_model(
        build_occupancy_target(dataset),
        OccupancyTrainingConfig(
            model_version="e2e-occupancy-v1",
            random_seed=17,
            alert_threshold=0.8,
            backtest_windows=2,
            track_experiment=False,
        ),
        tmp_path / "occupancy",
    )
    paths = {
        "WAIT": wait.artifact_path,
        "ARRIVAL": arrivals.artifact_path,
        "NO_SHOW": no_show.artifact_path,
        "OCCUPANCY": occupancy.artifact_path,
    }
    for name, path in paths.items():
        monkeypatch.setenv(f"AROGYAFLOW_{name}_MODEL_PATH", str(path))
    monkeypatch.delenv("AROGYAFLOW_DATABASE_URL", raising=False)
    get_settings.cache_clear()
    store = E2EStore()
    with TestClient(app) as client:
        app.state.predictions = PredictionApplication(app.state.settings, store)
        responses = [
            client.post(
                "/v1/predictions/wait-time",
                json={
                    "department_id": "department_demo",
                    "appointment_type": "new",
                    "priority_type": "routine",
                    "reminder_sent": True,
                    "weekday": 1,
                    "hour": 10,
                    "queue_length": 4,
                    "available_doctors": 2,
                    "available_rooms": 3,
                },
            ),
            client.post("/v1/forecasts/arrivals", json={"horizon_hours": 6}),
            client.post(
                "/v1/predictions/no-show",
                json={
                    "department_id": "department_demo",
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
            ),
            client.post("/v1/forecasts/occupancy", json={"horizon_hours": 6}),
        ]
        recent = client.get("/v1/predictions/recent?limit=10")
    get_settings.cache_clear()
    assert all(response.status_code == 200 for response in responses)
    assert all(response.json()["persisted"] is True for response in responses)
    assert len(recent.json()["records"]) == 4
