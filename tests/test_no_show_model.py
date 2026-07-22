from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from arogyaflow.data.generation import ScenarioConfig, ScenarioName, generate_dataset
from arogyaflow.data.validation import validate_dataset
from arogyaflow.exceptions import FeatureSchemaMismatchError
from arogyaflow.no_show import (
    FEATURE_COLUMNS,
    build_no_show_features,
    load_artifact,
    predict_no_show,
)
from arogyaflow.train_no_show import NoShowTrainingConfig, train_no_show_model


def test_patient_history_is_available_point_in_time_only() -> None:
    timestamp = pd.Timestamp
    appointments = pd.DataFrame(
        [
            {
                "appointment_id": "a1",
                "patient_key": "p1",
                "booking_created_at": timestamp("2026-01-01T08:00Z"),
                "scheduled_at": timestamp("2026-01-02T10:00Z"),
                "department_id": "dep",
                "appointment_type": "new",
                "priority_type": "routine",
                "reminder_sent": False,
                "status": "no_show",
            },
            {
                "appointment_id": "a2",
                "patient_key": "p1",
                "booking_created_at": timestamp("2026-01-01T12:00Z"),
                "scheduled_at": timestamp("2026-01-03T10:00Z"),
                "department_id": "dep",
                "appointment_type": "follow_up",
                "priority_type": "routine",
                "reminder_sent": True,
                "status": "completed",
            },
            {
                "appointment_id": "a3",
                "patient_key": "p1",
                "booking_created_at": timestamp("2026-01-02T12:00Z"),
                "scheduled_at": timestamp("2026-01-04T10:00Z"),
                "department_id": "dep",
                "appointment_type": "follow_up",
                "priority_type": "routine",
                "reminder_sent": True,
                "status": "completed",
            },
        ]
    )
    encounters = pd.DataFrame(
        {
            "appointment_id": ["a2", "a3"],
            "queue_entered_at": [timestamp("2026-01-03T10:00Z"), timestamp("2026-01-04T10:00Z")],
        }
    )
    features = build_no_show_features(
        {"appointments": appointments, "encounters": encounters},
        grace_minutes=60,
        late_minutes=15,
    ).set_index("appointment_id")
    assert features.loc["a2", "historical_appointments"] == 0
    assert features.loc["a3", "historical_no_shows"] == 1


def test_no_show_training_calibration_and_policy(tmp_path: Path) -> None:
    dataset = validate_dataset(
        generate_dataset(
            ScenarioConfig(
                scenario=ScenarioName.NORMAL_WEEK,
                seed=31,
                start=datetime(2026, 1, 5, tzinfo=UTC),
                days=5,
            )
        )
    ).clean
    features = build_no_show_features(dataset, grace_minutes=60, late_minutes=15)
    assert "reminder_sent" not in features.columns
    result = train_no_show_model(
        dataset,
        NoShowTrainingConfig(
            model_version="no-show-test-v1",
            random_seed=13,
            reminder_capacity_fraction=0.2,
            reminder_effectiveness=0.25,
            maximum_ece=0.35,
        ),
        tmp_path,
    )
    assert result.mlflow_run_id
    assert result.report["mlflow"]["registered_model_version"]
    assert result.report["calibration_acceptable"] is True
    assert result.report["policy_simulation"]["automatic_cancellation"] is False
    artifact = load_artifact(result.artifact_path)
    predictions = predict_no_show(artifact, features.head(4))
    assert predictions["no_show_probability"].between(0, 1).all()

    with pytest.raises(FeatureSchemaMismatchError):
        predict_no_show(artifact, features.drop(columns=FEATURE_COLUMNS[0]))
