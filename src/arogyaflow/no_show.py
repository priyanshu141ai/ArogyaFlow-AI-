from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib  # type: ignore[import-untyped]
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer  # type: ignore[import-untyped]
from sklearn.metrics import (  # type: ignore[import-untyped]
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # type: ignore[import-untyped]

from arogyaflow.data.generation import Dataset
from arogyaflow.exceptions import (
    DataContractError,
    FeatureSchemaMismatchError,
    ModelArtifactError,
    TrainingDataError,
)

NO_SHOW_SCHEMA_VERSION = "1.0"
CATEGORICAL_FEATURES = ("department_id", "appointment_type", "priority_type")
NUMERIC_FEATURES = (
    "booking_lead_hours",
    "scheduled_weekday",
    "scheduled_hour",
    "historical_appointments",
    "historical_no_shows",
    "historical_late_arrivals",
    "historical_reminders",
    "historical_no_show_rate",
)
FEATURE_COLUMNS = (*CATEGORICAL_FEATURES, *NUMERIC_FEATURES)


@dataclass
class NoShowArtifact:
    model_version: str
    schema_version: str
    feature_columns: tuple[str, ...]
    preprocessor: ColumnTransformer
    classifier: Any
    reminder_threshold: float


def build_no_show_features(
    dataset: Dataset, *, grace_minutes: int, late_minutes: int
) -> pd.DataFrame:
    if grace_minutes < 0 or late_minutes < 0:
        raise ValueError("grace_minutes and late_minutes must be non-negative")
    required = {"appointments", "encounters"}
    if missing := required - dataset.keys():
        raise DataContractError(f"Missing no-show tables: {sorted(missing)}")
    appointments = dataset["appointments"].copy()
    encounters = dataset["encounters"][["appointment_id", "queue_entered_at"]]
    appointments = appointments.merge(
        encounters, on="appointment_id", how="left", validate="one_to_one"
    )
    appointments["was_late"] = (
        appointments["queue_entered_at"]
        > appointments["scheduled_at"] + pd.Timedelta(minutes=late_minutes)
    ).fillna(False)
    appointments["history_available_at"] = appointments["scheduled_at"] + pd.Timedelta(
        minutes=grace_minutes
    )
    appointments["no_show"] = (appointments["status"] == "no_show").astype(int)

    history_columns = [
        "historical_appointments",
        "historical_no_shows",
        "historical_late_arrivals",
        "historical_reminders",
    ]
    for column in history_columns:
        appointments[column] = 0
    for _, patient in appointments.groupby("patient_key", sort=False):
        events = patient.sort_values("history_available_at")
        event_times = list(events["history_available_at"])
        no_shows = events["no_show"].cumsum().tolist()
        late = events["was_late"].astype(int).cumsum().tolist()
        reminders = events["reminder_sent"].astype(int).cumsum().tolist()
        for index, decision_time in patient["booking_created_at"].items():
            count = bisect_left(event_times, decision_time)
            appointments.at[index, "historical_appointments"] = count
            if count:
                appointments.at[index, "historical_no_shows"] = no_shows[count - 1]
                appointments.at[index, "historical_late_arrivals"] = late[count - 1]
                appointments.at[index, "historical_reminders"] = reminders[count - 1]

    appointments["historical_no_show_rate"] = np.where(
        appointments["historical_appointments"] > 0,
        appointments["historical_no_shows"] / appointments["historical_appointments"],
        0.0,
    )
    appointments["booking_lead_hours"] = (
        appointments["scheduled_at"] - appointments["booking_created_at"]
    ).dt.total_seconds() / 3600
    appointments["scheduled_weekday"] = appointments["scheduled_at"].dt.weekday
    appointments["scheduled_hour"] = appointments["scheduled_at"].dt.hour
    columns = [
        "appointment_id",
        "patient_key",
        "booking_created_at",
        *FEATURE_COLUMNS,
        "no_show",
    ]
    features = appointments[columns]
    if features.isna().any().any():
        raise TrainingDataError("No-show features contain missing values")
    return features


def make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                list(CATEGORICAL_FEATURES),
            ),
            ("numeric", StandardScaler(), list(NUMERIC_FEATURES)),
        ],
        verbose_feature_names_out=False,
    )


def expected_calibration_error(
    actual: pd.Series, probabilities: np.ndarray[Any, Any], bins: int = 10
) -> float:
    boundaries = np.linspace(0, 1, bins + 1)
    values = actual.to_numpy(dtype=float)
    error = 0.0
    for lower, upper in zip(boundaries[:-1], boundaries[1:], strict=True):
        mask = (probabilities >= lower) & (
            probabilities <= upper if upper == 1 else probabilities < upper
        )
        if mask.any():
            error += float(mask.mean()) * abs(
                float(values[mask].mean()) - float(probabilities[mask].mean())
            )
    return error


def classification_metrics(
    actual: pd.Series, probabilities: np.ndarray[Any, Any], threshold: float
) -> dict[str, float]:
    labels = (probabilities >= threshold).astype(int)
    return {
        "pr_auc": float(average_precision_score(actual, probabilities)),
        "roc_auc": float(roc_auc_score(actual, probabilities)),
        "brier_score": float(brier_score_loss(actual, probabilities)),
        "expected_calibration_error": expected_calibration_error(actual, probabilities),
        "precision": float(precision_score(actual, labels, zero_division=0)),
        "recall": float(recall_score(actual, labels, zero_division=0)),
        "f1": float(f1_score(actual, labels, zero_division=0)),
        "selection_rate": float(labels.mean()),
    }


def predict_no_show(artifact: NoShowArtifact, features: pd.DataFrame) -> pd.DataFrame:
    if missing := set(artifact.feature_columns) - set(features.columns):
        raise FeatureSchemaMismatchError(f"Missing inference features: {sorted(missing)}")
    model_input = features[list(artifact.feature_columns)]
    if model_input.isna().any().any():
        raise FeatureSchemaMismatchError("Inference features contain missing values")
    transformed = artifact.preprocessor.transform(model_input)
    probabilities = artifact.classifier.predict_proba(transformed)[:, 1]
    return pd.DataFrame(
        {
            "no_show_probability": probabilities,
            "reminder_priority": probabilities >= artifact.reminder_threshold,
        },
        index=features.index,
    )


def save_artifact(artifact: NoShowArtifact, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path)


def load_artifact(path: Path) -> NoShowArtifact:
    artifact = joblib.load(path)
    if not isinstance(artifact, NoShowArtifact):
        raise ModelArtifactError(f"Unexpected artifact type: {type(artifact).__name__}")
    return artifact
