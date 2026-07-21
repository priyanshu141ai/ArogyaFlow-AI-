import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib  # type: ignore[import-untyped]
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer  # type: ignore[import-untyped]
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # type: ignore[import-untyped]

from arogyaflow.analysis import build_waiting_time_target
from arogyaflow.data.generation import Dataset
from arogyaflow.exceptions import (
    DataContractError,
    FeatureSchemaMismatchError,
    ModelArtifactError,
    TrainingDataError,
)

FEATURE_SCHEMA_VERSION = "1.0"
CATEGORICAL_FEATURES = ("department_id", "appointment_type", "priority_type")
NUMERIC_FEATURES = (
    "reminder_sent",
    "weekday",
    "hour",
    "queue_length",
    "available_doctors",
    "available_rooms",
)
FEATURE_COLUMNS = (*CATEGORICAL_FEATURES, *NUMERIC_FEATURES)


@dataclass
class WaitTimeArtifact:
    model_version: str
    schema_version: str
    feature_columns: tuple[str, ...]
    preprocessor: ColumnTransformer
    point_model: Any
    lower_model: Any
    upper_model: Any


def build_wait_time_features(dataset: Dataset) -> pd.DataFrame:
    required = {"appointments", "encounters", "queue_events"}
    if missing := required - dataset.keys():
        raise DataContractError(f"Missing wait-time tables: {sorted(missing)}")
    waiting = build_waiting_time_target(dataset["encounters"])
    appointments = dataset["appointments"][
        ["appointment_id", "appointment_type", "priority_type", "reminder_sent"]
    ]
    encounter_appointments = dataset["encounters"][["encounter_id", "appointment_id"]]
    queues = dataset["queue_events"][
        [
            "encounter_id",
            "queue_length",
            "available_doctors",
            "available_rooms",
        ]
    ]
    features = (
        waiting.merge(encounter_appointments, on="encounter_id", validate="one_to_one")
        .merge(appointments, on="appointment_id", validate="many_to_one")
        .merge(queues, on="encounter_id", validate="one_to_one")
    )
    output_columns = ["encounter_id", "queue_entered_at", *FEATURE_COLUMNS, "wait_minutes"]
    features = features[output_columns]
    if features.isna().any().any():
        raise TrainingDataError("Wait-time features contain missing values")
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


def regression_metrics(actual: pd.Series, predicted: np.ndarray[Any, Any]) -> dict[str, float]:
    actual_values = actual.to_numpy(dtype=float)
    errors = predicted.astype(float) - actual_values
    absolute = np.abs(errors)
    return {
        "mae": float(absolute.mean()),
        "median_absolute_error": float(np.median(absolute)),
        "rmse": math.sqrt(float(np.mean(errors**2))),
        "p90_absolute_error": float(np.quantile(absolute, 0.9)),
        "underprediction_rate": float(np.mean(errors < 0)),
    }


def slice_metrics(
    features: pd.DataFrame, actual: pd.Series, predicted: np.ndarray[Any, Any]
) -> dict[str, dict[str, dict[str, float]]]:
    evaluated = features[["department_id", "queue_length"]].copy()
    evaluated["actual"] = actual.to_numpy(dtype=float)
    evaluated["prediction"] = predicted
    evaluated["congestion"] = pd.cut(
        evaluated["queue_length"],
        bins=[-1, 1, 3, float("inf")],
        labels=["low", "medium", "high"],
    )
    report: dict[str, dict[str, dict[str, float]]] = {}
    for dimension in ("department_id", "congestion"):
        report[dimension] = {}
        for value, group in evaluated.groupby(dimension, observed=True):
            report[dimension][str(value)] = regression_metrics(
                group["actual"], group["prediction"].to_numpy(dtype=float)
            )
    return report


def predict_wait_time(artifact: WaitTimeArtifact, features: pd.DataFrame) -> pd.DataFrame:
    if missing := set(artifact.feature_columns) - set(features.columns):
        raise FeatureSchemaMismatchError(f"Missing inference features: {sorted(missing)}")
    model_input = features[list(artifact.feature_columns)]
    if model_input.isna().any().any():
        raise FeatureSchemaMismatchError("Inference features contain missing values")
    transformed = artifact.preprocessor.transform(model_input)
    point = np.maximum(0.0, artifact.point_model.predict(transformed))
    lower = np.maximum(0.0, np.minimum(artifact.lower_model.predict(transformed), point))
    upper = np.maximum(point, artifact.upper_model.predict(transformed))
    return pd.DataFrame(
        {
            "predicted_wait_minutes": point,
            "lower_wait_minutes": lower,
            "upper_wait_minutes": upper,
        },
        index=features.index,
    )


def save_artifact(artifact: WaitTimeArtifact, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path)


def load_artifact(path: Path) -> WaitTimeArtifact:
    artifact = joblib.load(path)
    if not isinstance(artifact, WaitTimeArtifact):
        raise ModelArtifactError(f"Unexpected artifact type: {type(artifact).__name__}")
    return artifact
