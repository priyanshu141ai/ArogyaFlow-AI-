import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pydantic import ConfigDict, Field
from sklearn.calibration import CalibratedClassifierCV  # type: ignore[import-untyped]
from sklearn.ensemble import (  # type: ignore[import-untyped]
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.frozen import FrozenEstimator  # type: ignore[import-untyped]
from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]

from arogyaflow.baseline_pipeline import load_bronze
from arogyaflow.baselines import temporal_split
from arogyaflow.data.generation import Dataset
from arogyaflow.exceptions import TrainingDataError
from arogyaflow.no_show import (
    FEATURE_COLUMNS,
    NO_SHOW_SCHEMA_VERSION,
    NoShowArtifact,
    build_no_show_features,
    classification_metrics,
    expected_calibration_error,
    predict_no_show,
    save_artifact,
)
from arogyaflow.tracking import ModelKind, TrackingOptions, flatten_metrics, log_training_run


class NoShowTrainingConfig(TrackingOptions):
    model_config = ConfigDict(frozen=True)

    model_version: str = Field(min_length=1)
    random_seed: int
    reminder_capacity_fraction: float = Field(gt=0, le=1)
    reminder_effectiveness: float = Field(ge=0, le=1)
    grace_minutes: int = Field(default=60, ge=0)
    late_minutes: int = Field(default=15, ge=0)
    maximum_ece: float = Field(default=0.25, gt=0, le=1)
    experiment_name: str = "arogyaflow-no-show"
    registered_model_name: str = "arogyaflow-no-show"


@dataclass(frozen=True)
class NoShowTrainingResult:
    artifact_path: Path
    report_path: Path
    report: dict[str, Any]
    mlflow_run_id: str | None


def _candidates(seed: int) -> dict[str, Any]:
    return {
        "random_forest": RandomForestClassifier(
            n_estimators=100,
            min_samples_leaf=4,
            class_weight="balanced",
            random_state=seed,
            n_jobs=1,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=100,
            max_leaf_nodes=15,
            random_state=seed,
        ),
    }


def _require_both_classes(partition: pd.DataFrame, name: str) -> None:
    if partition["no_show"].nunique() != 2:
        raise TrainingDataError(f"{name} split must contain show and no-show examples")


def _capacity_threshold(probabilities: np.ndarray[Any, Any], fraction: float) -> float:
    capacity = max(1, int(len(probabilities) * fraction))
    return float(np.sort(probabilities)[-capacity])


def _reminder_simulation(
    actual: pd.Series,
    probabilities: np.ndarray[Any, Any],
    capacity_fraction: float,
    effectiveness: float,
) -> dict[str, float | int | bool]:
    capacity = max(1, int(len(probabilities) * capacity_fraction))
    selected = np.argsort(probabilities)[-capacity:]
    no_shows_selected = int(actual.to_numpy(dtype=int)[selected].sum())
    total_no_shows = int(actual.sum())
    return {
        "appointments": len(actual),
        "reminder_capacity": capacity,
        "observed_no_shows_prioritized": no_shows_selected,
        "recall_at_capacity": no_shows_selected / max(total_no_shows, 1),
        "estimated_slots_recovered": no_shows_selected * effectiveness,
        "automatic_cancellation": False,
    }


def train_no_show_model(
    dataset: Dataset, config: NoShowTrainingConfig, output_dir: Path
) -> NoShowTrainingResult:
    frame = build_no_show_features(
        dataset,
        grace_minutes=config.grace_minutes,
        late_minutes=config.late_minutes,
    )
    split = temporal_split(frame, "booking_created_at")
    for name, partition in (
        ("train", split.train),
        ("validation", split.validation),
        ("test", split.test),
    ):
        _require_both_classes(partition, name)

    from arogyaflow.no_show import make_preprocessor

    preprocessor = make_preprocessor()
    train_x = preprocessor.fit_transform(split.train[list(FEATURE_COLUMNS)])
    validation_x = preprocessor.transform(split.validation[list(FEATURE_COLUMNS)])
    test_x = preprocessor.transform(split.test[list(FEATURE_COLUMNS)])
    logistic = LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=config.random_seed
    ).fit(train_x, split.train["no_show"])
    logistic_validation = logistic.predict_proba(validation_x)[:, 1]
    validation_report: dict[str, dict[str, float]] = {
        "logistic_baseline": classification_metrics(
            split.validation["no_show"], logistic_validation, 0.5
        )
    }
    candidates = _candidates(config.random_seed)
    for name, classifier in candidates.items():
        classifier.fit(train_x, split.train["no_show"])
        validation_report[name] = classification_metrics(
            split.validation["no_show"], classifier.predict_proba(validation_x)[:, 1], 0.5
        )
    selected_name = min(candidates, key=lambda name: validation_report[name]["brier_score"])
    calibrated = CalibratedClassifierCV(
        FrozenEstimator(candidates[selected_name]), method="sigmoid"
    ).fit(validation_x, split.validation["no_show"])
    validation_probabilities = calibrated.predict_proba(validation_x)[:, 1]
    threshold = _capacity_threshold(validation_probabilities, config.reminder_capacity_fraction)
    test_probabilities = calibrated.predict_proba(test_x)[:, 1]
    test_metrics = classification_metrics(split.test["no_show"], test_probabilities, threshold)
    ece = expected_calibration_error(split.test["no_show"], test_probabilities)
    if ece > config.maximum_ece:
        raise TrainingDataError(f"Calibration ECE {ece:.3f} exceeds {config.maximum_ece:.3f}")

    artifact = NoShowArtifact(
        model_version=config.model_version,
        schema_version=NO_SHOW_SCHEMA_VERSION,
        feature_columns=FEATURE_COLUMNS,
        preprocessor=preprocessor,
        classifier=calibrated,
        reminder_threshold=threshold,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = output_dir / "no_show.joblib"
    save_artifact(artifact, artifact_path)
    inference = predict_no_show(artifact, split.test)
    report: dict[str, Any] = {
        "model_version": config.model_version,
        "feature_schema_version": NO_SHOW_SCHEMA_VERSION,
        "selected_model": selected_name,
        "split_boundaries": split.boundaries,
        "validation_models": validation_report,
        "test_metrics": test_metrics,
        "calibration_acceptable": True,
        "reminder_threshold": threshold,
        "reminder_capacity_fraction": config.reminder_capacity_fraction,
        "policy_simulation": _reminder_simulation(
            split.test["no_show"],
            inference["no_show_probability"].to_numpy(),
            config.reminder_capacity_fraction,
            config.reminder_effectiveness,
        ),
        "limitations": [
            "Threshold prioritizes reminders only and never cancels appointments.",
            "Recovery estimate depends on the configured reminder-effectiveness assumption.",
        ],
    }
    report_path = output_dir / "no_show_evaluation.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    run = log_training_run(
        config,
        model_kind=ModelKind.NO_SHOW,
        output_dir=output_dir,
        artifact_path=artifact_path,
        report_path=report_path,
        parameters={
            "model_version": config.model_version,
            "selected_model": selected_name,
            "feature_schema_version": NO_SHOW_SCHEMA_VERSION,
            "random_seed": config.random_seed,
        },
        metrics=flatten_metrics(report),
    )
    report["mlflow"] = run.__dict__ if run else {"tracking_enabled": False}
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return NoShowTrainingResult(artifact_path, report_path, report, run.run_id if run else None)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bronze", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("models/no_show"))
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--capacity-fraction", required=True, type=float)
    parser.add_argument("--reminder-effectiveness", required=True, type=float)
    parser.add_argument("--tracking-uri", default=os.environ.get("MLFLOW_TRACKING_URI"))
    parser.add_argument("--skip-tracking", action="store_true")
    args = parser.parse_args()
    config = NoShowTrainingConfig(
        model_version=args.model_version,
        random_seed=args.seed,
        reminder_capacity_fraction=args.capacity_fraction,
        reminder_effectiveness=args.reminder_effectiveness,
        mlflow_tracking_uri=args.tracking_uri,
        track_experiment=not args.skip_tracking,
    )
    train_no_show_model(load_bronze(args.bronze), config, args.output)


if __name__ == "__main__":
    main()
