import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
import shap  # type: ignore[import-untyped]
from pydantic import BaseModel, ConfigDict, Field
from sklearn.base import clone  # type: ignore[import-untyped]
from sklearn.dummy import DummyRegressor  # type: ignore[import-untyped]
from sklearn.ensemble import (  # type: ignore[import-untyped]
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Ridge  # type: ignore[import-untyped]

from arogyaflow.baseline_pipeline import load_bronze
from arogyaflow.baselines import temporal_split
from arogyaflow.data.generation import Dataset
from arogyaflow.exceptions import TrainingDataError
from arogyaflow.wait_time import (
    FEATURE_COLUMNS,
    FEATURE_SCHEMA_VERSION,
    WaitTimeArtifact,
    build_wait_time_features,
    make_preprocessor,
    predict_wait_time,
    regression_metrics,
    save_artifact,
    slice_metrics,
)


class WaitTimeTrainingConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_version: str = Field(min_length=1)
    random_seed: int
    shap_sample_size: int = Field(default=100, ge=1, le=1000)
    experiment_name: str = "arogyaflow-wait-time"
    track_experiment: bool = True


@dataclass(frozen=True)
class TrainingResult:
    artifact_path: Path
    report_path: Path
    report: dict[str, Any]
    mlflow_run_id: str | None


def _candidate_models(seed: int) -> dict[str, Any]:
    return {
        "ridge": Ridge(alpha=1.0),
        "random_forest": RandomForestRegressor(
            n_estimators=100,
            min_samples_leaf=3,
            random_state=seed,
            n_jobs=1,
        ),
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            max_iter=100,
            max_leaf_nodes=15,
            random_state=seed,
        ),
    }


def _shap_summary(
    artifact: WaitTimeArtifact, features: pd.DataFrame, sample_size: int
) -> list[dict[str, object]]:
    sample = features[list(FEATURE_COLUMNS)].head(sample_size)
    transformed = artifact.preprocessor.transform(sample)
    if isinstance(artifact.point_model, Ridge):
        values = np.asarray(
            shap.LinearExplainer(artifact.point_model, transformed)(transformed).values
        )
    else:
        values = np.asarray(
            shap.TreeExplainer(artifact.point_model).shap_values(
                transformed, check_additivity=False
            )
        )
    importance = np.abs(values).mean(axis=0)
    names = artifact.preprocessor.get_feature_names_out()
    ranking = sorted(zip(names, importance, strict=True), key=lambda item: item[1], reverse=True)
    return [
        {"feature": str(name), "mean_absolute_shap": float(value)} for name, value in ranking[:20]
    ]


def train_wait_time_model(
    dataset: Dataset, config: WaitTimeTrainingConfig, output_dir: Path
) -> TrainingResult:
    frame = build_wait_time_features(dataset)
    split = temporal_split(frame, "queue_entered_at")
    preprocessor = make_preprocessor()
    train_x = preprocessor.fit_transform(split.train[list(FEATURE_COLUMNS)])
    validation_x = preprocessor.transform(split.validation[list(FEATURE_COLUMNS)])
    candidate_metrics: dict[str, dict[str, float]] = {}
    candidates = _candidate_models(config.random_seed)
    for name, model in candidates.items():
        model.fit(train_x, split.train["wait_minutes"])
        candidate_metrics[name] = regression_metrics(
            split.validation["wait_minutes"], model.predict(validation_x)
        )
    selected_name = min(candidate_metrics, key=lambda name: candidate_metrics[name]["mae"])

    development = pd.concat([split.train, split.validation], ignore_index=True)
    final_preprocessor = make_preprocessor()
    development_x = final_preprocessor.fit_transform(development[list(FEATURE_COLUMNS)])
    test_x = final_preprocessor.transform(split.test[list(FEATURE_COLUMNS)])
    point_model = clone(candidates[selected_name]).fit(development_x, development["wait_minutes"])
    baseline_model = DummyRegressor(strategy="median").fit(
        development_x, development["wait_minutes"]
    )
    lower_model = HistGradientBoostingRegressor(
        loss="quantile", quantile=0.1, max_iter=100, random_state=config.random_seed
    ).fit(development_x, development["wait_minutes"])
    upper_model = HistGradientBoostingRegressor(
        loss="quantile", quantile=0.9, max_iter=100, random_state=config.random_seed
    ).fit(development_x, development["wait_minutes"])
    artifact = WaitTimeArtifact(
        model_version=config.model_version,
        schema_version=FEATURE_SCHEMA_VERSION,
        feature_columns=FEATURE_COLUMNS,
        preprocessor=final_preprocessor,
        point_model=point_model,
        lower_model=lower_model,
        upper_model=upper_model,
    )
    predictions = predict_wait_time(artifact, split.test)
    point_metrics = regression_metrics(
        split.test["wait_minutes"], predictions["predicted_wait_minutes"].to_numpy()
    )
    baseline_metrics = regression_metrics(
        split.test["wait_minutes"], baseline_model.predict(test_x)
    )
    if point_metrics["mae"] >= baseline_metrics["mae"]:
        raise TrainingDataError(
            "Selected model did not beat the median baseline on the test period"
        )

    actual = split.test["wait_minutes"].to_numpy(dtype=float)
    interval_coverage = float(
        np.mean(
            (actual >= predictions["lower_wait_minutes"].to_numpy())
            & (actual <= predictions["upper_wait_minutes"].to_numpy())
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = output_dir / "wait_time.joblib"
    save_artifact(artifact, artifact_path)
    report: dict[str, Any] = {
        "model_version": config.model_version,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "selected_model": selected_name,
        "split_boundaries": split.boundaries,
        "validation_candidates": candidate_metrics,
        "test_metrics": point_metrics,
        "baseline_test_metrics": baseline_metrics,
        "beats_baseline": True,
        "interval_coverage": interval_coverage,
        "slices": slice_metrics(
            split.test,
            split.test["wait_minutes"],
            predictions["predicted_wait_minutes"].to_numpy(),
        ),
        "shap_summary": _shap_summary(artifact, split.test, config.shap_sample_size),
    }
    report_path = output_dir / "wait_time_evaluation.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    run_id: str | None = None
    if config.track_experiment:
        tracking_path = (output_dir / "mlflow.db").resolve().as_posix()
        mlflow.set_tracking_uri(f"sqlite:///{tracking_path}")
        client = mlflow.MlflowClient()
        experiment = client.get_experiment_by_name(config.experiment_name)
        experiment_id = (
            experiment.experiment_id
            if experiment
            else client.create_experiment(
                config.experiment_name,
                artifact_location=(output_dir / "mlartifacts").resolve().as_uri(),
            )
        )
        with mlflow.start_run(experiment_id=experiment_id, run_name=config.model_version) as run:
            mlflow.log_params(
                {
                    "model_version": config.model_version,
                    "selected_model": selected_name,
                    "feature_schema_version": FEATURE_SCHEMA_VERSION,
                    "random_seed": config.random_seed,
                }
            )
            mlflow.log_metrics({f"test_{key}": value for key, value in point_metrics.items()})
            mlflow.log_artifact(str(artifact_path))
            mlflow.log_artifact(str(report_path))
            run_id = run.info.run_id
    return TrainingResult(artifact_path, report_path, report, run_id)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bronze", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("models/wait_time"))
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--seed", required=True, type=int)
    args = parser.parse_args()
    config = WaitTimeTrainingConfig(model_version=args.model_version, random_seed=args.seed)
    train_wait_time_model(load_bronze(args.bronze), config, args.output)


if __name__ == "__main__":
    main()
