from datetime import UTC, datetime
from pathlib import Path

import pytest

from arogyaflow.data.generation import ScenarioConfig, ScenarioName, generate_dataset
from arogyaflow.data.validation import validate_dataset
from arogyaflow.exceptions import FeatureSchemaMismatchError
from arogyaflow.train_wait_time import WaitTimeTrainingConfig, train_wait_time_model
from arogyaflow.wait_time import (
    FEATURE_COLUMNS,
    build_wait_time_features,
    load_artifact,
    predict_wait_time,
)


def test_wait_time_training_and_inference(tmp_path: Path) -> None:
    dataset = validate_dataset(
        generate_dataset(
            ScenarioConfig(
                scenario=ScenarioName.NORMAL_WEEK,
                seed=19,
                start=datetime(2026, 1, 5, tzinfo=UTC),
                days=3,
            )
        )
    ).clean
    result = train_wait_time_model(
        dataset,
        WaitTimeTrainingConfig(
            model_version="test-v1",
            random_seed=11,
            shap_sample_size=20,
        ),
        tmp_path,
    )
    assert result.report["beats_baseline"] is True
    assert result.mlflow_run_id
    assert result.report["mlflow"]["registered_model_version"]
    artifact = load_artifact(result.artifact_path)
    features = build_wait_time_features(dataset).head(3)
    predictions = predict_wait_time(artifact, features)
    assert (predictions["lower_wait_minutes"] <= predictions["predicted_wait_minutes"]).all()
    assert (predictions["predicted_wait_minutes"] <= predictions["upper_wait_minutes"]).all()

    with pytest.raises(FeatureSchemaMismatchError):
        predict_wait_time(artifact, features.drop(columns=FEATURE_COLUMNS[0]))
