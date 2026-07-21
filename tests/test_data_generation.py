from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError

from arogyaflow.data.generation import ScenarioConfig, ScenarioName, generate_dataset
from arogyaflow.data.pipeline import run_generation
from arogyaflow.data.validation import validate_dataset


def _config(*, anomaly_rate: float = 0.0) -> ScenarioConfig:
    return ScenarioConfig(
        scenario=ScenarioName.NORMAL_WEEK,
        seed=42,
        start=datetime(2026, 1, 5, tzinfo=UTC),
        days=1,
        anomaly_rate=anomaly_rate,
    )


def test_generation_is_reproducible() -> None:
    first = generate_dataset(_config())
    second = generate_dataset(_config())
    for table in first:
        pd.testing.assert_frame_equal(first[table], second[table])


def test_config_requires_timezone() -> None:
    with pytest.raises(ValidationError):
        ScenarioConfig(
            scenario=ScenarioName.NORMAL_WEEK,
            seed=1,
            start=datetime(2026, 1, 5),
        )


def test_invalid_rows_are_quarantined_without_loss() -> None:
    dataset = generate_dataset(_config(anomaly_rate=0.1))
    result = validate_dataset(dataset)
    assert len(result.quarantine["encounters"]) > 0
    for table, frame in dataset.items():
        assert len(frame) == len(result.clean[table]) + len(result.quarantine[table])


def test_pipeline_writes_manifest_and_layers(tmp_path: Path) -> None:
    manifest = run_generation(_config(anomaly_rate=0.05), tmp_path)
    assert manifest.random_seed == 42
    assert sum(manifest.anomaly_counts.values()) > 0
    assert list((tmp_path / "raw").rglob("*.csv"))
    assert list((tmp_path / "bronze").rglob("*.parquet"))
    assert list((tmp_path / "quarantine").rglob("*.parquet"))
    assert list((tmp_path / "reports" / "data_quality").glob("*.json"))
