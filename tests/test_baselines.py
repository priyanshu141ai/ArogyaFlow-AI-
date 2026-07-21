from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd

from arogyaflow.analysis import build_waiting_time_target
from arogyaflow.baseline_pipeline import run_baseline_analysis
from arogyaflow.baselines import split_is_leakage_free, temporal_split
from arogyaflow.data.generation import ScenarioConfig, ScenarioName, generate_dataset
from arogyaflow.data.validation import validate_dataset


def test_waiting_target_uses_queue_entry() -> None:
    entered = datetime(2026, 1, 1, 9, tzinfo=UTC)
    frame = pd.DataFrame(
        {
            "encounter_id": ["enc-1"],
            "department_id": ["dep-1"],
            "queue_entered_at": [entered],
            "consultation_started_at": [entered + timedelta(minutes=17)],
        }
    )
    target = build_waiting_time_target(frame)
    assert target.loc[0, "wait_minutes"] == 17


def test_temporal_split_is_strictly_ordered() -> None:
    frame = pd.DataFrame(
        {
            "at": pd.date_range("2026-01-01", periods=20, freq="h", tz="UTC"),
            "value": range(20),
        }
    )
    split = temporal_split(frame, "at")
    assert split_is_leakage_free(split, "at")


def test_phase3_pipeline_stores_metrics_and_checks(tmp_path: Path) -> None:
    config = ScenarioConfig(
        scenario=ScenarioName.NORMAL_WEEK,
        seed=7,
        start=datetime(2026, 1, 5, tzinfo=UTC),
        days=2,
    )
    dataset = validate_dataset(generate_dataset(config)).clean
    result = run_baseline_analysis(dataset, tmp_path)
    assert all(result["leakage_checks"].values())
    assert "department_weekday_hour_median" in result["metrics"]["waiting_time"]
    assert {"baseline_metrics.json", "eda.json", "leakage_checks.json"} == {
        path.name for path in tmp_path.glob("*.json")
    }
