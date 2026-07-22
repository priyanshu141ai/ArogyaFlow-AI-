import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

import arogyaflow.retraining as retraining
from arogyaflow.data.generation import Dataset
from arogyaflow.monitoring import MonitoringAlert, MonitoringReport
from arogyaflow.retraining import relative_improvement, retraining_status
from arogyaflow.train_wait_time import TrainingResult, WaitTimeTrainingConfig


def _report(component: str | None) -> MonitoringReport:
    alerts = (
        []
        if component is None
        else [
            MonitoringAlert(
                component=component,  # type: ignore[arg-type]
                metric="queue_length",
                observed=0.4,
                threshold=0.2,
                message="review",
            )
        ]
    )
    return MonitoringReport(
        generated_at=datetime(2026, 1, 1, tzinfo=UTC),
        status="review_required" if alerts else "healthy",
        reference_rows=100,
        current_rows=100,
        missing_rate=0,
        duplicate_rate=0,
        drift_scores={},
        metric_degradation={},
        alerts=alerts,
        retraining_recommended=component in {"drift", "performance"},
        retraining_blocked=component == "data_quality",
    )


def test_retraining_gates_data_quality_and_unnecessary_runs() -> None:
    assert retraining_status(_report("data_quality")).action == "blocked"  # type: ignore[union-attr]
    assert retraining_status(_report(None)).action == "skipped"  # type: ignore[union-attr]
    assert retraining_status(_report("drift")) is None


def test_relative_improvement_supports_loss_and_score_metrics() -> None:
    assert relative_improvement(10, 8, higher_is_better=False) == pytest.approx(0.2)
    assert relative_improvement(0.7, 0.77, higher_is_better=True) == pytest.approx(0.1)


def test_triggered_candidate_is_promoted_with_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    production = tmp_path / "production"
    production.mkdir()
    (production / "wait_time.joblib").write_text("old", encoding="utf-8")
    (production / "wait_time_evaluation.json").write_text(
        json.dumps({"test_metrics": {"mae": 10}}), encoding="utf-8"
    )

    def fake_train(
        dataset: Dataset, config: WaitTimeTrainingConfig, output_dir: Path
    ) -> TrainingResult:
        del dataset, config
        output_dir.mkdir(parents=True)
        artifact = output_dir / "wait_time.joblib"
        report = output_dir / "wait_time_evaluation.json"
        artifact.write_text("new", encoding="utf-8")
        payload = {"test_metrics": {"mae": 8.0}}
        report.write_text(json.dumps(payload), encoding="utf-8")
        return TrainingResult(artifact, report, payload, None)

    monkeypatch.setattr(retraining, "train_wait_time_model", fake_train)
    result = retraining.retrain_wait_time_model(
        {},
        _report("drift"),
        WaitTimeTrainingConfig(model_version="candidate-v2", random_seed=1, track_experiment=False),
        production,
        tmp_path / "candidates",
        minimum_relative_improvement=0.1,
    )
    assert result.action == "promoted"
    assert (production / "wait_time.joblib").read_text(encoding="utf-8") == "new"
    assert list((production / "archive").glob("*/wait_time.joblib"))
