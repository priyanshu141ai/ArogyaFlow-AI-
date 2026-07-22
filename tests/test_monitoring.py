from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from arogyaflow.api import app
from arogyaflow.config import get_settings
from arogyaflow.monitoring import MonitoringReport, MonitoringThresholds, build_monitoring_report


def _report() -> MonitoringReport:
    reference = pd.DataFrame({"queue_length": range(100), "department": ["a", "b"] * 50})
    current = pd.DataFrame({"queue_length": range(100, 200), "department": ["c"] * 100})
    current.loc[0, "queue_length"] = None
    return build_monitoring_report(
        reference,
        current,
        {"mae": 10.0, "recall": 0.8},
        {"mae": 15.0, "recall": 0.6},
        ("queue_length", "department"),
        MonitoringThresholds(),
    )


def test_monitoring_detects_quality_drift_and_degradation() -> None:
    report = _report()
    assert report.status == "review_required"
    assert report.missing_rate > 0
    assert report.drift_scores["department"] == 1
    assert report.metric_degradation["mae"] == 0.5
    assert {alert.component for alert in report.alerts} >= {"drift", "performance"}
    assert report.retraining_blocked is False
    assert report.retraining_recommended is True


def test_monitoring_report_api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    report = _report()
    report_path = tmp_path / "monitoring.json"
    report_path.write_text(report.model_dump_json(), encoding="utf-8")
    monkeypatch.setenv("AROGYAFLOW_MONITORING_REPORT_PATH", str(report_path))
    monkeypatch.delenv("AROGYAFLOW_DATABASE_URL", raising=False)
    get_settings.cache_clear()
    with TestClient(app) as client:
        response = client.get("/v1/monitoring/report")
    get_settings.cache_clear()
    assert response.status_code == 200
    assert response.json()["status"] == "review_required"
