import argparse
import json
import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from arogyaflow.baseline_pipeline import load_bronze
from arogyaflow.data.generation import Dataset
from arogyaflow.exceptions import TrainingDataError
from arogyaflow.monitoring import MonitoringReport, load_monitoring_report
from arogyaflow.time import utc_now
from arogyaflow.train_wait_time import (
    WaitTimeTrainingConfig,
    train_wait_time_model,
)


@dataclass(frozen=True)
class RetrainingResult:
    action: Literal["skipped", "blocked", "rejected", "promoted"]
    reason: str
    candidate_metric: float | None = None
    production_metric: float | None = None
    relative_improvement: float | None = None


def _metric(report: dict[str, object], path: str) -> float:
    value: object = report
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            raise TrainingDataError(f"Evaluation report is missing metric: {path}")
        value = value[part]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TrainingDataError(f"Evaluation metric is not numeric: {path}")
    return float(value)


def relative_improvement(current: float, candidate: float, *, higher_is_better: bool) -> float:
    gain = candidate - current if higher_is_better else current - candidate
    return gain / max(abs(current), 1e-9)


def retraining_status(report: MonitoringReport) -> RetrainingResult | None:
    quality_alert = report.retraining_blocked or any(
        alert.component == "data_quality" for alert in report.alerts
    )
    model_alert = report.retraining_recommended or any(
        alert.component in {"drift", "performance"} for alert in report.alerts
    )
    if quality_alert:
        return RetrainingResult("blocked", "Data quality must be fixed before retraining")
    if not model_alert:
        return RetrainingResult("skipped", "No drift or performance trigger")
    return None


def _read_report(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise TrainingDataError(f"Cannot read evaluation report: {path}") from exc
    if not isinstance(payload, dict):
        raise TrainingDataError("Evaluation report must contain a JSON object")
    return payload


def _atomic_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(f"{target.suffix}.tmp")
    shutil.copy2(source, temporary)
    temporary.replace(target)


def _promote(candidate_artifact: Path, candidate_report: Path, production_dir: Path) -> None:
    production_artifact = production_dir / "wait_time.joblib"
    production_report = production_dir / "wait_time_evaluation.json"
    if production_artifact.exists() or production_report.exists():
        archive = production_dir / "archive" / utc_now().strftime("%Y%m%dT%H%M%SZ")
        archive.mkdir(parents=True, exist_ok=False)
        for path in (production_artifact, production_report):
            if path.exists():
                shutil.copy2(path, archive / path.name)
    _atomic_copy(candidate_artifact, production_artifact)
    _atomic_copy(candidate_report, production_report)


def retrain_wait_time_model(
    dataset: Dataset,
    monitoring: MonitoringReport,
    config: WaitTimeTrainingConfig,
    production_dir: Path,
    candidate_root: Path,
    minimum_relative_improvement: float = 0.0,
) -> RetrainingResult:
    if minimum_relative_improvement < 0:
        raise ValueError("minimum_relative_improvement cannot be negative")
    if decision := retraining_status(monitoring):
        return decision
    candidate_dir = candidate_root / config.model_version
    if candidate_dir.exists():
        raise FileExistsError(f"Candidate model version already exists: {candidate_dir}")
    trained = train_wait_time_model(dataset, config, candidate_dir)
    candidate_metric = _metric(trained.report, "test_metrics.mae")
    production_report_path = production_dir / "wait_time_evaluation.json"
    if production_report_path.exists():
        production_metric = _metric(_read_report(production_report_path), "test_metrics.mae")
        improvement = relative_improvement(
            production_metric, candidate_metric, higher_is_better=False
        )
        if improvement < minimum_relative_improvement:
            return RetrainingResult(
                "rejected",
                "Candidate did not meet the promotion threshold",
                candidate_metric,
                production_metric,
                improvement,
            )
    else:
        production_metric = None
        improvement = None
    _promote(trained.artifact_path, trained.report_path, production_dir)
    return RetrainingResult(
        "promoted",
        "Candidate passed monitoring and evaluation gates",
        candidate_metric,
        production_metric,
        improvement,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bronze", required=True, type=Path)
    parser.add_argument("--monitoring-report", required=True, type=Path)
    parser.add_argument("--production", type=Path, default=Path("models/wait_time"))
    parser.add_argument("--candidates", type=Path, default=Path("models/candidates/wait_time"))
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--minimum-improvement", type=float, default=0.0)
    parser.add_argument("--tracking-uri", default=os.environ.get("MLFLOW_TRACKING_URI"))
    parser.add_argument("--skip-tracking", action="store_true")
    args = parser.parse_args()
    result = retrain_wait_time_model(
        load_bronze(args.bronze),
        load_monitoring_report(args.monitoring_report),
        WaitTimeTrainingConfig(
            model_version=args.model_version,
            random_seed=args.seed,
            mlflow_tracking_uri=args.tracking_uri,
            track_experiment=not args.skip_tracking,
        ),
        args.production,
        args.candidates,
        args.minimum_improvement,
    )
    print(json.dumps(asdict(result)))


if __name__ == "__main__":
    main()
