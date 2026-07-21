import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from arogyaflow.analysis import (
    build_arrival_target,
    build_waiting_time_target,
    profile_dataset,
    queue_demand_summary,
    target_definitions,
)
from arogyaflow.baselines import (
    arrival_baseline_report,
    split_is_leakage_free,
    temporal_split,
    waiting_baseline_report,
)
from arogyaflow.data.generation import Dataset
from arogyaflow.exceptions import DataContractError, FeatureLeakageError

_REQUIRED_TABLES = {"appointments", "encounters", "queue_events"}


def run_baseline_analysis(dataset: Dataset, output_dir: Path) -> dict[str, Any]:
    if missing := _REQUIRED_TABLES - dataset.keys():
        raise DataContractError(f"Missing baseline tables: {sorted(missing)}")
    waiting = build_waiting_time_target(dataset["encounters"])
    arrivals = build_arrival_target(dataset["appointments"], dataset["encounters"])
    waiting_split = temporal_split(waiting, "queue_entered_at")
    arrival_split = temporal_split(arrivals, "interval_start")
    leakage_checks = {
        "waiting_split_order": split_is_leakage_free(waiting_split, "queue_entered_at"),
        "arrival_split_order": split_is_leakage_free(arrival_split, "interval_start"),
        "waiting_features_exclude_future_events": not {
            "consultation_started_at",
            "consultation_ended_at",
            "checkout_at",
        }.intersection(waiting.columns),
        "forecast_fit_precedes_test": bool(
            arrival_split.train["interval_start"].max() < arrival_split.test["interval_start"].min()
        ),
    }
    if not all(leakage_checks.values()):
        raise FeatureLeakageError("One or more baseline leakage checks failed")

    metrics: dict[str, object] = {
        "targets": target_definitions(),
        "time_splits": {
            "waiting_time": waiting_split.boundaries,
            "arrivals": arrival_split.boundaries,
        },
        "waiting_time": waiting_baseline_report(waiting_split.train, waiting_split.test),
        "arrivals": arrival_baseline_report(arrival_split.train, arrival_split.test),
        "limitations": [
            "Synthetic operational data only; no real patient data.",
            "Baselines are references and are not deployment candidates.",
            "Forecast baselines use training history only.",
        ],
    }
    eda: dict[str, object] = {
        "profile": profile_dataset(dataset),
        "queue_and_demand": queue_demand_summary(dataset),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Any] = {
        "baseline_metrics.json": metrics,
        "eda.json": eda,
        "leakage_checks.json": leakage_checks,
    }
    for name, content in outputs.items():
        (output_dir / name).write_text(json.dumps(content, indent=2), encoding="utf-8")
    return {"metrics": metrics, "eda": eda, "leakage_checks": leakage_checks}


def load_bronze(directory: Path) -> Dataset:
    dataset = {path.stem: pd.read_parquet(path) for path in directory.glob("*.parquet")}
    if not dataset:
        raise FileNotFoundError(f"No Parquet tables found in {directory}")
    return dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bronze", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("reports/phase3"))
    args = parser.parse_args()
    run_baseline_analysis(load_bronze(args.bronze), args.output)


if __name__ == "__main__":
    main()
