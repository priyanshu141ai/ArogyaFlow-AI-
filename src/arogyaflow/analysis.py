from dataclasses import asdict, dataclass
from typing import cast

import pandas as pd

from arogyaflow.data.generation import Dataset
from arogyaflow.exceptions import DataContractError, DataQualityError


@dataclass(frozen=True)
class TargetDefinition:
    name: str
    unit: str
    definition: str
    prediction_time: str


TARGET_DEFINITIONS = (
    TargetDefinition(
        "waiting_time",
        "minutes",
        "consultation_started_at - queue_entered_at",
        "queue_entered_at",
    ),
    TargetDefinition(
        "arrivals",
        "patients_per_interval",
        "count of queue entries by hospital and department",
        "interval_start",
    ),
)


def target_definitions() -> list[dict[str, str]]:
    return [asdict(definition) for definition in TARGET_DEFINITIONS]


def build_waiting_time_target(encounters: pd.DataFrame) -> pd.DataFrame:
    required = {
        "encounter_id",
        "department_id",
        "queue_entered_at",
        "consultation_started_at",
    }
    if missing := required - set(encounters.columns):
        raise DataContractError(f"Missing encounter columns: {sorted(missing)}")
    target = encounters[
        ["encounter_id", "department_id", "queue_entered_at", "consultation_started_at"]
    ].copy()
    target["wait_minutes"] = (
        target["consultation_started_at"] - target["queue_entered_at"]
    ).dt.total_seconds() / 60
    if target["wait_minutes"].isna().any() or (target["wait_minutes"] < 0).any():
        raise DataQualityError("Waiting-time target contains missing or negative values")
    target["weekday"] = target["queue_entered_at"].dt.weekday
    target["hour"] = target["queue_entered_at"].dt.hour
    return target.drop(columns="consultation_started_at")


def build_arrival_target(
    appointments: pd.DataFrame, encounters: pd.DataFrame, frequency: str = "1h"
) -> pd.DataFrame:
    if pd.Timedelta(frequency) <= pd.Timedelta(0):
        raise ValueError("frequency must be positive")
    hospital = appointments[["appointment_id", "hospital_id"]]
    events = encounters[["appointment_id", "department_id", "queue_entered_at"]].merge(
        hospital, on="appointment_id", validate="many_to_one"
    )
    events["interval_start"] = events["queue_entered_at"].dt.floor(frequency)
    counts = (
        events.groupby(["hospital_id", "department_id", "interval_start"], as_index=False)
        .size()
        .rename(columns={"size": "arrivals"})
    )
    completed: list[pd.DataFrame] = []
    for (hospital_id, department_id), group in counts.groupby(
        ["hospital_id", "department_id"], sort=True
    ):
        intervals = pd.date_range(
            group["interval_start"].min(), group["interval_start"].max(), freq=frequency
        )
        full = group.set_index("interval_start").reindex(intervals, fill_value=0)
        full.index.name = "interval_start"
        full = full.assign(hospital_id=str(hospital_id), department_id=str(department_id))
        completed.append(full.reset_index())
    if not completed:
        raise DataQualityError("No encounter arrivals available")
    return pd.concat(completed, ignore_index=True)[
        ["hospital_id", "department_id", "interval_start", "arrivals"]
    ]


def profile_dataset(dataset: Dataset) -> dict[str, dict[str, object]]:
    profile: dict[str, dict[str, object]] = {}
    for table, frame in dataset.items():
        time_columns = [
            str(column) for column in frame.columns if str(column).endswith(("_at", "_time"))
        ]
        profile[table] = {
            "rows": len(frame),
            "columns": len(frame.columns),
            "duplicate_rows": int(frame.duplicated().sum()),
            "null_rates": {
                column: round(float(rate), 6) for column, rate in frame.isna().mean().items()
            },
            "time_ranges": {
                column: {
                    "minimum": frame[column].min().isoformat(),
                    "maximum": frame[column].max().isoformat(),
                }
                for column in time_columns
                if not frame.empty and frame[column].notna().any()
            },
        }
    return profile


def queue_demand_summary(dataset: Dataset) -> dict[str, list[dict[str, object]]]:
    queue = dataset["queue_events"]
    queue_summary = (
        queue.groupby("department_id")
        .agg(
            events=("queue_event_id", "size"),
            average_queue=("queue_length", "mean"),
            p95_queue=("queue_length", lambda values: values.quantile(0.95)),
            average_doctors=("available_doctors", "mean"),
        )
        .reset_index()
    )
    arrivals = build_arrival_target(dataset["appointments"], dataset["encounters"])
    demand_summary = (
        arrivals.groupby("department_id")
        .agg(
            average_hourly_arrivals=("arrivals", "mean"),
            p95_hourly_arrivals=("arrivals", lambda values: values.quantile(0.95)),
            peak_hourly_arrivals=("arrivals", "max"),
        )
        .reset_index()
    )
    return {
        "queue": cast(list[dict[str, object]], queue_summary.to_dict(orient="records")),
        "demand": cast(list[dict[str, object]], demand_summary.to_dict(orient="records")),
    }
