import math
from dataclasses import dataclass
from typing import Any, cast

import pandas as pd

from arogyaflow.exceptions import DataQualityError


@dataclass(frozen=True)
class TemporalSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    boundaries: dict[str, str]


def temporal_split(
    frame: pd.DataFrame,
    time_column: str,
    train_fraction: float = 0.7,
    validation_fraction: float = 0.15,
) -> TemporalSplit:
    if train_fraction <= 0 or validation_fraction <= 0 or train_fraction + validation_fraction >= 1:
        raise ValueError("split fractions must be positive and total less than one")
    ordered = frame.sort_values(time_column).reset_index(drop=True)
    timestamps = pd.Index(ordered[time_column].drop_duplicates().sort_values())
    if len(timestamps) < 3:
        raise DataQualityError("At least three unique timestamps are required for temporal splits")
    train_end_index = max(1, int(len(timestamps) * train_fraction)) - 1
    validation_end_index = max(
        train_end_index + 1, int(len(timestamps) * (train_fraction + validation_fraction))
    )
    validation_end_index = min(validation_end_index, len(timestamps) - 2)
    train_end = timestamps[train_end_index]
    validation_end = timestamps[validation_end_index]
    train = ordered.loc[ordered[time_column] <= train_end]
    validation = ordered.loc[
        (ordered[time_column] > train_end) & (ordered[time_column] <= validation_end)
    ]
    test = ordered.loc[ordered[time_column] > validation_end]
    if train.empty or validation.empty or test.empty:
        raise DataQualityError("Temporal split produced an empty partition")
    return TemporalSplit(
        train=train,
        validation=validation,
        test=test,
        boundaries={
            "train_start": pd.Timestamp(train[time_column].min()).isoformat(),
            "train_end": pd.Timestamp(train[time_column].max()).isoformat(),
            "validation_start": pd.Timestamp(validation[time_column].min()).isoformat(),
            "validation_end": pd.Timestamp(validation[time_column].max()).isoformat(),
            "test_start": pd.Timestamp(test[time_column].min()).isoformat(),
            "test_end": pd.Timestamp(test[time_column].max()).isoformat(),
        },
    )


def _wait_metrics(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    error = predicted.astype(float) - actual.astype(float)
    absolute_error = error.abs()
    return {
        "mae": float(absolute_error.mean()),
        "median_absolute_error": float(absolute_error.median()),
        "rmse": math.sqrt(float((error**2).mean())),
        "p90_absolute_error": float(absolute_error.quantile(0.9)),
        "underprediction_rate": float((error < 0).mean()),
    }


def waiting_baseline_report(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, dict[str, float]]:
    fallback = float(train["wait_minutes"].median())
    models: dict[str, tuple[str, ...]] = {
        "global_median": (),
        "department_median": ("department_id",),
        "department_weekday_hour_median": ("department_id", "weekday", "hour"),
    }
    report: dict[str, dict[str, float]] = {}
    for name, groups in models.items():
        if not groups:
            predicted = pd.Series(fallback, index=test.index)
        else:
            medians = train.groupby(list(groups), as_index=False).agg(
                prediction=("wait_minutes", "median")
            )
            predicted = test.merge(medians, on=list(groups), how="left")["prediction"].fillna(
                fallback
            )
            predicted.index = test.index
        report[name] = _wait_metrics(test["wait_minutes"], predicted)
    return report


def _arrival_metrics(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    error = predicted.astype(float) - actual.astype(float)
    absolute_error = error.abs()
    return {
        "mae": float(absolute_error.mean()),
        "rmse": math.sqrt(float((error**2).mean())),
        "wape": float(absolute_error.sum()) / max(float(actual.abs().sum()), 1.0),
        "bias": float(error.mean()),
    }


def arrival_baseline_report(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, dict[str, float]]:
    key_columns = ["hospital_id", "department_id"]
    history: dict[tuple[str, str, pd.Timestamp], float] = {}
    for hospital, department, interval, arrivals in train[
        [*key_columns, "interval_start", "arrivals"]
    ].itertuples(index=False, name=None):
        history[(str(hospital), str(department), pd.Timestamp(cast(Any, interval)))] = float(
            cast(Any, arrivals)
        )
    seasonal: dict[tuple[str, str, int, int], float] = {}
    seasonal_values = (
        train.assign(
            weekday=train["interval_start"].dt.weekday,
            hour=train["interval_start"].dt.hour,
        )
        .groupby([*key_columns, "weekday", "hour"])["arrivals"]
        .mean()
    )
    for raw_key, value in seasonal_values.items():
        hospital, department, weekday, hour = cast(tuple[Any, Any, Any, Any], raw_key)
        seasonal[(str(hospital), str(department), int(weekday), int(hour))] = float(value)
    last: dict[tuple[str, str], float] = {}
    last_rows = train.sort_values("interval_start").groupby(key_columns).tail(1)
    for hospital, department, arrivals in last_rows[[*key_columns, "arrivals"]].itertuples(
        index=False, name=None
    ):
        last[(str(hospital), str(department))] = float(cast(Any, arrivals))
    global_fallback = float(train["arrivals"].median())
    predictions: dict[str, list[float]] = {
        "last_observed": [],
        "same_hour_previous_day": [],
        "same_hour_previous_week": [],
        "seasonal_mean": [],
    }
    for hospital, department, interval in test[[*key_columns, "interval_start"]].itertuples(
        index=False, name=None
    ):
        timestamp = pd.Timestamp(cast(Any, interval))
        group = (str(hospital), str(department))
        fallback = seasonal.get((*group, timestamp.weekday(), timestamp.hour), global_fallback)
        predictions["last_observed"].append(last.get(group, global_fallback))
        predictions["same_hour_previous_day"].append(
            history.get((*group, timestamp - pd.Timedelta(days=1)), fallback)
        )
        predictions["same_hour_previous_week"].append(
            history.get((*group, timestamp - pd.Timedelta(days=7)), fallback)
        )
        predictions["seasonal_mean"].append(fallback)
    return {
        name: _arrival_metrics(test["arrivals"], pd.Series(values, index=test.index))
        for name, values in predictions.items()
    }


def split_is_leakage_free(split: TemporalSplit, time_column: str) -> bool:
    return bool(
        split.train[time_column].max() < split.validation[time_column].min()
        and split.validation[time_column].max() < split.test[time_column].min()
    )
