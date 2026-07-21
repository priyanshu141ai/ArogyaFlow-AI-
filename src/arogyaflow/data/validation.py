from dataclasses import dataclass

import pandas as pd
import pandera.pandas as pa

from arogyaflow.data.contracts import CONTRACTS
from arogyaflow.data.generation import Dataset
from arogyaflow.exceptions import DataContractError, DataQualityError

_REFERENCES = (
    ("doctors", "department_id", "departments", "department_id"),
    ("staffing_rosters", "department_id", "departments", "department_id"),
    ("appointments", "department_id", "departments", "department_id"),
    ("appointments", "doctor_id", "doctors", "doctor_id"),
    ("encounters", "appointment_id", "appointments", "appointment_id"),
    ("queue_events", "encounter_id", "encounters", "encounter_id"),
    ("admissions", "encounter_id", "encounters", "encounter_id"),
    ("bed_events", "bed_id", "beds", "bed_id"),
    ("bed_events", "admission_id", "admissions", "admission_id"),
)


@dataclass(frozen=True)
class ValidationResult:
    clean: Dataset
    quarantine: Dataset
    report: dict[str, dict[str, int]]


def _validate_table(table: str, frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        validated = CONTRACTS[table].validate(frame, lazy=True)
        return validated, frame.iloc[0:0].assign(quality_error=pd.Series(dtype=str))
    except KeyError as exc:
        raise DataContractError(f"No data contract for table: {table}") from exc
    except pa.errors.SchemaErrors as exc:
        cases = exc.failure_cases
        invalid_indexes = {index for index in cases["index"].dropna() if index in frame.index}
        if not invalid_indexes:
            invalid_indexes = set(frame.index)
        reason = "; ".join(sorted(set(cases["check"].astype(str))))
        quarantine = frame.loc[list(invalid_indexes)].copy()
        quarantine["quality_error"] = reason
        clean = frame.drop(index=list(invalid_indexes))
        try:
            CONTRACTS[table].validate(clean, lazy=True)
        except pa.errors.SchemaErrors as clean_error:
            raise DataContractError(f"Unable to isolate invalid {table} rows") from clean_error
        return clean, quarantine


def validate_dataset(dataset: Dataset) -> ValidationResult:
    missing = CONTRACTS.keys() - dataset.keys()
    unexpected = dataset.keys() - CONTRACTS.keys()
    if missing or unexpected:
        raise DataContractError(
            f"Dataset tables mismatch; missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )

    clean: Dataset = {}
    quarantine: Dataset = {}
    for table, frame in dataset.items():
        clean[table], quarantine[table] = _validate_table(table, frame)

    for child, foreign_key, parent, primary_key in _REFERENCES:
        invalid_mask = ~clean[child][foreign_key].isin(clean[parent][primary_key])
        if invalid_mask.any():
            invalid = clean[child].loc[invalid_mask].copy()
            invalid["quality_error"] = f"invalid reference: {foreign_key} -> {parent}.{primary_key}"
            quarantine[child] = pd.concat([quarantine[child], invalid], ignore_index=True)
            clean[child] = clean[child].loc[~invalid_mask]

    report = {
        table: {
            "input_rows": len(dataset[table]),
            "valid_rows": len(clean[table]),
            "quarantined_rows": len(quarantine[table]),
        }
        for table in dataset
    }
    for table, counts in report.items():
        if counts["input_rows"] != counts["valid_rows"] + counts["quarantined_rows"]:
            raise DataQualityError(f"Row conservation failed for {table}")
    return ValidationResult(clean=clean, quarantine=quarantine, report=report)
