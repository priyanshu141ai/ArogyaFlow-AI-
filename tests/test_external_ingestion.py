from datetime import UTC, datetime
from pathlib import Path

import pytest

from arogyaflow.data.generation import ScenarioConfig, ScenarioName, generate_dataset
from arogyaflow.data.ingestion import load_external_dataset, run_external_ingestion
from arogyaflow.exceptions import DataContractError


def _write_csv_extract(source: Path) -> None:
    source.mkdir()
    dataset = generate_dataset(
        ScenarioConfig(
            scenario=ScenarioName.NORMAL_WEEK,
            seed=41,
            start=datetime(2026, 1, 5, tzinfo=UTC),
            days=2,
        )
    )
    for table, frame in dataset.items():
        frame.to_csv(source / f"{table}.csv", index=False)


def test_external_extract_is_validated_versioned_and_deidentified(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_csv_extract(source)
    manifest = run_external_ingestion(source, tmp_path / "data", "hospital-export")
    bronze = tmp_path / "data" / "bronze" / "hospital-export" / manifest.source_hash[:12]
    assert manifest.synthetic_data_only is False
    assert manifest.row_counts["appointments"] > 0
    assert (bronze / "appointments.parquet").is_file()


def test_external_extract_rejects_direct_identifiers(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_csv_extract(source)
    appointments = source / "appointments.csv"
    appointments.write_text(
        appointments.read_text(encoding="utf-8").replace(
            "appointment_id,", "patient_name,appointment_id,", 1
        ),
        encoding="utf-8",
    )
    with pytest.raises(DataContractError, match="Direct identifiers"):
        load_external_dataset(source)
