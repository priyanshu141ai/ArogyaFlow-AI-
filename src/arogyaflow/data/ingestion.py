"""Ingest de-identified HMIS/FHIR extracts mapped to ArogyaFlow tables."""

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from arogyaflow.data.contracts import CONTRACTS, data_dictionary
from arogyaflow.data.generation import Dataset
from arogyaflow.data.validation import validate_dataset
from arogyaflow.exceptions import DataContractError
from arogyaflow.time import utc_now

_DIRECT_IDENTIFIERS = {
    "aadhaar",
    "abha_number",
    "address",
    "date_of_birth",
    "email",
    "mobile",
    "patient_name",
    "phone",
    "phone_number",
}


@dataclass(frozen=True)
class ExternalDatasetManifest:
    dataset_name: str
    source_hash: str
    imported_at: str
    synthetic_data_only: bool
    source_files: dict[str, str]
    row_counts: dict[str, int]
    quarantine_counts: dict[str, int]
    schema: dict[str, dict[str, str]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _table_path(source: Path, table: str) -> Path:
    matches = [
        path for suffix in (".parquet", ".csv") if (path := source / f"{table}{suffix}").is_file()
    ]
    if len(matches) != 1:
        raise DataContractError(f"Expected one CSV or Parquet file for table: {table}")
    return matches[0]


def _read_table(table: str, path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    forbidden = _DIRECT_IDENTIFIERS.intersection(map(str.lower, frame.columns))
    if forbidden:
        raise DataContractError(f"Direct identifiers are forbidden: {sorted(forbidden)}")
    for name, column in CONTRACTS[table].columns.items():
        if name in frame and "datetime64" in str(column.dtype):
            frame[name] = pd.to_datetime(frame[name], utc=True, errors="coerce")
    return frame


def load_external_dataset(source: Path) -> tuple[Dataset, dict[str, Path]]:
    if not source.is_dir():
        raise DataContractError(f"External dataset directory does not exist: {source}")
    files = {table: _table_path(source, table) for table in CONTRACTS}
    return {table: _read_table(table, path) for table, path in files.items()}, files


def _source_hash(files: dict[str, Path]) -> str:
    digest = hashlib.sha256()
    for table, path in sorted(files.items()):
        digest.update(f"{table}:{path.name}".encode())
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def run_external_ingestion(
    source: Path, output_root: Path, dataset_name: str
) -> ExternalDatasetManifest:
    if not dataset_name.strip():
        raise DataContractError("Dataset name cannot be empty")
    dataset, files = load_external_dataset(source)
    source_hash = _source_hash(files)
    run_key = source_hash[:12]
    bronze_dir = output_root / "bronze" / dataset_name / run_key
    quarantine_dir = output_root / "quarantine" / dataset_name / run_key
    manifest_path = output_root / "manifests" / f"{dataset_name}-{run_key}.json"
    quality_path = output_root / "reports" / "data_quality" / f"{dataset_name}-{run_key}.json"
    if manifest_path.exists():
        raise FileExistsError(f"Immutable dataset already exists: {manifest_path}")

    result = validate_dataset(dataset)
    bronze_dir.mkdir(parents=True, exist_ok=False)
    for table, frame in result.clean.items():
        frame.to_parquet(bronze_dir / f"{table}.parquet", index=False)
        if not result.quarantine[table].empty:
            quarantine_dir.mkdir(parents=True, exist_ok=True)
            result.quarantine[table].to_parquet(quarantine_dir / f"{table}.parquet", index=False)

    manifest = ExternalDatasetManifest(
        dataset_name=dataset_name,
        source_hash=source_hash,
        imported_at=utc_now().isoformat(),
        synthetic_data_only=False,
        source_files={table: path.name for table, path in files.items()},
        row_counts={table: len(frame) for table, frame in result.clean.items()},
        quarantine_counts={table: len(frame) for table, frame in result.quarantine.items()},
        schema=data_dictionary(),
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    quality_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest.to_dict(), indent=2), encoding="utf-8")
    quality_path.write_text(json.dumps(result.report, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("data"))
    parser.add_argument("--dataset-name", required=True)
    args = parser.parse_args()
    print(json.dumps(run_external_ingestion(args.source, args.output, args.dataset_name).to_dict()))


if __name__ == "__main__":
    main()
