import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

from arogyaflow.data.contracts import data_dictionary
from arogyaflow.data.generation import GENERATOR_VERSION, ScenarioConfig, generate_dataset
from arogyaflow.data.validation import validate_dataset
from arogyaflow.time import utc_now


@dataclass(frozen=True)
class DatasetManifest:
    generator_version: str
    configuration_hash: str
    random_seed: int
    generated_at: str
    scenario_name: str
    date_range: tuple[str, str]
    row_counts: dict[str, int]
    anomaly_counts: dict[str, int]
    schema: dict[str, dict[str, str]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _configuration_hash(config: ScenarioConfig) -> str:
    payload = config.model_dump_json(exclude_none=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def run_generation(config: ScenarioConfig, output_root: Path) -> DatasetManifest:
    config_hash = _configuration_hash(config)
    run_key = config_hash[:12]
    raw_dir = output_root / "raw" / config.scenario / run_key
    bronze_dir = output_root / "bronze" / config.scenario / run_key
    quarantine_dir = output_root / "quarantine" / config.scenario / run_key
    manifest_path = output_root / "manifests" / f"{config.scenario}-{run_key}.json"
    quality_path = output_root / "reports" / "data_quality" / f"{config.scenario}-{run_key}.json"
    if manifest_path.exists():
        raise FileExistsError(f"Immutable dataset already exists: {manifest_path}")

    dataset = generate_dataset(config)
    result = validate_dataset(dataset)
    for directory in (
        raw_dir,
        bronze_dir,
        quarantine_dir,
        manifest_path.parent,
        quality_path.parent,
    ):
        directory.mkdir(parents=True, exist_ok=False)
    for table, frame in dataset.items():
        frame.to_csv(raw_dir / f"{table}.csv", index=False)
        result.clean[table].to_parquet(bronze_dir / f"{table}.parquet", index=False)
        if not result.quarantine[table].empty:
            result.quarantine[table].to_parquet(quarantine_dir / f"{table}.parquet", index=False)

    end = config.start + timedelta(days=config.days)
    manifest = DatasetManifest(
        generator_version=GENERATOR_VERSION,
        configuration_hash=config_hash,
        random_seed=config.seed,
        generated_at=utc_now().isoformat(),
        scenario_name=config.scenario,
        date_range=(config.start.isoformat(), end.isoformat()),
        row_counts={table: len(frame) for table, frame in result.clean.items()},
        anomaly_counts={table: len(frame) for table, frame in result.quarantine.items()},
        schema=data_dictionary(),
    )
    manifest_path.write_text(json.dumps(manifest.to_dict(), indent=2), encoding="utf-8")
    quality_path.write_text(json.dumps(result.report, indent=2), encoding="utf-8")
    return manifest
