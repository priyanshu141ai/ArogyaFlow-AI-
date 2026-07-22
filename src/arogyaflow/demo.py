import argparse
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from secrets import token_urlsafe
from time import monotonic, sleep
from urllib.error import URLError
from urllib.request import urlopen

from pydantic import AnyHttpUrl, BaseModel, ConfigDict, Field, ValidationError, model_validator

from arogyaflow.arrival_forecasting import ArrivalTrainingConfig, train_arrival_from_dataset
from arogyaflow.bed_occupancy import (
    OccupancyTrainingConfig,
    build_occupancy_target,
    train_occupancy_model,
)
from arogyaflow.data.generation import ScenarioConfig, ScenarioName, generate_dataset
from arogyaflow.data.validation import validate_dataset
from arogyaflow.exceptions import DemoSetupError
from arogyaflow.monitoring import MonitoringThresholds, build_monitoring_report
from arogyaflow.train_no_show import NoShowTrainingConfig, train_no_show_model
from arogyaflow.train_wait_time import WaitTimeTrainingConfig, train_wait_time_model


class DemoConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    scenario: ScenarioName
    data_seed: int
    start: datetime
    days: int = Field(ge=10, le=30)
    wait_model_version: str = Field(min_length=1)
    wait_seed: int
    wait_shap_sample_size: int = Field(ge=1, le=1000)
    arrival_model_version: str = Field(min_length=1)
    arrival_seed: int
    forecast_backtest_windows: int = Field(ge=2, le=8)
    no_show_model_version: str = Field(min_length=1)
    no_show_seed: int
    reminder_capacity_fraction: float = Field(gt=0, le=1)
    reminder_effectiveness: float = Field(ge=0, le=1)
    maximum_ece: float = Field(gt=0, le=1)
    occupancy_model_version: str = Field(min_length=1)
    occupancy_seed: int
    occupancy_alert_threshold: float = Field(gt=0, le=1)
    wait_output_dir: Path
    arrival_output_dir: Path
    no_show_output_dir: Path
    occupancy_output_dir: Path
    monitoring_report_path: Path
    monitoring_columns: tuple[str, ...] = Field(min_length=1)
    monitoring_thresholds: MonitoringThresholds
    track_experiment: bool
    mlflow_tracking_uri: AnyHttpUrl
    mlflow_health_url: AnyHttpUrl
    postgres_user: str = Field(min_length=1)
    postgres_database: str = Field(min_length=1)

    @model_validator(mode="after")
    def require_portable_configuration(self) -> "DemoConfig":
        if self.start.tzinfo is None or self.start.utcoffset() is None:
            raise ValueError("start must be timezone-aware")
        paths = (
            self.wait_output_dir,
            self.arrival_output_dir,
            self.no_show_output_dir,
            self.occupancy_output_dir,
            self.monitoring_report_path,
        )
        if any(path.is_absolute() or ".." in path.parts for path in paths):
            raise ValueError("demo paths must stay relative to the project root")
        return self


@dataclass(frozen=True)
class DemoArtifacts:
    wait_model: Path
    arrival_model: Path
    no_show_model: Path
    occupancy_model: Path
    monitoring_report: Path


def load_demo_config(path: Path) -> DemoConfig:
    try:
        return DemoConfig.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError) as exc:
        raise DemoSetupError("Demo configuration is missing or invalid") from exc


def prepare_demo(config: DemoConfig, project_root: Path = Path(".")) -> DemoArtifacts:
    dataset = validate_dataset(
        generate_dataset(
            ScenarioConfig(
                scenario=config.scenario,
                seed=config.data_seed,
                start=config.start,
                days=config.days,
            )
        )
    ).clean
    tracking_uri = str(config.mlflow_tracking_uri)
    wait = train_wait_time_model(
        dataset,
        WaitTimeTrainingConfig(
            model_version=config.wait_model_version,
            random_seed=config.wait_seed,
            shap_sample_size=config.wait_shap_sample_size,
            track_experiment=config.track_experiment,
            mlflow_tracking_uri=tracking_uri,
        ),
        project_root / config.wait_output_dir,
    )
    arrivals = train_arrival_from_dataset(
        dataset,
        ArrivalTrainingConfig(
            model_version=config.arrival_model_version,
            random_seed=config.arrival_seed,
            backtest_windows=config.forecast_backtest_windows,
            track_experiment=config.track_experiment,
            mlflow_tracking_uri=tracking_uri,
        ),
        project_root / config.arrival_output_dir,
    )
    no_show = train_no_show_model(
        dataset,
        NoShowTrainingConfig(
            model_version=config.no_show_model_version,
            random_seed=config.no_show_seed,
            reminder_capacity_fraction=config.reminder_capacity_fraction,
            reminder_effectiveness=config.reminder_effectiveness,
            maximum_ece=config.maximum_ece,
            track_experiment=config.track_experiment,
            mlflow_tracking_uri=tracking_uri,
        ),
        project_root / config.no_show_output_dir,
    )
    occupancy = train_occupancy_model(
        build_occupancy_target(dataset),
        OccupancyTrainingConfig(
            model_version=config.occupancy_model_version,
            random_seed=config.occupancy_seed,
            alert_threshold=config.occupancy_alert_threshold,
            backtest_windows=config.forecast_backtest_windows,
            track_experiment=config.track_experiment,
            mlflow_tracking_uri=tracking_uri,
        ),
        project_root / config.occupancy_output_dir,
    )
    appointments = dataset["appointments"].sort_values("booking_created_at")
    split_at = int(len(appointments) * 0.7)
    reference, current = appointments.iloc[:split_at], appointments.iloc[split_at:]
    report = build_monitoring_report(
        reference,
        current,
        {},
        {},
        config.monitoring_columns,
        config.monitoring_thresholds,
    )
    report_path = project_root / config.monitoring_report_path
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return DemoArtifacts(
        wait.artifact_path,
        arrivals.artifact_path,
        no_show.artifact_path,
        occupancy.artifact_path,
        report_path,
    )


def _demo_environment(config: DemoConfig, secret_path: Path) -> dict[str, str]:
    stored: dict[str, str] = {}
    if secret_path.is_file():
        try:
            for line in secret_path.read_text(encoding="utf-8").splitlines():
                key, separator, value = line.partition("=")
                if separator and value:
                    stored[key] = value
        except OSError as exc:
            raise DemoSetupError("Cannot read the local demo secret file") from exc
    environment = os.environ.copy()
    values = {
        "POSTGRES_USER": environment.get("POSTGRES_USER")
        or stored.get("POSTGRES_USER")
        or config.postgres_user,
        "POSTGRES_DB": environment.get("POSTGRES_DB")
        or stored.get("POSTGRES_DB")
        or config.postgres_database,
        "POSTGRES_PASSWORD": environment.get("POSTGRES_PASSWORD")
        or stored.get("POSTGRES_PASSWORD")
        or token_urlsafe(32),
        "AROGYAFLOW_API_KEY": environment.get("AROGYAFLOW_API_KEY")
        or stored.get("AROGYAFLOW_API_KEY")
        or token_urlsafe(32),
    }
    try:
        secret_path.write_text(
            "".join(f"{key}={value}\n" for key, value in values.items()), encoding="utf-8"
        )
        secret_path.chmod(0o600)
    except OSError as exc:
        raise DemoSetupError("Cannot write the local demo secret file") from exc
    environment.update(values)
    return environment


def _wait_for_mlflow(url: str, timeout_seconds: float = 60) -> None:
    deadline = monotonic() + timeout_seconds
    while monotonic() < deadline:
        try:
            with urlopen(url, timeout=2) as response:  # noqa: S310 - versioned local demo URL
                if response.status < 500:
                    return
        except (OSError, TimeoutError, URLError):
            sleep(1)
    raise DemoSetupError("MLflow did not become ready before the timeout")


def launch_demo(config: DemoConfig, project_root: Path = Path(".")) -> None:
    environment = _demo_environment(config, project_root / ".arogyaflow-demo.env")
    infrastructure = ["docker", "compose", "up", "-d", "db"]
    if config.track_experiment:
        infrastructure.append("mlflow")
    try:
        subprocess.run(infrastructure, cwd=project_root, env=environment, check=True)
        if config.track_experiment:
            _wait_for_mlflow(str(config.mlflow_health_url))
        prepare_demo(config, project_root)
        subprocess.run(
            ["docker", "compose", "up", "--build"],
            cwd=project_root,
            env=environment,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise DemoSetupError("Docker demo stack failed") from exc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config/demo.json"))
    args = parser.parse_args()
    launch_demo(load_demo_config(args.config))


if __name__ == "__main__":
    main()
