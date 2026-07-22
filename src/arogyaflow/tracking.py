import logging
import math
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from arogyaflow.exceptions import ExperimentTrackingError

logger = logging.getLogger(__name__)


class ModelKind(StrEnum):
    WAIT_TIME = "wait_time"
    ARRIVALS = "arrivals"
    NO_SHOW = "no_show"
    OCCUPANCY = "occupancy"


class TrackingOptions(BaseModel):
    model_config = ConfigDict(frozen=True)

    track_experiment: bool = True
    mlflow_tracking_uri: str | None = None
    experiment_name: str = Field(min_length=1)
    registered_model_name: str = Field(min_length=1)


@dataclass(frozen=True)
class TrackedRun:
    run_id: str
    model_uri: str
    registered_model_version: int | None
    tracking_uri: str


def _python_model(model_kind: ModelKind) -> Any:
    import mlflow

    class Model(mlflow.pyfunc.PythonModel):  # type: ignore[name-defined,misc]
        def __init__(self) -> None:
            self.artifact: Any = None

        def load_context(self, context: Any) -> None:
            artifact_path = Path(context.artifacts["artifact"])
            if model_kind == ModelKind.WAIT_TIME:
                from arogyaflow.wait_time import load_artifact as load_wait_time_artifact

                self.artifact = load_wait_time_artifact(artifact_path)
            elif model_kind == ModelKind.ARRIVALS:
                from arogyaflow.arrival_forecasting import load_arrival_artifact

                self.artifact = load_arrival_artifact(artifact_path)
            elif model_kind == ModelKind.NO_SHOW:
                from arogyaflow.no_show import load_artifact as load_no_show_artifact

                self.artifact = load_no_show_artifact(artifact_path)
            else:
                from arogyaflow.bed_occupancy import load_occupancy_artifact

                self.artifact = load_occupancy_artifact(artifact_path)

        def predict(
            self,
            context: Any,
            model_input: pd.DataFrame,
            params: dict[str, Any] | None = None,
        ) -> pd.DataFrame:
            del context, params
            if model_kind == ModelKind.WAIT_TIME:
                from arogyaflow.wait_time import predict_wait_time

                return predict_wait_time(self.artifact, model_input)
            if model_kind == ModelKind.NO_SHOW:
                from arogyaflow.no_show import predict_no_show

                return predict_no_show(self.artifact, model_input)
            if len(model_input) != 1 or "horizon_hours" not in model_input:
                raise ValueError("Forecast models require one horizon_hours row")
            horizon = int(model_input["horizon_hours"].iloc[0])
            if model_kind == ModelKind.ARRIVALS:
                from arogyaflow.arrival_forecasting import forecast_arrivals

                return pd.DataFrame(
                    point.model_dump(mode="json")
                    for point in forecast_arrivals(self.artifact, horizon).points
                )
            from arogyaflow.bed_occupancy import forecast_occupancy

            return pd.DataFrame(
                point.model_dump(mode="json")
                for point in forecast_occupancy(self.artifact, horizon).points
            )

    return Model()


def flatten_metrics(payload: dict[str, Any], prefix: str = "") -> dict[str, float]:
    metrics: dict[str, float] = {}
    for name, value in payload.items():
        key = f"{prefix}.{name}" if prefix else name
        if isinstance(value, dict):
            metrics.update(flatten_metrics(value, key))
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            number = float(value)
            if math.isfinite(number):
                metrics[key] = number
    return metrics


def log_training_run(
    config: TrackingOptions,
    *,
    model_kind: ModelKind,
    output_dir: Path,
    artifact_path: Path,
    report_path: Path,
    parameters: dict[str, str | int | float | bool],
    metrics: dict[str, float],
) -> TrackedRun | None:
    if not config.track_experiment:
        return None
    import mlflow

    tracking_uri = config.mlflow_tracking_uri or (
        f"sqlite:///{(output_dir / 'mlflow.db').resolve().as_posix()}"
    )
    try:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(config.experiment_name)
        with mlflow.start_run(run_name=str(parameters["model_version"])) as run:
            mlflow.log_params(parameters)
            mlflow.log_metrics(metrics)
            mlflow.log_artifact(str(report_path), artifact_path="evaluation")
            model_info = mlflow.pyfunc.log_model(
                name="model",
                python_model=_python_model(model_kind),
                artifacts={"artifact": str(artifact_path)},
                registered_model_name=config.registered_model_name,
            )
            registered_version = getattr(model_info, "registered_model_version", None)
            return TrackedRun(
                run_id=run.info.run_id,
                model_uri=model_info.model_uri,
                registered_model_version=(
                    int(registered_version) if registered_version is not None else None
                ),
                tracking_uri=tracking_uri,
            )
    except Exception as exc:
        logger.exception("mlflow_tracking_failed", extra={"model_kind": model_kind.value})
        raise ExperimentTrackingError("MLflow tracking or registry update failed") from exc
