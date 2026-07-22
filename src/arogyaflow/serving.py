from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Literal, TypeVar

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from arogyaflow.arrival_forecasting import (
    ArrivalArtifact,
    ArrivalForecast,
    forecast_arrivals,
    load_arrival_artifact,
)
from arogyaflow.bed_occupancy import (
    OccupancyArtifact,
    OccupancyForecast,
    forecast_occupancy,
    load_occupancy_artifact,
)
from arogyaflow.config import Settings
from arogyaflow.database import PredictionRecord, PredictionStore, PredictionType
from arogyaflow.exceptions import ConfigurationError, ModelArtifactError
from arogyaflow.no_show import NoShowArtifact, predict_no_show
from arogyaflow.no_show import load_artifact as load_no_show_artifact
from arogyaflow.time import utc_now
from arogyaflow.wait_time import WaitTimeArtifact, predict_wait_time
from arogyaflow.wait_time import load_artifact as load_wait_time_artifact


class ApiModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class PersistedResponse(ApiModel):
    request_id: str
    created_at: datetime
    persisted: bool


ResponseModel = TypeVar("ResponseModel", bound=PersistedResponse)


class WaitTimePredictionRequest(ApiModel):
    department_id: str = Field(min_length=1)
    appointment_type: Literal["new", "follow_up"]
    priority_type: Literal["routine", "urgent"]
    reminder_sent: bool
    weekday: int = Field(ge=0, le=6)
    hour: int = Field(ge=0, le=23)
    queue_length: int = Field(ge=0)
    available_doctors: int = Field(ge=0)
    available_rooms: int = Field(ge=0)


class WaitTimePredictionResponse(PersistedResponse):
    model_version: str
    schema_version: str
    predicted_wait_minutes: float
    lower_wait_minutes: float
    upper_wait_minutes: float


class NoShowPredictionRequest(ApiModel):
    department_id: str = Field(min_length=1)
    appointment_type: Literal["new", "follow_up"]
    priority_type: Literal["routine", "urgent"]
    booking_lead_hours: float = Field(ge=0)
    scheduled_weekday: int = Field(ge=0, le=6)
    scheduled_hour: int = Field(ge=0, le=23)
    historical_appointments: int = Field(ge=0)
    historical_no_shows: int = Field(ge=0)
    historical_late_arrivals: int = Field(ge=0)
    historical_reminders: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_history_counts(self) -> "NoShowPredictionRequest":
        for value in (
            self.historical_no_shows,
            self.historical_late_arrivals,
            self.historical_reminders,
        ):
            if value > self.historical_appointments:
                raise ValueError("Historical event counts cannot exceed appointments")
        return self

    def inference_payload(self) -> dict[str, object]:
        payload = self.model_dump()
        appointments = self.historical_appointments
        payload["historical_no_show_rate"] = (
            self.historical_no_shows / appointments if appointments else 0.0
        )
        return payload


class NoShowPredictionResponse(PersistedResponse):
    model_version: str
    schema_version: str
    no_show_probability: float
    reminder_priority: bool
    automatic_cancellation: Literal[False] = False


class ForecastRequest(ApiModel):
    horizon_hours: Literal[6, 12, 24]


class ArrivalForecastResponse(PersistedResponse):
    model_version: str
    schema_version: str
    forecast: ArrivalForecast


class OccupancyForecastResponse(PersistedResponse):
    model_version: str
    schema_version: str
    forecast: OccupancyForecast


class RecentPredictionsResponse(ApiModel):
    records: list[PredictionRecord]


def _required_artifact(path: Path | None, model_name: str) -> Path:
    if path is None:
        raise ConfigurationError(f"{model_name} artifact path is not configured")
    if not path.is_file():
        raise ModelArtifactError(f"{model_name} artifact does not exist")
    return path


@lru_cache(maxsize=16)
def _wait_artifact(path: Path) -> WaitTimeArtifact:
    return load_wait_time_artifact(path)


@lru_cache(maxsize=16)
def _arrival_artifact(path: Path) -> ArrivalArtifact:
    return load_arrival_artifact(path)


@lru_cache(maxsize=16)
def _no_show_artifact(path: Path) -> NoShowArtifact:
    return load_no_show_artifact(path)


@lru_cache(maxsize=16)
def _occupancy_artifact(path: Path) -> OccupancyArtifact:
    return load_occupancy_artifact(path)


class PredictionApplication:
    def __init__(self, settings: Settings, store: PredictionStore | None) -> None:
        self._settings = settings
        self._store = store

    def _persist(
        self,
        *,
        prediction_type: PredictionType,
        request_id: str,
        request_payload: dict[str, object],
        response: ResponseModel,
        model_version: str,
        schema_version: str,
    ) -> ResponseModel:
        if self._store is None:
            return response
        persisted = response.model_copy(update={"persisted": True})
        self._store.insert_prediction(
            request_id=request_id,
            prediction_type=prediction_type,
            model_version=model_version,
            schema_version=schema_version,
            request_payload=request_payload,
            response_payload=persisted.model_dump(mode="json"),
            created_at=persisted.created_at,
        )
        return persisted

    def predict_wait_time(
        self, payload: WaitTimePredictionRequest, request_id: str
    ) -> WaitTimePredictionResponse:
        artifact = _wait_artifact(_required_artifact(self._settings.wait_model_path, "Wait-time"))
        prediction = predict_wait_time(artifact, pd.DataFrame([payload.model_dump()])).iloc[0]
        response = WaitTimePredictionResponse(
            request_id=request_id,
            created_at=utc_now(),
            persisted=False,
            model_version=artifact.model_version,
            schema_version=artifact.schema_version,
            predicted_wait_minutes=float(prediction["predicted_wait_minutes"]),
            lower_wait_minutes=float(prediction["lower_wait_minutes"]),
            upper_wait_minutes=float(prediction["upper_wait_minutes"]),
        )
        return self._persist(
            prediction_type=PredictionType.WAIT_TIME,
            request_id=request_id,
            request_payload=payload.model_dump(),
            response=response,
            model_version=artifact.model_version,
            schema_version=artifact.schema_version,
        )

    def forecast_arrivals(
        self, payload: ForecastRequest, request_id: str
    ) -> ArrivalForecastResponse:
        artifact = _arrival_artifact(
            _required_artifact(self._settings.arrival_model_path, "Arrival")
        )
        forecast = forecast_arrivals(artifact, payload.horizon_hours)
        response = ArrivalForecastResponse(
            request_id=request_id,
            created_at=utc_now(),
            persisted=False,
            model_version=artifact.model_version,
            schema_version=artifact.schema_version,
            forecast=forecast,
        )
        return self._persist(
            prediction_type=PredictionType.ARRIVALS,
            request_id=request_id,
            request_payload=payload.model_dump(),
            response=response,
            model_version=artifact.model_version,
            schema_version=artifact.schema_version,
        )

    def predict_no_show(
        self, payload: NoShowPredictionRequest, request_id: str
    ) -> NoShowPredictionResponse:
        artifact = _no_show_artifact(
            _required_artifact(self._settings.no_show_model_path, "No-show")
        )
        prediction = predict_no_show(artifact, pd.DataFrame([payload.inference_payload()])).iloc[0]
        response = NoShowPredictionResponse(
            request_id=request_id,
            created_at=utc_now(),
            persisted=False,
            model_version=artifact.model_version,
            schema_version=artifact.schema_version,
            no_show_probability=float(prediction["no_show_probability"]),
            reminder_priority=bool(prediction["reminder_priority"]),
        )
        return self._persist(
            prediction_type=PredictionType.NO_SHOW,
            request_id=request_id,
            request_payload=payload.model_dump(),
            response=response,
            model_version=artifact.model_version,
            schema_version=artifact.schema_version,
        )

    def forecast_occupancy(
        self, payload: ForecastRequest, request_id: str
    ) -> OccupancyForecastResponse:
        artifact = _occupancy_artifact(
            _required_artifact(self._settings.occupancy_model_path, "Occupancy")
        )
        forecast = forecast_occupancy(artifact, payload.horizon_hours)
        response = OccupancyForecastResponse(
            request_id=request_id,
            created_at=utc_now(),
            persisted=False,
            model_version=artifact.model_version,
            schema_version=artifact.schema_version,
            forecast=forecast,
        )
        return self._persist(
            prediction_type=PredictionType.OCCUPANCY,
            request_id=request_id,
            request_payload=payload.model_dump(),
            response=response,
            model_version=artifact.model_version,
            schema_version=artifact.schema_version,
        )

    def recent_predictions(self, limit: int) -> RecentPredictionsResponse:
        if self._store is None:
            raise ConfigurationError("Prediction persistence is not configured")
        return RecentPredictionsResponse(records=self._store.recent_predictions(limit))
