import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Literal, cast

from fastapi import FastAPI, Query, Request, Response, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from arogyaflow import __version__
from arogyaflow.config import get_settings
from arogyaflow.database import PostgresDatabase
from arogyaflow.exceptions import (
    ArogyaFlowError,
    ConfigurationError,
    ModelArtifactError,
    PersistenceError,
)
from arogyaflow.identifiers import new_identifier
from arogyaflow.logging import bind_request_id, configure_logging, reset_request_id
from arogyaflow.serving import (
    ArrivalForecastResponse,
    ForecastRequest,
    NoShowPredictionRequest,
    NoShowPredictionResponse,
    OccupancyForecastResponse,
    PredictionApplication,
    RecentPredictionsResponse,
    WaitTimePredictionRequest,
    WaitTimePredictionResponse,
)
from arogyaflow.simulation import ScenarioComparison, SimulationRequest, compare_scenarios

logger = logging.getLogger(__name__)


class HealthStatus(BaseModel):
    status: Literal["live", "ready"]


class ServiceMetadata(BaseModel):
    service_version: str
    schema_versions: dict[str, str]
    model_versions: dict[str, str | None]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(settings.log_level)
    app.state.settings = settings
    database = PostgresDatabase(settings) if settings.database_url else None
    if database:
        database.open()
    app.state.database = database
    app.state.predictions = PredictionApplication(settings, database)
    try:
        yield
    finally:
        if database:
            database.close()


def _predictions(request: Request) -> PredictionApplication:
    return cast(PredictionApplication, request.app.state.predictions)


def create_app() -> FastAPI:
    application = FastAPI(title="ArogyaFlow AI", lifespan=lifespan)

    @application.middleware("http")
    async def request_context(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request_id = request.headers.get("X-Request-ID") or new_identifier("req")
        request.state.request_id = request_id
        token = bind_request_id(request_id)
        try:
            response = await call_next(request)
            response.headers["X-Request-ID"] = request_id
            return response
        except Exception:
            logger.exception("request_failed")
            raise
        finally:
            reset_request_id(token)

    @application.exception_handler(ArogyaFlowError)
    async def domain_error(request: Request, exc: ArogyaFlowError) -> JSONResponse:
        unavailable = isinstance(exc, (ConfigurationError, ModelArtifactError, PersistenceError))
        return JSONResponse(
            status_code=(
                status.HTTP_503_SERVICE_UNAVAILABLE
                if unavailable
                else status.HTTP_422_UNPROCESSABLE_CONTENT
            ),
            content={
                "code": exc.__class__.__name__,
                "message": str(exc),
                "request_id": getattr(request.state, "request_id", None),
            },
        )

    @application.get("/health/live", response_model=HealthStatus)
    async def live() -> HealthStatus:
        return HealthStatus(status="live")

    @application.get("/health/ready", response_model=HealthStatus)
    def ready(request: Request) -> HealthStatus:
        database = cast(PostgresDatabase | None, request.app.state.database)
        if database:
            database.ping()
        return HealthStatus(status="ready")

    @application.get("/v1/meta", response_model=ServiceMetadata)
    async def metadata(request: Request) -> ServiceMetadata:
        settings = request.app.state.settings
        return ServiceMetadata(
            service_version=__version__,
            schema_versions={
                "wait_time": "1.0",
                "arrivals": "1.0",
                "no_show": "1.0",
                "occupancy": "1.0",
                "simulation": "1.0",
            },
            model_versions={
                "wait_time": settings.wait_model_version,
                "arrivals": settings.arrival_model_version,
                "no_show": settings.no_show_model_version,
                "occupancy": settings.occupancy_model_version,
            },
        )

    @application.post("/v1/simulations/compare", response_model=ScenarioComparison)
    def simulation_comparison(payload: SimulationRequest) -> ScenarioComparison:
        return compare_scenarios(payload.base, payload.constraints)

    @application.post("/v1/predictions/wait-time", response_model=WaitTimePredictionResponse)
    def wait_time_prediction(
        payload: WaitTimePredictionRequest, request: Request
    ) -> WaitTimePredictionResponse:
        return _predictions(request).predict_wait_time(payload, request.state.request_id)

    @application.post("/v1/forecasts/arrivals", response_model=ArrivalForecastResponse)
    def arrival_forecast(payload: ForecastRequest, request: Request) -> ArrivalForecastResponse:
        return _predictions(request).forecast_arrivals(payload, request.state.request_id)

    @application.post("/v1/predictions/no-show", response_model=NoShowPredictionResponse)
    def no_show_prediction(
        payload: NoShowPredictionRequest, request: Request
    ) -> NoShowPredictionResponse:
        return _predictions(request).predict_no_show(payload, request.state.request_id)

    @application.post("/v1/forecasts/occupancy", response_model=OccupancyForecastResponse)
    def occupancy_forecast(payload: ForecastRequest, request: Request) -> OccupancyForecastResponse:
        return _predictions(request).forecast_occupancy(payload, request.state.request_id)

    @application.get("/v1/predictions/recent", response_model=RecentPredictionsResponse)
    def recent_predictions(
        request: Request, limit: int = Query(default=20, ge=1, le=100)
    ) -> RecentPredictionsResponse:
        return _predictions(request).recent_predictions(limit)

    return application


app = create_app()
