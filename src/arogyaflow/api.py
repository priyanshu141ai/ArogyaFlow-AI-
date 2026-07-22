import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from time import perf_counter
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
from arogyaflow.monitoring import MonitoringReport, load_monitoring_report
from arogyaflow.observability import ApplicationMetrics
from arogyaflow.security import RateLimiter, is_protected_path, valid_api_key
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
    metrics = ApplicationMetrics()
    app.state.metrics = metrics
    app.state.rate_limiter = RateLimiter(settings.rate_limit_requests_per_minute)
    database = PostgresDatabase(settings) if settings.database_url else None
    if database:
        database.open()
        metrics.database_ready.set(1)
    else:
        metrics.database_ready.set(0)
    model_paths = {
        "wait_time": settings.wait_model_path,
        "arrivals": settings.arrival_model_path,
        "no_show": settings.no_show_model_path,
        "occupancy": settings.occupancy_model_path,
    }
    for name, path in model_paths.items():
        metrics.model_available.labels(name).set(int(path is not None and path.is_file()))
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
        started_at = perf_counter()
        request_id = request.headers.get("X-Request-ID") or new_identifier("req")
        request.state.request_id = request_id
        token = bind_request_id(request_id)
        response: Response | None = None
        try:
            settings = request.app.state.settings
            content_length = request.headers.get("Content-Length")
            try:
                oversized = content_length is not None and (
                    int(content_length) < 0 or int(content_length) > settings.max_request_bytes
                )
            except ValueError:
                response = JSONResponse(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    content={
                        "code": "InvalidContentLength",
                        "message": "Content-Length must be a non-negative integer",
                        "request_id": request_id,
                    },
                )
            else:
                if oversized:
                    response = JSONResponse(
                        status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                        content={
                            "code": "RequestTooLarge",
                            "message": "Request body exceeds the configured limit",
                            "request_id": request_id,
                        },
                    )
            if response is None and is_protected_path(request.url.path):
                client = request.client.host if request.client else "unknown"
                limiter = cast(RateLimiter, request.app.state.rate_limiter)
                if not limiter.allow(client):
                    response = JSONResponse(
                        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                        headers={"Retry-After": "60"},
                        content={
                            "code": "RateLimitExceeded",
                            "message": "Request rate limit exceeded",
                            "request_id": request_id,
                        },
                    )
                else:
                    expected = settings.api_key.get_secret_value() if settings.api_key else None
                    if not valid_api_key(expected, request.headers.get("X-API-Key")):
                        response = JSONResponse(
                            status_code=status.HTTP_401_UNAUTHORIZED,
                            content={
                                "code": "AuthenticationRequired",
                                "message": "A valid API key is required",
                                "request_id": request_id,
                            },
                        )
            if response is None:
                response = await call_next(request)
            response.headers.update(
                {
                    "X-Request-ID": request_id,
                    "X-Content-Type-Options": "nosniff",
                    "X-Frame-Options": "DENY",
                    "Referrer-Policy": "no-referrer",
                    "Cache-Control": "no-store",
                }
            )
            return response
        except Exception:
            logger.exception("request_failed")
            raise
        finally:
            route = request.scope.get("route")
            route_name = getattr(route, "path", "blocked")
            metrics = cast(ApplicationMetrics, request.app.state.metrics)
            metrics.observe_request(
                request.method,
                route_name,
                response.status_code if response else status.HTTP_500_INTERNAL_SERVER_ERROR,
                perf_counter() - started_at,
            )
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
        metrics = cast(ApplicationMetrics, request.app.state.metrics)
        if database:
            try:
                database.ping()
            except PersistenceError:
                metrics.database_ready.set(0)
                raise
            metrics.database_ready.set(1)
        return HealthStatus(status="ready")

    @application.get("/metrics", include_in_schema=False)
    async def metrics(request: Request) -> Response:
        application_metrics = cast(ApplicationMetrics, request.app.state.metrics)
        return Response(
            content=application_metrics.render(),
            headers={"Content-Type": application_metrics.content_type},
        )

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

    @application.get("/v1/monitoring/report", response_model=MonitoringReport)
    def monitoring_report(request: Request) -> MonitoringReport:
        settings = request.app.state.settings
        return load_monitoring_report(settings.monitoring_report_path)

    return application


app = create_app()
