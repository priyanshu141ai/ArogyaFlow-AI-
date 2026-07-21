import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, Request, Response, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from arogyaflow import __version__
from arogyaflow.config import get_settings
from arogyaflow.exceptions import ArogyaFlowError
from arogyaflow.identifiers import new_identifier
from arogyaflow.logging import bind_request_id, configure_logging, reset_request_id
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
    yield


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
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
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
    async def ready() -> HealthStatus:
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
    async def simulation_comparison(payload: SimulationRequest) -> ScenarioComparison:
        return compare_scenarios(payload.base, payload.constraints)

    return application


app = create_app()
