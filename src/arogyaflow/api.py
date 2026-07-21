import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, Request, Response
from pydantic import BaseModel

from arogyaflow.config import get_settings
from arogyaflow.identifiers import new_identifier
from arogyaflow.logging import bind_request_id, configure_logging, reset_request_id

logger = logging.getLogger(__name__)


class HealthStatus(BaseModel):
    status: Literal["live", "ready"]


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

    @application.get("/health/live", response_model=HealthStatus)
    async def live() -> HealthStatus:
        return HealthStatus(status="live")

    @application.get("/health/ready", response_model=HealthStatus)
    async def ready() -> HealthStatus:
        return HealthStatus(status="ready")

    return application


app = create_app()
