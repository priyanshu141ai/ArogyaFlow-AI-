import json
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token

from arogyaflow.time import utc_now

_request_id: ContextVar[str | None] = ContextVar("request_id", default=None)
_job_id: ContextVar[str | None] = ContextVar("job_id", default=None)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        event: dict[str, object] = {
            "timestamp": utc_now().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if request_id := _request_id.get():
            event["request_id"] = request_id
        if job_id := _job_id.get():
            event["job_id"] = job_id
        if record.exc_info:
            event["exception"] = self.formatException(record.exc_info)
        return json.dumps(event, default=str)


def configure_logging(level: str) -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    logging.basicConfig(level=level, handlers=[handler], force=True)


def bind_request_id(value: str) -> Token[str | None]:
    return _request_id.set(value)


def reset_request_id(token: Token[str | None]) -> None:
    _request_id.reset(token)


@contextmanager
def job_log_context(job_id: str) -> Iterator[None]:
    token = _job_id.set(job_id)
    try:
        yield
    finally:
        _job_id.reset(token)
