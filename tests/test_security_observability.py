from collections.abc import Generator

import pytest
from fastapi.testclient import TestClient

from arogyaflow.api import app
from arogyaflow.config import get_settings
from arogyaflow.security import RateLimiter


@pytest.fixture(autouse=True)
def clear_settings() -> Generator[None]:
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_api_key_security_headers_and_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AROGYAFLOW_ENVIRONMENT", "production")
    monkeypatch.setenv("AROGYAFLOW_API_KEY", "test-api-key")
    monkeypatch.delenv("AROGYAFLOW_DATABASE_URL", raising=False)
    with TestClient(app) as client:
        denied = client.get("/v1/meta")
        accepted = client.get("/v1/meta", headers={"X-API-Key": "test-api-key"})
        metrics = client.get("/metrics", headers={"X-API-Key": "test-api-key"})
        live = client.get("/health/live")
    assert denied.status_code == 401
    assert accepted.status_code == 200
    assert accepted.headers["X-Content-Type-Options"] == "nosniff"
    assert live.status_code == 200
    assert "arogyaflow_http_requests_total" in metrics.text
    assert 'route="/v1/meta",status="200"' in metrics.text


def test_request_size_and_rate_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AROGYAFLOW_MAX_REQUEST_BYTES", "1024")
    monkeypatch.setenv("AROGYAFLOW_RATE_LIMIT_REQUESTS_PER_MINUTE", "1")
    monkeypatch.delenv("AROGYAFLOW_DATABASE_URL", raising=False)
    with TestClient(app) as client:
        oversized = client.post(
            "/v1/simulations/compare",
            content=b"{}",
            headers={"Content-Length": "1025"},
        )
        first = client.get("/v1/meta")
        limited = client.get("/v1/meta")
    assert oversized.status_code == 413
    assert first.status_code == 200
    assert limited.status_code == 429
    assert limited.headers["Retry-After"] == "60"


def test_rate_limiter_window() -> None:
    limiter = RateLimiter(limit=2, window_seconds=10)
    assert limiter.allow("client", now=0)
    assert limiter.allow("client", now=1)
    assert not limiter.allow("client", now=2)
    assert limiter.allow("client", now=11)
