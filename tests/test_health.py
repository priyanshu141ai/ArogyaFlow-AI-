from fastapi.testclient import TestClient

from arogyaflow.api import app


def test_health_endpoints() -> None:
    with TestClient(app) as client:
        assert client.get("/health/live").json() == {"status": "live"}
        assert client.get("/health/ready").json() == {"status": "ready"}


def test_request_id_is_propagated() -> None:
    with TestClient(app) as client:
        response = client.get("/health/live", headers={"X-Request-ID": "test-request"})
    assert response.headers["X-Request-ID"] == "test-request"
