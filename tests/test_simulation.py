from fastapi.testclient import TestClient

from arogyaflow.api import app
from arogyaflow.simulation import (
    SimulationConfig,
    SimulationConstraints,
    SimulationScenario,
    compare_scenarios,
)


def _request() -> tuple[SimulationConfig, SimulationConstraints]:
    return (
        SimulationConfig(
            random_seed=29,
            duration_minutes=480,
            arrivals_per_hour=8,
            doctors=3,
            rooms=4,
            mean_service_minutes=20,
        ),
        SimulationConstraints(
            max_doctors=4,
            max_rooms=4,
            minimum_p90_improvement_minutes=1,
        ),
    )


def test_simulation_is_deterministic_and_requires_approval() -> None:
    base, constraints = _request()
    first = compare_scenarios(base, constraints)
    second = compare_scenarios(base, constraints)
    assert first == second
    assert all(result.invariants_passed for result in first.results)
    assert len({result.arrivals for result in first.results}) == 1
    assert first.recommendation.human_approval_required is True
    assert first.recommendation.automatically_applied is False
    assert {result.scenario for result in first.results} == set(SimulationScenario)


def test_simulation_api_and_metadata() -> None:
    base, constraints = _request()
    with TestClient(app) as client:
        response = client.post(
            "/v1/simulations/compare",
            json={
                "base": base.model_dump(mode="json"),
                "constraints": constraints.model_dump(mode="json"),
            },
        )
        metadata = client.get("/v1/meta")
    assert response.status_code == 200
    assert response.json()["recommendation"]["human_approval_required"] is True
    assert metadata.json()["schema_versions"]["simulation"] == "1.0"


def test_simulation_api_returns_correlated_domain_error() -> None:
    base, constraints = _request()
    payload = {
        "base": base.model_dump(mode="json"),
        "constraints": constraints.model_copy(update={"max_doctors": 2}).model_dump(mode="json"),
    }
    with TestClient(app) as client:
        response = client.post(
            "/v1/simulations/compare", json=payload, headers={"X-Request-ID": "req-test"}
        )
    assert response.status_code == 422
    assert response.json() == {
        "code": "SimulationConfigurationError",
        "message": "Baseline resources exceed configured limits",
        "request_id": "req-test",
    }
