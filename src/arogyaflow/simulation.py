import argparse
import json
import math
import random
from collections.abc import Generator
from enum import StrEnum
from pathlib import Path

import numpy as np
import simpy
from pydantic import BaseModel, ConfigDict, Field

from arogyaflow.exceptions import (
    RecommendationConstraintError,
    SimulationConfigurationError,
)


class SimulationScenario(StrEnum):
    BASELINE = "baseline"
    DOCTOR_SHORTAGE = "doctor_shortage"
    ADDITIONAL_STAFFING = "additional_staffing"


class SimulationConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    random_seed: int
    duration_minutes: int = Field(gt=0, le=10_080)
    arrivals_per_hour: float = Field(gt=0)
    doctors: int = Field(gt=0)
    rooms: int = Field(gt=0)
    mean_service_minutes: float = Field(gt=0)


class SimulationConstraints(BaseModel):
    model_config = ConfigDict(frozen=True)

    max_doctors: int = Field(gt=0)
    max_rooms: int = Field(gt=0)
    minimum_p90_improvement_minutes: float = Field(ge=0)


class SimulationResult(BaseModel):
    scenario: SimulationScenario
    doctors: int
    rooms: int
    arrivals: int
    completed: int
    mean_wait_minutes: float
    p50_wait_minutes: float
    p90_wait_minutes: float
    max_queue_length: int
    resource_utilization: float
    invariants_passed: bool


class FlowRecommendation(BaseModel):
    recommended_scenario: SimulationScenario
    additional_doctors: int
    expected_p90_wait_reduction_minutes: float
    reason: str
    human_approval_required: bool = True
    automatically_applied: bool = False


class ScenarioComparison(BaseModel):
    results: list[SimulationResult]
    recommendation: FlowRecommendation
    assumptions: list[str]


class SimulationRequest(BaseModel):
    base: SimulationConfig
    constraints: SimulationConstraints


def _scenario_config(
    base: SimulationConfig,
    scenario: SimulationScenario,
    constraints: SimulationConstraints,
) -> SimulationConfig:
    if base.doctors > constraints.max_doctors or base.rooms > constraints.max_rooms:
        raise SimulationConfigurationError("Baseline resources exceed configured limits")
    if scenario == SimulationScenario.DOCTOR_SHORTAGE:
        return base.model_copy(update={"doctors": max(1, base.doctors - 1)})
    if scenario == SimulationScenario.ADDITIONAL_STAFFING:
        return base.model_copy(update={"doctors": min(constraints.max_doctors, base.doctors + 1)})
    return base


def simulate_flow(config: SimulationConfig, scenario: SimulationScenario) -> SimulationResult:
    arrival_rng = random.Random(f"{config.random_seed}:arrivals")
    service_rng = random.Random(f"{config.random_seed}:services")
    environment = simpy.Environment()
    doctors = simpy.Resource(environment, capacity=config.doctors)
    rooms = simpy.Resource(environment, capacity=config.rooms)
    waits: list[float] = []
    services: list[float] = []
    arrivals = completed = max_queue = 0

    def patient() -> Generator[simpy.events.Event, object, None]:
        nonlocal completed, max_queue
        entered = environment.now
        with doctors.request() as doctor_request, rooms.request() as room_request:
            max_queue = max(max_queue, len(doctors.queue), len(rooms.queue))
            yield doctor_request & room_request
            waits.append(environment.now - entered)
            sigma = 0.35
            service = service_rng.lognormvariate(
                math.log(config.mean_service_minutes) - sigma**2 / 2, sigma
            )
            services.append(service)
            yield environment.timeout(service)
            completed += 1

    def arrivals_process() -> Generator[simpy.events.Event, object, None]:
        nonlocal arrivals
        rate_per_minute = config.arrivals_per_hour / 60
        while True:
            delay = arrival_rng.expovariate(rate_per_minute)
            if environment.now + delay > config.duration_minutes:
                return
            yield environment.timeout(delay)
            arrivals += 1
            environment.process(patient())

    environment.process(arrivals_process())
    environment.run()
    if not waits:
        raise SimulationConfigurationError("Simulation produced no patients")
    elapsed = max(environment.now, float(config.duration_minutes))
    utilization = sum(services) / max(config.doctors * elapsed, 1)
    invariants = completed == arrivals and min(waits) >= 0 and 0 <= utilization <= 1
    if not invariants:
        raise SimulationConfigurationError("Simulation invariants failed")
    return SimulationResult(
        scenario=scenario,
        doctors=config.doctors,
        rooms=config.rooms,
        arrivals=arrivals,
        completed=completed,
        mean_wait_minutes=float(np.mean(waits)),
        p50_wait_minutes=float(np.quantile(waits, 0.5)),
        p90_wait_minutes=float(np.quantile(waits, 0.9)),
        max_queue_length=max_queue,
        resource_utilization=utilization,
        invariants_passed=True,
    )


def compare_scenarios(
    base: SimulationConfig, constraints: SimulationConstraints
) -> ScenarioComparison:
    scenarios = list(SimulationScenario)
    results = [
        simulate_flow(_scenario_config(base, scenario, constraints), scenario)
        for scenario in scenarios
    ]
    by_scenario = {result.scenario: result for result in results}
    baseline = by_scenario[SimulationScenario.BASELINE]
    additional = by_scenario[SimulationScenario.ADDITIONAL_STAFFING]
    improvement = max(0.0, baseline.p90_wait_minutes - additional.p90_wait_minutes)
    extra_doctors = additional.doctors - baseline.doctors
    if extra_doctors < 0:
        raise RecommendationConstraintError("Additional staffing cannot reduce doctor count")
    if extra_doctors and improvement >= constraints.minimum_p90_improvement_minutes:
        recommended = SimulationScenario.ADDITIONAL_STAFFING
        reason = "Additional staffing meets the configured P90 wait-improvement threshold."
    else:
        recommended = SimulationScenario.BASELINE
        reason = "Keep baseline staffing; constrained improvement is below the required threshold."
    return ScenarioComparison(
        results=results,
        recommendation=FlowRecommendation(
            recommended_scenario=recommended,
            additional_doctors=extra_doctors
            if recommended == SimulationScenario.ADDITIONAL_STAFFING
            else 0,
            expected_p90_wait_reduction_minutes=improvement,
            reason=reason,
        ),
        assumptions=[
            "Arrivals follow a stationary Poisson process.",
            "Service times follow a log-normal distribution.",
            "Recommendations require human approval and are never auto-applied.",
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--duration", required=True, type=int)
    parser.add_argument("--arrivals-per-hour", required=True, type=float)
    parser.add_argument("--doctors", required=True, type=int)
    parser.add_argument("--rooms", required=True, type=int)
    parser.add_argument("--service-minutes", required=True, type=float)
    parser.add_argument("--max-doctors", required=True, type=int)
    parser.add_argument("--max-rooms", required=True, type=int)
    parser.add_argument("--minimum-improvement", required=True, type=float)
    parser.add_argument("--output", type=Path, default=Path("reports/simulation.json"))
    args = parser.parse_args()
    comparison = compare_scenarios(
        SimulationConfig(
            random_seed=args.seed,
            duration_minutes=args.duration,
            arrivals_per_hour=args.arrivals_per_hour,
            doctors=args.doctors,
            rooms=args.rooms,
            mean_service_minutes=args.service_minutes,
        ),
        SimulationConstraints(
            max_doctors=args.max_doctors,
            max_rooms=args.max_rooms,
            minimum_p90_improvement_minutes=args.minimum_improvement,
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(comparison.model_dump(mode="json"), indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
