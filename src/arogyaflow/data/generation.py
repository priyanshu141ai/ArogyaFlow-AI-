import math
import random
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import TypeAlias
from uuid import NAMESPACE_URL, uuid5

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

Dataset: TypeAlias = dict[str, pd.DataFrame]
GENERATOR_VERSION = "1.0.0"


class ScenarioName(StrEnum):
    NORMAL_WEEK = "normal_week"
    FESTIVAL_SURGE = "festival_surge"
    DOCTOR_SHORTAGE = "doctor_shortage"
    SEASONAL_INFECTION_SURGE = "seasonal_infection_surge"
    BED_CLOSURE = "bed_closure"
    SYSTEM_DATA_DELAY = "system_data_delay"
    COMBINED_STRESS_TEST = "combined_stress_test"


class ScenarioConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    scenario: ScenarioName
    seed: int
    start: datetime
    days: int = Field(default=7, ge=1, le=365)
    anomaly_rate: float = Field(default=0.0, ge=0.0, le=0.25)

    @model_validator(mode="after")
    def require_timezone(self) -> "ScenarioConfig":
        if self.start.tzinfo is None or self.start.utcoffset() is None:
            raise ValueError("start must be timezone-aware")
        return self


_PROFILES: dict[ScenarioName, tuple[float, float, float, float]] = {
    ScenarioName.NORMAL_WEEK: (1.0, 1.0, 1.0, 1.0),
    ScenarioName.FESTIVAL_SURGE: (1.7, 1.0, 1.0, 1.0),
    ScenarioName.DOCTOR_SHORTAGE: (1.0, 0.5, 1.0, 1.0),
    ScenarioName.SEASONAL_INFECTION_SURGE: (1.5, 0.9, 1.2, 1.0),
    ScenarioName.BED_CLOSURE: (1.0, 1.0, 0.65, 1.0),
    ScenarioName.SYSTEM_DATA_DELAY: (1.0, 1.0, 1.0, 8.0),
    ScenarioName.COMBINED_STRESS_TEST: (1.8, 0.5, 0.65, 8.0),
}
_DEPARTMENTS = (
    ("dep_emergency", "Emergency", 5.0, 4, 5),
    ("dep_general", "General Medicine", 3.5, 3, 4),
    ("dep_pediatrics", "Pediatrics", 2.5, 2, 3),
)


def _stable_id(kind: str, seed: int, ordinal: int) -> str:
    return f"{kind}_{uuid5(NAMESPACE_URL, f'arogyaflow:{seed}:{kind}:{ordinal}').hex}"


def _record_times(
    event_time: datetime, rng: random.Random, latency: float
) -> tuple[datetime, datetime]:
    recorded = event_time + timedelta(minutes=rng.uniform(0.1, latency))
    return recorded, recorded + timedelta(minutes=rng.uniform(0.1, latency))


def generate_dataset(config: ScenarioConfig) -> Dataset:
    rng = random.Random(config.seed)
    demand, doctor_factor, bed_factor, latency = _PROFILES[config.scenario]
    start = config.start.astimezone(UTC).replace(minute=0, second=0, microsecond=0)
    hospital_id = "hospital_demo"
    departments: list[dict[str, object]] = []
    doctors: list[dict[str, object]] = []
    beds: list[dict[str, object]] = []
    rosters: list[dict[str, object]] = []
    appointments: list[dict[str, object]] = []
    encounters: list[dict[str, object]] = []
    queue_events: list[dict[str, object]] = []
    admissions: list[dict[str, object]] = []
    bed_events: list[dict[str, object]] = []

    doctor_ids: dict[str, list[str]] = {}
    for dep_index, (dep_id, name, _, base_doctors, rooms) in enumerate(_DEPARTMENTS):
        available_doctors = max(1, round(base_doctors * doctor_factor))
        departments.append(
            {
                "department_id": dep_id,
                "hospital_id": hospital_id,
                "department_name": name,
                "nominal_capacity": rooms,
            }
        )
        doctor_ids[dep_id] = []
        for doctor_number in range(base_doctors):
            doctor_id = _stable_id("doctor", config.seed, dep_index * 10 + doctor_number)
            doctor_ids[dep_id].append(doctor_id)
            doctors.append({"doctor_id": doctor_id, "department_id": dep_id})
        for day in range(config.days):
            shift_start = start + timedelta(days=day, hours=8)
            recorded, ingested = _record_times(shift_start, rng, latency)
            rosters.append(
                {
                    "roster_id": _stable_id("roster", config.seed, dep_index * config.days + day),
                    "department_id": dep_id,
                    "event_time": shift_start,
                    "recorded_time": recorded,
                    "ingested_time": ingested,
                    "shift_start": shift_start,
                    "shift_end": shift_start + timedelta(hours=10),
                    "available_doctors": available_doctors,
                }
            )

    bed_count = max(1, round(30 * bed_factor))
    beds.extend(
        {"bed_id": _stable_id("bed", config.seed, number), "ward_id": "ward_general"}
        for number in range(bed_count)
    )
    bed_available_at = {str(bed["bed_id"]): start for bed in beds}

    patient_ordinal = encounter_ordinal = admission_ordinal = 0
    for day in range(config.days):
        weekend = (start + timedelta(days=day)).weekday() >= 5
        for dep_id, _, base_rate, base_doctors, rooms in _DEPARTMENTS:
            available_doctors = max(1, round(base_doctors * doctor_factor))
            for hour in range(8, 18):
                peak = 1.35 if hour in {10, 11, 16} else 0.8 if hour in {8, 17} else 1.0
                expected = base_rate * demand * peak * (0.7 if weekend else 1.0)
                arrivals = max(0, round(expected + rng.gauss(0, 0.8)))
                queue_length = max(0, arrivals - available_doctors)
                for _ in range(arrivals):
                    appointment_id = _stable_id("appointment", config.seed, patient_ordinal)
                    patient_key = _stable_id("patient", config.seed, patient_ordinal % 200)
                    scheduled = start + timedelta(days=day, hours=hour, minutes=rng.randrange(60))
                    lead_days = rng.randint(1, 30)
                    booking = scheduled - timedelta(days=lead_days)
                    reminder_sent = rng.random() < 0.8
                    no_show_probability = min(
                        0.55, 0.04 + lead_days * 0.004 + (0.07 if not reminder_sent else 0.0)
                    )
                    no_show = rng.random() < no_show_probability
                    recorded, ingested = _record_times(booking, rng, latency)
                    doctor_id = rng.choice(doctor_ids[dep_id])
                    appointments.append(
                        {
                            "appointment_id": appointment_id,
                            "patient_key": patient_key,
                            "hospital_id": hospital_id,
                            "department_id": dep_id,
                            "doctor_id": doctor_id,
                            "event_time": booking,
                            "recorded_time": recorded,
                            "ingested_time": ingested,
                            "booking_created_at": booking,
                            "scheduled_at": scheduled,
                            "appointment_type": rng.choice(("new", "follow_up")),
                            "priority_type": "urgent" if rng.random() < 0.12 else "routine",
                            "reminder_sent": reminder_sent,
                            "status": "no_show" if no_show else "completed",
                        }
                    )
                    patient_ordinal += 1
                    if no_show:
                        continue
                    encounter_id = _stable_id("encounter", config.seed, encounter_ordinal)
                    entered = scheduled + timedelta(minutes=rng.randint(-10, 20))
                    local_queue = max(0, queue_length + rng.randint(-1, 2))
                    wait = max(
                        0.0,
                        4
                        + 5.5 * local_queue
                        + 7 * (base_doctors - available_doctors)
                        + rng.gauss(0, 5),
                    )
                    consultation_started = entered + timedelta(minutes=wait)
                    consultation_ended = consultation_started + timedelta(
                        minutes=max(5.0, rng.lognormvariate(math.log(16), 0.3))
                    )
                    checkout = consultation_ended + timedelta(minutes=rng.uniform(2, 12))
                    recorded, ingested = _record_times(entered, rng, latency)
                    encounters.append(
                        {
                            "encounter_id": encounter_id,
                            "appointment_id": appointment_id,
                            "department_id": dep_id,
                            "event_time": entered,
                            "recorded_time": recorded,
                            "ingested_time": ingested,
                            "queue_entered_at": entered,
                            "consultation_started_at": consultation_started,
                            "consultation_ended_at": consultation_ended,
                            "checkout_at": checkout,
                        }
                    )
                    queue_events.append(
                        {
                            "queue_event_id": _stable_id("queue", config.seed, encounter_ordinal),
                            "encounter_id": encounter_id,
                            "department_id": dep_id,
                            "event_time": entered,
                            "recorded_time": recorded,
                            "ingested_time": ingested,
                            "queue_length": local_queue,
                            "available_doctors": available_doctors,
                            "available_rooms": rooms,
                        }
                    )
                    if rng.random() < 0.05:
                        admission_id = _stable_id("admission", config.seed, admission_ordinal)
                        requested_admission = checkout + timedelta(minutes=rng.uniform(10, 40))
                        bed_id = min(bed_available_at, key=bed_available_at.__getitem__)
                        admitted = max(requested_admission, bed_available_at[bed_id])
                        discharged = admitted + timedelta(hours=rng.uniform(12, 96))
                        bed_available_at[bed_id] = discharged
                        recorded_admission, ingested_admission = _record_times(
                            admitted, rng, latency
                        )
                        admissions.append(
                            {
                                "admission_id": admission_id,
                                "encounter_id": encounter_id,
                                "patient_key": patient_key,
                                "ward_id": "ward_general",
                                "event_time": admitted,
                                "recorded_time": recorded_admission,
                                "ingested_time": ingested_admission,
                                "admitted_at": admitted,
                                "expected_discharge_at": admitted + timedelta(hours=48),
                                "discharged_at": discharged,
                            }
                        )
                        for event_number, (event_type, event_time) in enumerate(
                            (("occupied", admitted), ("released", discharged))
                        ):
                            event_recorded, event_ingested = _record_times(event_time, rng, latency)
                            bed_events.append(
                                {
                                    "bed_event_id": _stable_id(
                                        "bed_event",
                                        config.seed,
                                        admission_ordinal * 2 + event_number,
                                    ),
                                    "bed_id": bed_id,
                                    "admission_id": admission_id,
                                    "event_time": event_time,
                                    "recorded_time": event_recorded,
                                    "ingested_time": event_ingested,
                                    "event_type": event_type,
                                }
                            )
                        admission_ordinal += 1
                    encounter_ordinal += 1

    frames = {
        "departments": pd.DataFrame(departments),
        "doctors": pd.DataFrame(doctors),
        "beds": pd.DataFrame(beds),
        "staffing_rosters": pd.DataFrame(rosters),
        "appointments": pd.DataFrame(appointments),
        "encounters": pd.DataFrame(encounters),
        "queue_events": pd.DataFrame(queue_events),
        "admissions": pd.DataFrame(admissions),
        "bed_events": pd.DataFrame(bed_events),
    }
    anomaly_count = min(
        len(frames["encounters"]), round(len(frames["encounters"]) * config.anomaly_rate)
    )
    if config.anomaly_rate and anomaly_count == 0 and len(frames["encounters"]):
        anomaly_count = 1
    if anomaly_count:
        indexes = rng.sample(list(frames["encounters"].index), anomaly_count)
        frames["encounters"].loc[indexes, "consultation_started_at"] = frames["encounters"].loc[
            indexes, "queue_entered_at"
        ] - pd.Timedelta(minutes=5)
    return frames
