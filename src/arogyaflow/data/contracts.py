import pandera.pandas as pa

_TIME = pa.Column("datetime64[ns, UTC]")
_EVENT_TIMES = {
    "event_time": _TIME,
    "recorded_time": _TIME,
    "ingested_time": _TIME,
}
_EVENT_ORDER = pa.Check(
    lambda frame: (
        (frame["event_time"] <= frame["recorded_time"])
        & (frame["recorded_time"] <= frame["ingested_time"])
    ),
    error="event_time <= recorded_time <= ingested_time",
)

CONTRACTS: dict[str, pa.DataFrameSchema] = {
    "departments": pa.DataFrameSchema(
        {
            "department_id": pa.Column(str, unique=True),
            "hospital_id": pa.Column(str),
            "department_name": pa.Column(str),
            "nominal_capacity": pa.Column(int, pa.Check.ge(1)),
        },
        strict=True,
    ),
    "doctors": pa.DataFrameSchema(
        {"doctor_id": pa.Column(str, unique=True), "department_id": pa.Column(str)}, strict=True
    ),
    "beds": pa.DataFrameSchema(
        {"bed_id": pa.Column(str, unique=True), "ward_id": pa.Column(str)}, strict=True
    ),
    "staffing_rosters": pa.DataFrameSchema(
        {
            "roster_id": pa.Column(str, unique=True),
            "department_id": pa.Column(str),
            **_EVENT_TIMES,
            "shift_start": _TIME,
            "shift_end": _TIME,
            "available_doctors": pa.Column(int, pa.Check.ge(1)),
        },
        checks=[_EVENT_ORDER, pa.Check(lambda frame: frame["shift_start"] < frame["shift_end"])],
        strict=True,
    ),
    "appointments": pa.DataFrameSchema(
        {
            "appointment_id": pa.Column(str, unique=True),
            "patient_key": pa.Column(str),
            "hospital_id": pa.Column(str),
            "department_id": pa.Column(str),
            "doctor_id": pa.Column(str),
            **_EVENT_TIMES,
            "booking_created_at": _TIME,
            "scheduled_at": _TIME,
            "appointment_type": pa.Column(str, pa.Check.isin(["new", "follow_up"])),
            "priority_type": pa.Column(str, pa.Check.isin(["routine", "urgent"])),
            "reminder_sent": pa.Column(bool),
            "status": pa.Column(str, pa.Check.isin(["completed", "no_show"])),
        },
        checks=[
            _EVENT_ORDER,
            pa.Check(lambda frame: frame["booking_created_at"] < frame["scheduled_at"]),
        ],
        strict=True,
    ),
    "encounters": pa.DataFrameSchema(
        {
            "encounter_id": pa.Column(str, unique=True),
            "appointment_id": pa.Column(str),
            "department_id": pa.Column(str),
            **_EVENT_TIMES,
            "queue_entered_at": _TIME,
            "consultation_started_at": _TIME,
            "consultation_ended_at": _TIME,
            "checkout_at": _TIME,
        },
        checks=[
            _EVENT_ORDER,
            pa.Check(lambda frame: frame["queue_entered_at"] <= frame["consultation_started_at"]),
            pa.Check(
                lambda frame: frame["consultation_started_at"] < frame["consultation_ended_at"]
            ),
            pa.Check(lambda frame: frame["consultation_ended_at"] <= frame["checkout_at"]),
        ],
        strict=True,
    ),
    "queue_events": pa.DataFrameSchema(
        {
            "queue_event_id": pa.Column(str, unique=True),
            "encounter_id": pa.Column(str),
            "department_id": pa.Column(str),
            **_EVENT_TIMES,
            "queue_length": pa.Column(int, pa.Check.ge(0)),
            "available_doctors": pa.Column(int, pa.Check.ge(1)),
            "available_rooms": pa.Column(int, pa.Check.ge(1)),
        },
        checks=_EVENT_ORDER,
        strict=True,
    ),
    "admissions": pa.DataFrameSchema(
        {
            "admission_id": pa.Column(str, unique=True),
            "encounter_id": pa.Column(str),
            "patient_key": pa.Column(str),
            "ward_id": pa.Column(str),
            **_EVENT_TIMES,
            "admitted_at": _TIME,
            "expected_discharge_at": _TIME,
            "discharged_at": _TIME,
        },
        checks=[
            _EVENT_ORDER,
            pa.Check(lambda frame: frame["admitted_at"] < frame["discharged_at"]),
        ],
        strict=True,
    ),
    "bed_events": pa.DataFrameSchema(
        {
            "bed_event_id": pa.Column(str, unique=True),
            "bed_id": pa.Column(str),
            "admission_id": pa.Column(str),
            **_EVENT_TIMES,
            "event_type": pa.Column(str, pa.Check.isin(["occupied", "released"])),
        },
        checks=_EVENT_ORDER,
        strict=True,
    ),
}


def data_dictionary() -> dict[str, dict[str, str]]:
    return {
        table: {name: str(column.dtype) for name, column in schema.columns.items()}
        for table, schema in CONTRACTS.items()
    }
