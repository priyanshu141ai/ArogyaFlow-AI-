from typing import Any, cast

import pandas as pd
import streamlit as st

from arogyaflow.config import get_settings
from arogyaflow.dashboard_client import DashboardClient
from arogyaflow.exceptions import DashboardApiError

NAVIGATION = (
    "Overview",
    "Wait time",
    "Arrivals",
    "No-show",
    "Occupancy",
    "Simulation",
)
WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")


def _styles() -> None:
    st.markdown(
        """
        <style>
        :root { --navy:#0b2344; --teal:#009c9a; --border:#d9e2ec; }
        .stApp { background:#ffffff; color:var(--navy); }
        [data-testid="stSidebar"] { background:#0b2344; }
        [data-testid="stSidebar"] * { color:#f4f8fb; }
        [data-testid="stSidebar"] [role="radiogroup"] label {
            padding:.55rem .7rem; border-radius:10px; margin:.15rem 0;
        }
        [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
            background:#087f83;
        }
        [data-testid="stSidebar"] [role="radiogroup"]
        label[data-baseweb="radio"] > div:first-child {
            display:none;
        }
        [data-testid="stToolbar"], #MainMenu, footer { visibility:hidden; }
        h1, h2, h3 { color:var(--navy); letter-spacing:-.02em; }
        div[data-testid="stForm"], div[data-testid="stDataFrame"] {
            border:1px solid var(--border); border-radius:10px; padding:1rem;
        }
        .stButton button, .stFormSubmitButton button {
            background:#009c9a; color:#fff; border:0; border-radius:8px; font-weight:650;
        }
        .stButton button:hover, .stFormSubmitButton button:hover { background:#087f83; color:#fff; }
        [data-testid="stMetric"] { border-left:3px solid #009c9a; padding-left:1rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _post(client: DashboardClient, path: str, payload: dict[str, Any]) -> dict[str, Any] | None:
    try:
        return client.post(path, payload)
    except DashboardApiError as exc:
        st.error(str(exc))
        return None


def _metadata(result: dict[str, Any]) -> None:
    st.caption(
        f"Model {result['model_version']} · Schema {result['schema_version']} · "
        f"{'Persisted' if result['persisted'] else 'Persistence not configured'}"
    )


def _overview(client: DashboardClient) -> None:
    st.title("Operational overview")
    try:
        metadata = client.get("/v1/meta")
    except DashboardApiError as exc:
        st.error(str(exc))
        return
    st.caption("Synthetic operational data · Asia/Kolkata")
    columns = st.columns(5)
    columns[0].metric("Service", metadata["service_version"])
    versions = cast(dict[str, str | None], metadata["model_versions"])
    for column, (name, version) in zip(columns[1:], versions.items(), strict=True):
        column.metric(name.replace("_", " ").title(), version or "Missing")
    st.subheader("Recent persisted predictions")
    try:
        recent = client.get("/v1/predictions/recent?limit=50")
    except DashboardApiError as exc:
        st.info(str(exc))
        return
    records = pd.DataFrame(cast(list[dict[str, Any]], recent["records"]))
    if records.empty:
        st.info("No persisted predictions yet.")
        return
    counts = records.groupby("prediction_type").size().rename("requests")
    chart, table = st.columns([1, 2])
    chart.bar_chart(counts)
    table.dataframe(
        records[["request_id", "prediction_type", "model_version", "created_at"]],
        hide_index=True,
        width="stretch",
    )


def _wait_time(client: DashboardClient) -> None:
    st.title("Wait-time prediction")
    st.caption("Estimate operational queue delay with a prediction interval.")
    with st.form("wait-time"):
        left, middle, right = st.columns(3)
        department = left.text_input("Department ID", value="department_demo")
        appointment = left.selectbox("Appointment type", ("new", "follow_up"))
        priority = left.selectbox("Priority", ("routine", "urgent"))
        queue = middle.number_input("Queue length", min_value=0, value=4)
        doctors = middle.number_input("Available doctors", min_value=0, value=2)
        rooms = middle.number_input("Available rooms", min_value=0, value=3)
        weekday = right.selectbox("Weekday", range(7), format_func=lambda day: WEEKDAYS[day])
        hour = right.number_input("Hour", min_value=0, max_value=23, value=10)
        reminder = right.checkbox("Reminder sent", value=True)
        submitted = st.form_submit_button("Predict wait time")
    if not submitted:
        return
    result = _post(
        client,
        "/v1/predictions/wait-time",
        {
            "department_id": department,
            "appointment_type": appointment,
            "priority_type": priority,
            "reminder_sent": reminder,
            "weekday": weekday,
            "hour": hour,
            "queue_length": queue,
            "available_doctors": doctors,
            "available_rooms": rooms,
        },
    )
    if result:
        point, lower, upper = st.columns(3)
        point.metric("Predicted wait", f"{result['predicted_wait_minutes']:.1f} min")
        lower.metric("Lower estimate", f"{result['lower_wait_minutes']:.1f} min")
        upper.metric("Upper estimate", f"{result['upper_wait_minutes']:.1f} min")
        _metadata(result)


def _arrivals(client: DashboardClient) -> None:
    st.title("Patient-arrival forecast")
    horizon = st.select_slider("Forecast horizon", options=(6, 12, 24), value=12)
    if not st.button("Forecast arrivals"):
        return
    result = _post(client, "/v1/forecasts/arrivals", {"horizon_hours": horizon})
    if not result:
        return
    forecast = cast(dict[str, Any], result["forecast"])
    points = pd.DataFrame(cast(list[dict[str, Any]], forecast["points"]))
    points["interval_start"] = pd.to_datetime(points["interval_start"], utc=True)
    hospital = points.loc[points["level"] == "hospital"].set_index("interval_start")
    st.line_chart(hospital[["lower_arrivals", "predicted_arrivals", "upper_arrivals"]])
    st.dataframe(
        points.loc[points["level"] == "department"],
        hide_index=True,
        width="stretch",
    )
    _metadata(result)


def _no_show(client: DashboardClient) -> None:
    st.title("No-show probability")
    st.caption("Reminder prioritization only; appointments are never cancelled automatically.")
    with st.form("no-show"):
        first, second, third = st.columns(3)
        department = first.text_input("Department ID", value="department_demo")
        appointment = first.selectbox("Appointment type", ("new", "follow_up"))
        priority = first.selectbox("Priority", ("routine", "urgent"))
        lead = second.number_input("Booking lead hours", min_value=0.0, value=48.0)
        weekday = second.selectbox(
            "Scheduled weekday", range(7), format_func=lambda day: WEEKDAYS[day]
        )
        hour = second.number_input("Scheduled hour", min_value=0, max_value=23, value=11)
        appointments = third.number_input("Historical appointments", min_value=0, value=3)
        no_shows = third.number_input("Historical no-shows", min_value=0, value=1)
        late = third.number_input("Historical late arrivals", min_value=0, value=1)
        reminders = third.number_input("Historical reminders", min_value=0, value=2)
        submitted = st.form_submit_button("Predict no-show probability")
    if not submitted:
        return
    result = _post(
        client,
        "/v1/predictions/no-show",
        {
            "department_id": department,
            "appointment_type": appointment,
            "priority_type": priority,
            "booking_lead_hours": lead,
            "scheduled_weekday": weekday,
            "scheduled_hour": hour,
            "historical_appointments": appointments,
            "historical_no_shows": no_shows,
            "historical_late_arrivals": late,
            "historical_reminders": reminders,
        },
    )
    if result:
        probability = float(result["no_show_probability"])
        st.metric("Calibrated no-show probability", f"{probability:.1%}")
        st.progress(probability)
        if result["reminder_priority"]:
            st.warning("Reminder priority: review for outreach.")
        else:
            st.success("No additional reminder priority triggered.")
        _metadata(result)


def _occupancy(client: DashboardClient) -> None:
    st.title("Bed-occupancy forecast")
    horizon = st.select_slider("Forecast horizon", options=(6, 12, 24), value=12)
    if not st.button("Forecast occupancy"):
        return
    result = _post(client, "/v1/forecasts/occupancy", {"horizon_hours": horizon})
    if not result:
        return
    forecast = cast(dict[str, Any], result["forecast"])
    points = pd.DataFrame(cast(list[dict[str, Any]], forecast["points"]))
    points["interval_start"] = pd.to_datetime(points["interval_start"], utc=True)
    chart = points.pivot(
        index="interval_start", columns="ward_id", values="predicted_occupancy_ratio"
    )
    st.line_chart(chart)
    alerts = points.loc[points["capacity_alert"]]
    if alerts.empty:
        st.success("No capacity threshold breaches in this horizon.")
    else:
        st.warning(f"{len(alerts)} capacity-alert points require review.")
        st.dataframe(alerts, hide_index=True, width="stretch")
    _metadata(result)


def _simulation(client: DashboardClient) -> None:
    st.title("Operational scenario comparison")
    with st.form("simulation"):
        first, second, third = st.columns(3)
        doctors = first.number_input("Doctors", min_value=1, value=3)
        rooms = first.number_input("Rooms", min_value=1, value=4)
        arrivals = first.number_input("Arrivals/hour", min_value=0.1, value=8.0)
        duration = second.number_input("Duration (minutes)", min_value=1, value=480)
        service = second.number_input("Mean service minutes", min_value=0.1, value=20.0)
        maximum = second.number_input("Maximum doctors", min_value=doctors, value=doctors + 1)
        improvement = third.number_input("Minimum P90 improvement", min_value=0.0, value=1.0)
        seed = third.number_input("Simulation seed", min_value=0, value=None)
        submitted = st.form_submit_button("Compare scenarios")
    if not submitted:
        return
    if seed is None:
        st.error("Enter a simulation seed.")
        return
    result = _post(
        client,
        "/v1/simulations/compare",
        {
            "base": {
                "random_seed": seed,
                "duration_minutes": duration,
                "arrivals_per_hour": arrivals,
                "doctors": doctors,
                "rooms": rooms,
                "mean_service_minutes": service,
            },
            "constraints": {
                "max_doctors": maximum,
                "max_rooms": rooms,
                "minimum_p90_improvement_minutes": improvement,
            },
        },
    )
    if result:
        st.dataframe(result["results"], hide_index=True, width="stretch")
        recommendation = cast(dict[str, Any], result["recommendation"])
        st.info(f"Human approval required: {recommendation['reason']}")
        st.caption(" · ".join(cast(list[str], result["assumptions"])))


def main() -> None:
    st.set_page_config(
        page_title="ArogyaFlow AI", page_icon=":material/local_hospital:", layout="wide"
    )
    _styles()
    settings = get_settings()
    client = DashboardClient(str(settings.api_base_url), settings.api_timeout_seconds)
    with st.sidebar:
        st.title("ArogyaFlow AI")
        st.caption("Operational decision-support")
        page = st.radio("Navigation", NAVIGATION, label_visibility="collapsed")
        st.divider()
        st.caption("Synthetic data only · Human approval required")
    pages = {
        "Overview": _overview,
        "Wait time": _wait_time,
        "Arrivals": _arrivals,
        "No-show": _no_show,
        "Occupancy": _occupancy,
        "Simulation": _simulation,
    }
    pages[page](client)


if __name__ == "__main__":
    main()
