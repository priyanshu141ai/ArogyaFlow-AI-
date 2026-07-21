import streamlit as st

from arogyaflow.simulation import SimulationConfig, SimulationConstraints, compare_scenarios


def main() -> None:
    st.set_page_config(page_title="ArogyaFlow AI", page_icon="🏥")
    st.title("ArogyaFlow AI")
    st.caption("Operational scenario comparison")
    doctors = st.number_input("Doctors", min_value=1, value=3)
    rooms = st.number_input("Rooms", min_value=1, value=4)
    arrivals = st.number_input("Arrivals/hour", min_value=0.1, value=8.0)
    duration = st.number_input("Duration (minutes)", min_value=1, value=480)
    service_minutes = st.number_input("Mean service minutes", min_value=0.1, value=20.0)
    max_doctors = st.number_input("Maximum doctors", min_value=doctors, value=doctors + 1)
    minimum_improvement = st.number_input("Minimum P90 improvement", min_value=0.0, value=1.0)
    seed = st.number_input("Simulation seed", min_value=0, value=None)
    if st.button("Compare scenarios"):
        if seed is None:
            st.error("Enter a simulation seed.")
            return
        comparison = compare_scenarios(
            SimulationConfig(
                random_seed=seed,
                duration_minutes=duration,
                arrivals_per_hour=arrivals,
                doctors=doctors,
                rooms=rooms,
                mean_service_minutes=service_minutes,
            ),
            SimulationConstraints(
                max_doctors=max_doctors,
                max_rooms=rooms,
                minimum_p90_improvement_minutes=minimum_improvement,
            ),
        )
        st.dataframe([result.model_dump() for result in comparison.results])
        st.info(comparison.recommendation.reason)


if __name__ == "__main__":
    main()
