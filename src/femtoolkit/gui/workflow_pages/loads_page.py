"""Load page: configure region-based loads (Version 24, spec section 10)."""

from __future__ import annotations

import streamlit as st

from femtoolkit.application.project import LoadConfig
from femtoolkit.application.validation import validate_loads
from femtoolkit.gui.components import error_banner, require_project
from femtoolkit.gui.state import AppState

_REGIONS = ("left", "right", "top", "bottom")


def render(state: AppState) -> None:
    """Render the Load configuration page."""
    st.header("Loads")
    st.caption("A nodal force or heat flow, applied to every node on a boundary region.")

    if not require_project(state):
        return

    project = state.project
    is_thermal = project.analysis_type in ("thermal_steady_state", "thermomechanical")
    dof_options = ["HEAT_FLUX"] if is_thermal else ["X", "Y"]
    unit = "W" if is_thermal else "N"
    load_type = "Prescribed Heat Flow" if is_thermal else "Nodal Force"

    st.subheader("Add Load")
    with st.form("add_load_form"):
        st.write(f"Load Type: **{load_type}**")
        region = st.selectbox("Location (Region)", options=_REGIONS)
        direction = st.selectbox("Direction / DOF", options=dof_options)
        magnitude = st.number_input(f"Magnitude ({unit})", value=0.0)
        submitted = st.form_submit_button("Add")
    if submitted:
        project.loads.append(LoadConfig(region=region, dof=direction, magnitude=magnitude))
        st.success(f"Added load on '{region}'.")

    st.subheader("Configured Loads")
    if not project.loads:
        st.info("No loads have been defined.")
    else:
        for index, load in enumerate(project.loads):
            col_info, col_remove = st.columns([5, 1])
            with col_info:
                st.write(
                    f"**{index + 1}.** Region: `{load.region}`, Direction: `{load.dof}`, "
                    f"Magnitude: `{load.magnitude}` {unit}"
                )
            with col_remove:
                if st.button("Remove", key=f"remove_load_{index}"):
                    project.loads.pop(index)
                    st.rerun()

    errors = validate_loads(project)
    if errors:
        error_banner("Load validation", errors)
