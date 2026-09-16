"""Boundary condition page: configure region-based constraints (Version 24, spec section 9)."""

from __future__ import annotations

import streamlit as st

from femtoolkit.application.project import BoundaryConditionConfig
from femtoolkit.application.validation import validate_boundary_conditions
from femtoolkit.gui.components import error_banner, require_project
from femtoolkit.gui.state import AppState

_REGIONS = ("left", "right", "top", "bottom")


def render(state: AppState) -> None:
    """Render the Boundary Conditions configuration page."""
    st.header("Boundary Conditions")
    st.caption(
        "Prescribed displacement or temperature, applied to every node on a boundary region."
    )

    if not require_project(state):
        return

    project = state.project
    is_thermal = project.analysis_type in ("thermal_steady_state", "thermomechanical")
    dof_options = ["TEMPERATURE"] if is_thermal else ["X", "Y"]
    unit = "K" if is_thermal else "m"

    st.subheader("Add Boundary Condition")
    with st.form("add_boundary_condition_form"):
        region = st.selectbox("Node / Region", options=_REGIONS)
        dof = st.selectbox("DOF", options=dof_options)
        value = st.number_input(f"Prescribed Value ({unit})", value=0.0)
        submitted = st.form_submit_button("Add")
    if submitted:
        project.boundary_conditions.append(
            BoundaryConditionConfig(region=region, dof=dof, value=value)
        )
        st.success(f"Added boundary condition on '{region}'.")

    st.subheader("Configured Boundary Conditions")
    if not project.boundary_conditions:
        st.info("No boundary conditions have been defined.")
    else:
        for index, bc in enumerate(project.boundary_conditions):
            col_info, col_remove = st.columns([5, 1])
            with col_info:
                st.write(
                    f"**{index + 1}.** Region: `{bc.region}`, DOF: `{bc.dof}`, "
                    f"Value: `{bc.value}` {unit}"
                )
            with col_remove:
                if st.button("Remove", key=f"remove_bc_{index}"):
                    project.boundary_conditions.pop(index)
                    st.rerun()

    errors = validate_boundary_conditions(project)
    if errors:
        error_banner("Boundary condition validation", errors)
