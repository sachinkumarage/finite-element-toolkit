"""FEA Analysis page: run a single existing FEA example with user-chosen parameters.

Reuses the existing :mod:`femtoolkit.application` service layer directly -- no
solver logic is implemented in this module.
"""

from __future__ import annotations

import streamlit as st

from femtoolkit.application.project import BoundaryConditionConfig, LoadConfig, Project
from femtoolkit.application.simulation_service import SimulationService

_RESULT_KEY = "quickstart_fea_result"

_EXAMPLES = ("Cantilever Beam",)


def _build_cantilever_project(
    thickness: float, youngs_modulus: float, load_magnitude: float
) -> Project:
    project = Project(name="Quickstart Cantilever", analysis_type="linear_static")
    project.material.youngs_modulus = youngs_modulus
    project.material.poisson_ratio = 0.3
    project.material.density = 7850.0
    project.mesh.width = 2.0
    project.mesh.height = 0.4
    project.mesh.nx = 10
    project.mesh.ny = 3
    project.mesh.thickness = thickness
    project.boundary_conditions = [
        BoundaryConditionConfig(region="left", dof="X", value=0.0),
        BoundaryConditionConfig(region="left", dof="Y", value=0.0),
    ]
    project.loads = [LoadConfig(region="right", dof="Y", magnitude=load_magnitude)]
    return project


def render() -> None:
    """Render the FEA Analysis page."""
    st.header("FEA Analysis")
    st.caption("Select an example, set its parameters, and run it through the real solver.")

    example = st.selectbox("Example", options=_EXAMPLES)

    col_thickness, col_modulus, col_load = st.columns(3)
    thickness = col_thickness.number_input(
        "Beam thickness (m)", min_value=0.001, max_value=0.1, value=0.01, step=0.001, format="%.4f"
    )
    youngs_modulus = col_modulus.number_input(
        "Young's modulus (Pa)", min_value=1.0e9, max_value=500.0e9, value=200.0e9, step=1.0e9,
        format="%.3e",
    )
    load_magnitude = col_load.number_input(
        "Tip load (N)", min_value=-50000.0, max_value=0.0, value=-4000.0, step=500.0
    )

    if st.button("Run FEA", type="primary"):
        if example == "Cantilever Beam":
            project = _build_cantilever_project(thickness, youngs_modulus, load_magnitude)
        result = SimulationService().run(project)
        st.session_state[_RESULT_KEY] = result

    result = st.session_state.get(_RESULT_KEY)
    if result is None:
        st.info("Set parameters above and click 'Run FEA'.")
        return

    if not result.succeeded:
        st.error("The analysis did not complete.")
        for message in result.errors:
            st.write(f"- {message}")
        return

    st.success("Analysis completed.")
    col_disp, col_stress = st.columns(2)
    col_disp.metric("Maximum displacement (m)", f"{result.summary.maximum_displacement:.4e}")
    col_stress.metric(
        "Maximum von Mises stress (Pa)", f"{result.summary.maximum_von_mises_stress:.4e}"
    )

    st.bar_chart(
        {
            "Maximum displacement (m)": [result.summary.maximum_displacement],
            "Maximum von Mises stress (GPa)": [result.summary.maximum_von_mises_stress / 1e9],
        }
    )


__all__ = ["render"]
