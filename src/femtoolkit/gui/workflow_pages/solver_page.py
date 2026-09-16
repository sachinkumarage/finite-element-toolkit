"""Solver page: engineering-oriented solver configuration (Version 24, spec section 11)."""

from __future__ import annotations

import streamlit as st

from femtoolkit.application.validation import validate_solver
from femtoolkit.gui.components import error_banner, require_project
from femtoolkit.gui.state import AppState


def render(state: AppState) -> None:
    """Render the Solver configuration page."""
    st.header("Solver")
    st.caption("Solver settings for the currently selected analysis type.")

    if not require_project(state):
        return

    project = state.project
    st.write("**Solver type:** Direct linear solve (`numpy.linalg.solve` on the reduced system)")
    st.caption(
        "Both analysis types this version's GUI supports (Linear Static, Steady-State "
        "Thermal) solve directly -- no iteration is involved, so the settings below are "
        "currently unused, but are validated and stored for a future nonlinear GUI workflow."
    )

    tolerance = st.number_input(
        "Convergence tolerance",
        value=project.solver.tolerance,
        format="%.1e",
        help="Newton-Raphson residual-ratio tolerance, used by a future nonlinear workflow.",
    )
    max_iterations = st.number_input(
        "Maximum iterations",
        value=project.solver.max_iterations,
        min_value=1,
        step=1,
        help="Maximum Newton-Raphson iterations, used by a future nonlinear workflow.",
    )

    project.solver.tolerance = tolerance
    project.solver.max_iterations = int(max_iterations)

    errors = validate_solver(project)
    if errors:
        error_banner("Solver configuration invalid", errors)
    else:
        st.success("Solver configuration valid.")
