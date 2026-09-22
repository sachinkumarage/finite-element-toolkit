"""Run page: validate and execute the simulation (Version 24 spec section 12,
extended Version 26 spec section 24 with post-solve solver diagnostics,
extended Version 27 spec section 26 with a post-solve performance summary).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import streamlit as st

from femtoolkit.application.simulation_service import SimulationService
from femtoolkit.application.validation import validate_project
from femtoolkit.gui.components import error_banner, require_project, run_status_banner
from femtoolkit.gui.state import AppState

if TYPE_CHECKING:
    from femtoolkit.performance.profiler import PerformanceReport

_simulation_service = SimulationService()


def render(state: AppState) -> None:
    """Render the Run Simulation page."""
    st.header("Run Simulation")
    st.caption("Validate Model -> Run Simulation -> Simulation Status -> Results Available")

    if not require_project(state):
        return

    project = state.project

    st.subheader("1. Validate Model")
    validation = validate_project(project)
    if validation.is_valid:
        st.success("Project configuration is valid.")
    else:
        error_banner("Validation failed", validation.errors)

    st.subheader("2. Run Simulation")
    run_clicked = st.button("Run Simulation", type="primary", disabled=not validation.is_valid)
    if run_clicked:
        with st.spinner("Running simulation..."):
            state.last_run = _simulation_service.run(project)

    st.subheader("3. Simulation Status")
    run_status_banner(state.last_run)

    if state.has_results():
        st.subheader("4. Results Available")
        st.info("Open the Results or Visualization page to inspect the solved fields.")
        _render_analysis_complete(state)


def _render_analysis_complete(state: AppState) -> None:
    st.success("Analysis Complete")
    diagnostics = state.last_run.solver_diagnostics
    if diagnostics is None:
        st.caption("Solver: Dense Direct (default) -- no diagnostics object for this path.")
    else:
        col_solver, col_dofs, col_time, col_status = st.columns(4)
        col_solver.metric("Solver", diagnostics.solver_name)
        col_dofs.metric("DOFs", diagnostics.diagnostics.get("dofs", "-"))
        col_time.metric("Solve Time (s)", f"{diagnostics.solve_time:.4f}")
        col_status.metric("Status", "Converged" if diagnostics.converged else "Not Converged")

        if "nnz" in diagnostics.diagnostics:
            st.caption(
                f"Non-zero entries: {diagnostics.diagnostics['nnz']}, "
                f"density: {diagnostics.diagnostics.get('density', 0.0):.4f}"
            )
        if diagnostics.iterations is not None:
            st.caption(
                f"Iterations: {diagnostics.iterations}, "
                f"final residual: {diagnostics.residual_norm:.3e}"
            )

    report = state.last_run.performance_report
    if report is not None:
        st.divider()
        _render_performance_summary(report)


def _render_performance_summary(report: PerformanceReport) -> None:
    st.subheader("Simulation Performance")
    mode_label = "Parallel" if report.execution_mode == "parallel" else "Serial"
    caption = f"Execution Mode: {mode_label}"
    if report.workers is not None:
        caption += f" ({report.workers} workers)"
    st.caption(caption)

    stage_labels = (
        ("Element", report.element_time),
        ("Assembly", report.assembly_time),
        ("Solve", report.solve_time),
    )
    columns = st.columns(len(stage_labels) + 1)
    for column, (label, value) in zip(columns, stage_labels, strict=False):
        column.metric(f"{label} (s)", f"{value:.4f}" if value is not None else "-")
    columns[-1].metric("Total (s)", f"{report.total_time:.4f}")
