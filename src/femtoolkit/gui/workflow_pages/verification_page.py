"""Verification & Validation page (Version 29 spec section 15).

Surfaces the Version 29 verification/validation/reporting framework
inside the existing GUI: running the standard analytical benchmark
suite, viewing the current project's solver convergence and equilibrium
check (both already computed by
:class:`~femtoolkit.application.simulation_service.SimulationService`,
Version 29), running a mesh-refinement convergence study over the
current project, comparing against an uploaded reference dataset, and
generating/downloading an engineering report -- all through the
existing :mod:`femtoolkit.application`/:mod:`femtoolkit.verification`/
:mod:`femtoolkit.validation`/:mod:`femtoolkit.reporting` layers, with no
verification/solving logic implemented in this module itself.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import numpy as np
import streamlit as st

from femtoolkit.application.simulation_service import SimulationService
from femtoolkit.gui.components import require_project
from femtoolkit.gui.state import AppState
from femtoolkit.reporting import (
    EngineeringReport,
    collect_reproducibility_metadata,
    render_html,
    render_markdown,
)
from femtoolkit.validation import compare_to_reference_dataset
from femtoolkit.verification.benchmarks import standard_benchmark_suite
from femtoolkit.verification.runner import VerificationRunner
from femtoolkit.verification.solver_verification import solver_convergence_record
from femtoolkit.verification.status import VerificationStatus

if TYPE_CHECKING:
    from femtoolkit.verification.runner import VerificationReport

_simulation_service = SimulationService()

_BENCHMARK_REPORT_KEY = "femtoolkit_benchmark_report"
_MESH_CONVERGENCE_KEY = "femtoolkit_mesh_convergence_history"

_STATUS_ICONS = {
    VerificationStatus.PASS: "✅",
    VerificationStatus.FAIL: "❌",
    VerificationStatus.WARNING: "⚠️",
    VerificationStatus.NOT_AVAILABLE: "❓",
    VerificationStatus.NOT_RUN: "➖",
}


def render(state: AppState) -> None:
    """Render the Verification & Validation page."""
    st.header("Verification & Validation")
    st.caption(
        "Analytical benchmarks, solver/equilibrium diagnostics, mesh convergence, "
        "reference-data comparison, and report generation."
    )

    _render_benchmark_suite()
    st.divider()
    _render_current_run_diagnostics(state)
    st.divider()
    _render_mesh_convergence(state)
    st.divider()
    _render_validation_dataset(state)
    st.divider()
    _render_report_generation(state)


def _render_benchmark_suite() -> None:
    st.subheader("1. Analytical Benchmarks")
    st.caption(
        "Independent of the current project -- verifies the toolkit's own element "
        "formulations against closed-form solutions (axial bar, truss, cantilever "
        "beam, 1D thermal conduction, Q4/CST/HEX8 patch tests)."
    )
    if st.button("Run Standard Benchmark Suite"):
        st.session_state[_BENCHMARK_REPORT_KEY] = VerificationRunner().run_all(
            standard_benchmark_suite()
        )

    report: VerificationReport | None = st.session_state.get(_BENCHMARK_REPORT_KEY)
    if report is None:
        st.info("No benchmarks have been run yet.")
        return

    for result in report.results:
        icon = _STATUS_ICONS[result.status]
        rel_err = "-" if result.relative_error is None else f"{result.relative_error:.3e}"
        st.write(f"{icon} **{result.case_name}** -- {result.quantity}: relative error {rel_err}")

    if report.all_passed:
        st.success(f"All {len(report.results)} benchmark cases passed.")
    else:
        st.warning(f"{report.passed}/{len(report.results)} benchmark cases passed.")


def _render_current_run_diagnostics(state: AppState) -> None:
    st.subheader("2. Current Run: Solver Convergence & Equilibrium")
    if not state.has_results():
        st.info("Run a simulation on the Run page to see its diagnostics here.")
        return

    run_result = state.last_run
    if run_result.solver_diagnostics is None:
        st.caption("Solver: Dense Direct (default) -- no convergence record for this path.")
    else:
        record = solver_convergence_record(run_result.solver_diagnostics)
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Solver", record.solver_name)
        col_b.metric("Converged", "Yes" if record.converged else "No")
        col_c.metric("Iterations", record.iterations if record.iterations is not None else "-")
        st.caption(f"Preconditioner: {record.preconditioner or 'none (not yet implemented)'}")

    if run_result.equilibrium_check is None:
        st.caption("No equilibrium check is available for this run.")
    else:
        check = run_result.equilibrium_check
        icon = _STATUS_ICONS[check.status]
        st.write(f"{icon} **{check.name}**: {check.message}")


def _render_mesh_convergence(state: AppState) -> None:
    st.subheader("3. Mesh Convergence Study")
    if not require_project(state):
        return

    project = state.project
    st.caption(
        "Re-solves the current project at 1x, 2x, and 4x its configured mesh "
        "subdivisions, tracking the maximum displacement/temperature at each level."
    )
    if st.button("Run Mesh Convergence Study"):
        history = []
        for factor, label in ((1, "Coarse"), (2, "Medium"), (4, "Fine")):
            scaled_project = copy.deepcopy(project)
            scaled_project.mesh.nx = max(1, project.mesh.nx * factor)
            scaled_project.mesh.ny = max(1, project.mesh.ny * factor)
            run_result = _simulation_service.run(scaled_project)
            if not run_result.succeeded:
                st.error(f"{label} mesh failed to solve: {run_result.errors}")
                history = []
                break
            value = (
                run_result.summary.maximum_displacement
                if run_result.summary.maximum_displacement is not None
                else run_result.summary.maximum_temperature
            )
            history.append((label, scaled_project.mesh.nx * scaled_project.mesh.ny, value))
        st.session_state[_MESH_CONVERGENCE_KEY] = history

    history = st.session_state.get(_MESH_CONVERGENCE_KEY)
    if not history:
        st.info("No mesh convergence study has been run yet.")
        return

    previous_value: float | None = None
    for label, element_count, value in history:
        change = (
            "-"
            if previous_value is None
            else f"{abs(value - previous_value) / max(abs(value), 1e-12):.3e}"
        )
        st.write(
            f"**{label}** ({element_count} elements): value={value:.6e}, relative change={change}"
        )
        previous_value = value


def _render_validation_dataset(state: AppState) -> None:
    st.subheader("4. Reference Dataset Comparison")
    st.caption(
        "Upload a JSON reference dataset (see docs/validation.md) to compare against "
        "the current run's key result. No experimental data ships with this toolkit -- "
        "a comparison only happens when you supply a real dataset."
    )
    uploaded_file = st.file_uploader("Reference dataset (JSON)", type=["json"])
    if uploaded_file is None:
        return
    if not state.has_results():
        st.warning("Run a simulation first so there is a result to compare against.")
        return

    import json

    from femtoolkit.validation.datasets import reference_dataset_from_dict

    try:
        data = json.loads(uploaded_file.getvalue().decode("utf-8"))
        dataset = reference_dataset_from_dict(data)
    except Exception as error:  # noqa: BLE001 - any malformed upload must degrade to a message
        st.error(f"Could not read reference dataset: {error}")
        return

    summary = state.last_run.summary
    simulation_value = summary.maximum_displacement
    if simulation_value is None:
        simulation_value = summary.maximum_temperature
    if simulation_value is None or len(dataset.values) != 1:
        st.warning(
            "This page only supports comparing a single reference value against the "
            "current run's headline result; the uploaded dataset or run result does "
            "not match that shape."
        )
        return

    result = compare_to_reference_dataset(dataset, np.array([simulation_value]))
    icon = _STATUS_ICONS[result.status]
    st.write(f"{icon} {result.message}")


def _render_report_generation(state: AppState) -> None:
    st.subheader("5. Engineering Report")
    if not require_project(state):
        return
    if not state.has_results():
        st.info("Run a simulation to generate a report for it.")
        return

    project = state.project
    run_result = state.last_run
    benchmark_report: VerificationReport | None = st.session_state.get(_BENCHMARK_REPORT_KEY)

    metadata = collect_reproducibility_metadata(
        model_name=project.name,
        analysis_type=project.analysis_type,
        mesh_statistics={"width": project.mesh.width, "height": project.mesh.height},
        element_types=[project.mesh.element_type],
        material_properties={
            "youngs_modulus": project.material.youngs_modulus or 0.0,
            "poisson_ratio": project.material.poisson_ratio or 0.0,
        },
        boundary_conditions_summary=f"{len(project.boundary_conditions)} boundary condition(s).",
        loads_summary=f"{len(project.loads)} load(s).",
        solver_name=(
            run_result.solver_diagnostics.solver_name
            if run_result.solver_diagnostics is not None
            else "Dense Direct"
        ),
        execution_mode=project.execution.mode,
    )

    key_results = {}
    if run_result.summary.maximum_displacement is not None:
        key_results["Maximum displacement (m)"] = run_result.summary.maximum_displacement
    if run_result.summary.maximum_von_mises_stress is not None:
        key_results["Maximum von Mises stress (Pa)"] = run_result.summary.maximum_von_mises_stress
    if run_result.summary.maximum_temperature is not None:
        key_results["Maximum temperature (K)"] = run_result.summary.maximum_temperature

    report = EngineeringReport(
        title=f"{project.name}: Engineering Simulation Report",
        simulation_summary=f"{project.analysis_type} analysis of '{project.name}'.",
        model_description=f"Project '{project.name}', analysis type {project.analysis_type}.",
        geometry_description=(
            f"Rectangular domain {project.mesh.width} m x {project.mesh.height} m."
        ),
        mesh_summary={"width": project.mesh.width, "height": project.mesh.height},
        materials_summary={"Material": {"E": project.material.youngs_modulus or 0.0}},
        boundary_conditions_summary=f"{len(project.boundary_conditions)} boundary condition(s).",
        loads_summary=f"{len(project.loads)} load(s).",
        analysis_type=project.analysis_type,
        solver_configuration=(
            f"matrix_type={project.solver.matrix_type}, solver_type={project.solver.solver_type}"
        ),
        reproducibility=metadata,
        verification_results=benchmark_report.results if benchmark_report is not None else [],
        equilibrium_checks=(
            [run_result.equilibrium_check] if run_result.equilibrium_check is not None else []
        ),
        key_results=key_results,
    )

    markdown_text = render_markdown(report)
    html_text = render_html(report)

    col_a, col_b = st.columns(2)
    with col_a:
        st.download_button(
            "Download Markdown Report",
            data=markdown_text,
            file_name="report.md",
            mime="text/markdown",
        )
    with col_b:
        st.download_button(
            "Download HTML Report", data=html_text, file_name="report.html", mime="text/html"
        )
    st.text_area("Preview (Markdown)", value=markdown_text, height=300)
