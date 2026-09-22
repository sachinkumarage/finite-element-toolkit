"""Solver page: engineering-oriented solver configuration (Version 24 spec
section 11, extended Version 26 spec section 21: matrix representation and
solver selection, extended Version 27 spec section 26: element-computation
execution settings).
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

import streamlit as st

from femtoolkit.application.validation import validate_execution, validate_solver
from femtoolkit.execution.config import ExecutionConfig, resolve_workers
from femtoolkit.gui.components import error_banner, require_project
from femtoolkit.gui.state import AppState

if TYPE_CHECKING:
    from femtoolkit.application.project import Project
    from femtoolkit.solvers.results import SolverResult

_MATRIX_TYPE_LABELS = {"dense": "Dense", "sparse": "Sparse"}
_SOLVER_TYPE_LABELS = {"direct": "Direct", "conjugate_gradient": "Conjugate Gradient"}
_EXECUTION_MODE_LABELS = {"serial": "Serial", "parallel": "Parallel"}


def render(state: AppState) -> None:
    """Render the Solver configuration page."""
    st.header("Solver")
    st.caption("Matrix representation, solver selection, and convergence settings.")

    if not require_project(state):
        return

    project = state.project

    col_matrix, col_solver = st.columns(2)
    with col_matrix:
        matrix_type = st.selectbox(
            "Matrix Type",
            options=list(_MATRIX_TYPE_LABELS),
            index=list(_MATRIX_TYPE_LABELS).index(project.solver.matrix_type),
            format_func=lambda key: _MATRIX_TYPE_LABELS[key],
            help=(
                "Dense: every entry stored, O(N^2) memory -- simplest, fine for small "
                "models. Sparse: only non-zero entries stored -- scales to much larger "
                "models."
            ),
        )
    with col_solver:
        solver_options = (
            ["direct"] if matrix_type == "dense" else ["direct", "conjugate_gradient"]
        )
        current_solver_type = (
            project.solver.solver_type if project.solver.solver_type in solver_options else "direct"
        )
        solver_type = st.selectbox(
            "Solver",
            options=solver_options,
            index=solver_options.index(current_solver_type),
            format_func=lambda key: _SOLVER_TYPE_LABELS[key],
            help=(
                "Direct: exact factorization, always converges for a well-posed system. "
                "Conjugate Gradient: iterative, requires a symmetric positive-definite "
                "matrix (available only with a sparse representation here)."
            ),
        )

    if matrix_type == "dense" and "conjugate_gradient" not in solver_options:
        st.caption("Conjugate Gradient requires a sparse matrix representation.")

    col_tol, col_iter = st.columns(2)
    with col_tol:
        tolerance = st.number_input(
            "Tolerance",
            value=project.solver.tolerance,
            format="%.1e",
            disabled=solver_type != "conjugate_gradient",
            help="Relative-residual convergence tolerance, used only by Conjugate Gradient.",
        )
    with col_iter:
        max_iterations = st.number_input(
            "Maximum Iterations",
            value=project.solver.max_iterations,
            min_value=1,
            step=1,
            disabled=solver_type != "conjugate_gradient",
            help="Maximum solver iterations, used only by Conjugate Gradient.",
        )

    project.solver.matrix_type = matrix_type
    project.solver.solver_type = solver_type
    project.solver.tolerance = tolerance
    project.solver.max_iterations = int(max_iterations)

    errors = validate_solver(project)
    if errors:
        error_banner("Solver configuration invalid", errors)
    else:
        st.success("Solver configuration valid.")

    st.divider()
    _render_execution_settings(project)

    if state.has_results() and state.last_run.solver_diagnostics is not None:
        st.divider()
        _render_last_diagnostics(state.last_run.solver_diagnostics)


def _render_execution_settings(project: Project) -> None:
    st.subheader("Execution Settings")
    st.caption("How per-element stiffness/conductivity matrices are computed (Version 27).")

    col_mode, col_workers, col_batch = st.columns(3)
    with col_mode:
        mode = st.selectbox(
            "Mode",
            options=list(_EXECUTION_MODE_LABELS),
            index=list(_EXECUTION_MODE_LABELS).index(project.execution.mode),
            format_func=lambda key: _EXECUTION_MODE_LABELS[key],
            help=(
                "Serial: one element at a time in this process (the default -- matches "
                "every prior version). Parallel: elements are split across several worker "
                "processes/threads. Only worthwhile for large meshes; small meshes are "
                "typically slower in parallel due to process-startup overhead."
            ),
        )
    with col_workers:
        automatic_workers = resolve_workers(ExecutionConfig())
        use_automatic = project.execution.workers is None
        workers_input = st.number_input(
            "Workers",
            value=project.execution.workers or automatic_workers,
            min_value=1,
            step=1,
            disabled=mode != "parallel",
            help=f"Number of worker processes. Automatic default: {automatic_workers}.",
        )
        automatic_checkbox = st.checkbox(
            "Automatic",
            value=use_automatic,
            disabled=mode != "parallel",
            help="Use the automatic worker count instead of the value above.",
        )
    with col_batch:
        batch_size = st.text_input(
            "Batch Size",
            value=str(project.execution.batch_size),
            disabled=mode != "parallel",
            help="'auto' or a positive integer number of elements per worker task.",
        )

    project.execution.mode = mode
    project.execution.workers = None if automatic_checkbox else int(workers_input)
    project.execution.batch_size = "auto" if batch_size.strip() == "auto" else batch_size.strip()
    if project.execution.batch_size != "auto":
        # Left as the invalid string on a ValueError; validate_execution reports it below.
        with contextlib.suppress(ValueError):
            project.execution.batch_size = int(project.execution.batch_size)

    execution_errors = validate_execution(project)
    if execution_errors:
        error_banner("Execution configuration invalid", execution_errors)
    else:
        st.success("Execution configuration valid.")


def _render_last_diagnostics(diagnostics: SolverResult) -> None:
    st.subheader("Last Run: Solver Diagnostics")
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Solver", diagnostics.solver_name)
    col_b.metric("Status", "Converged" if diagnostics.converged else "Not Converged")
    col_c.metric("Solve Time (s)", f"{diagnostics.solve_time:.4f}")

    if diagnostics.iterations is not None:
        st.caption(
            f"Iterations: {diagnostics.iterations}, "
            f"relative residual: {diagnostics.relative_residual:.3e}"
        )
    if "nnz" in diagnostics.diagnostics:
        st.caption(
            f"DOFs: {diagnostics.diagnostics.get('dofs')}, "
            f"non-zero entries: {diagnostics.diagnostics.get('nnz')}, "
            f"density: {diagnostics.diagnostics.get('density', 0.0):.4f}"
        )
