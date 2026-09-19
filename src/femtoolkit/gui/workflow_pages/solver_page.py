"""Solver page: engineering-oriented solver configuration (Version 24 spec
section 11, extended Version 26 spec section 21: matrix representation and
solver selection).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import streamlit as st

from femtoolkit.application.validation import validate_solver
from femtoolkit.gui.components import error_banner, require_project
from femtoolkit.gui.state import AppState

if TYPE_CHECKING:
    from femtoolkit.solvers.results import SolverResult

_MATRIX_TYPE_LABELS = {"dense": "Dense", "sparse": "Sparse"}
_SOLVER_TYPE_LABELS = {"direct": "Direct", "conjugate_gradient": "Conjugate Gradient"}


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

    if state.has_results() and state.last_run.solver_diagnostics is not None:
        st.divider()
        _render_last_diagnostics(state.last_run.solver_diagnostics)


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
