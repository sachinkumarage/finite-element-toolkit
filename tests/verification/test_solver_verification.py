"""Tests for femtoolkit.verification.solver_verification (Version 29)."""

from __future__ import annotations

import numpy as np

from femtoolkit.analysis import BoundaryCondition, NodalLoad, StaticLinearAnalysis, TranslationDOF
from femtoolkit.materials import LinearElastic2D
from femtoolkit.mesh.generator import create_quad_mesh
from femtoolkit.solvers import ConjugateGradientSolver, DenseDirectSolver, SparseDirectSolver
from femtoolkit.verification.solver_verification import (
    compare_solver_solutions,
    solver_convergence_record,
)
from femtoolkit.verification.status import VerificationStatus
from femtoolkit.verification.tolerance import Tolerance

X, Y = TranslationDOF.X, TranslationDOF.Y


def _build(solver):
    material = LinearElastic2D(youngs_modulus=200e9, poisson_ratio=0.3, formulation="plane_stress")
    mesh = create_quad_mesh(width=4.0, height=1.0, nx=10, ny=3, material=material, thickness=0.02)
    analysis = StaticLinearAnalysis(mesh, solver=solver)
    for node in mesh.nodes:
        if node.x == 0.0:
            analysis.add_boundary_condition(BoundaryCondition(node.id, X, 0.0))
            analysis.add_boundary_condition(BoundaryCondition(node.id, Y, 0.0))
    for node in mesh.nodes:
        if node.x == 4.0 and node.y == 0.0:
            analysis.add_load(NodalLoad(node.id, Y, -10000.0))
    return analysis


def test_direct_vs_iterative_solutions_agree() -> None:
    direct_result = _build(SparseDirectSolver()).solve()
    iterative_result = _build(ConjugateGradientSolver(tolerance=1e-12, max_iterations=5000)).solve()

    comparison = compare_solver_solutions(
        "Cantilever: direct vs CG", direct_result.displacements, iterative_result.displacements
    )

    assert comparison.status is VerificationStatus.PASS


def test_mismatched_solutions_fail() -> None:
    direct_result = _build(SparseDirectSolver()).solve()
    corrupted_solution = direct_result.displacements.copy()
    corrupted_solution[0] += 1.0  # a full meter of displacement error is not tolerance noise

    comparison = compare_solver_solutions(
        "Corrupted comparison",
        direct_result.displacements,
        corrupted_solution,
        tolerance=Tolerance(absolute=1e-9, relative=1e-9),
    )

    assert comparison.status is VerificationStatus.FAIL


def test_solver_convergence_record_from_direct_solver() -> None:
    analysis = _build(DenseDirectSolver())
    analysis.solve()
    record = solver_convergence_record(analysis.last_solver_result)

    assert record.solver_name == "Dense Direct"
    assert record.matrix_type == "dense"
    assert record.iterations is None
    assert record.preconditioner is None
    assert record.converged is True


def test_solver_convergence_record_from_iterative_solver() -> None:
    analysis = _build(ConjugateGradientSolver(tolerance=1e-10, max_iterations=2000))
    analysis.solve()
    record = solver_convergence_record(analysis.last_solver_result)

    assert record.solver_name == "Conjugate Gradient"
    assert record.matrix_type == "sparse"
    assert record.iterations is not None
    assert record.iterations > 0
    assert record.non_zero_entries is not None


def test_residual_history_tracking_is_opt_in_and_matches_iteration_count() -> None:
    analysis = _build(
        ConjugateGradientSolver(tolerance=1e-10, max_iterations=2000, track_residual_history=True)
    )
    analysis.solve()
    history = analysis.last_solver_result.diagnostics["residual_history"]

    assert len(history) == analysis.last_solver_result.iterations
    assert history[-1] < history[0]  # residual decreased over the solve


def test_residual_history_absent_by_default() -> None:
    analysis = _build(ConjugateGradientSolver(tolerance=1e-10, max_iterations=2000))
    analysis.solve()
    assert "residual_history" not in analysis.last_solver_result.diagnostics


def test_compare_solver_solutions_preserves_reference_and_numerical_values() -> None:
    reference = np.array([1.0, 2.0, 3.0])
    numerical = np.array([1.0000001, 2.0000001, 3.0000001])
    comparison = compare_solver_solutions("Synthetic", reference, numerical)
    np.testing.assert_array_equal(comparison.reference_value, reference)
    np.testing.assert_array_equal(comparison.numerical_value, numerical)
