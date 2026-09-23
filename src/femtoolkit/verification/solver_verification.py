"""Solver convergence recording and direct-vs-iterative comparison (Version 29).

This module never solves anything itself -- it only reads the
diagnostics the Version 26 solver infrastructure
(:class:`~femtoolkit.solvers.results.SolverResult`) and the Version 27
performance infrastructure
(:class:`~femtoolkit.performance.profiler.PerformanceReport`) already
produce, and packages them into the same structured
:class:`~femtoolkit.verification.status.VerificationStatus` vocabulary
the rest of the verification framework uses. Advanced preconditioning
(Jacobi, incomplete LU/Cholesky) is Version 28 scope and is not
implemented by this toolkit yet; :attr:`SolverConvergenceRecord.preconditioner`
is therefore always ``None`` here -- reported honestly as unavailable
rather than fabricated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from femtoolkit.verification.cases import VerificationResult
from femtoolkit.verification.status import VerificationStatus
from femtoolkit.verification.tolerance import Tolerance

if TYPE_CHECKING:
    import numpy as np

    from femtoolkit.solvers.results import SolverResult

_DEFAULT_SOLUTION_TOLERANCE = Tolerance(absolute=1e-6, relative=1e-6)


@dataclass(frozen=True)
class SolverConvergenceRecord:
    """A structured snapshot of one solver's convergence diagnostics.

    Attributes:
        solver_name: The solver's display name (e.g. ``"Sparse
            Direct"``, ``"Conjugate Gradient"``), from
            :attr:`~femtoolkit.solvers.results.SolverResult.solver_name`.
        matrix_type: ``"dense"`` or ``"sparse"``, inferred from whether
            the result carries sparse-specific diagnostics (``nnz``).
        preconditioner: Always ``None`` -- this toolkit does not yet
            implement solver preconditioning (see the module
            docstring). Reserved for a future version.
        iterations: Iteration count, or ``None`` for a direct solver.
        final_residual: The solve's final absolute residual norm.
        relative_residual: The solve's final relative residual.
        converged: Whether the solve reported convergence.
        tolerance: The solver's own configured convergence tolerance,
            if it used one (``None`` for a direct solver, which has no
            convergence tolerance concept).
        solve_time: Wall-clock solve time, in seconds.
        degrees_of_freedom: Total DOF count, if the solver's
            diagnostics recorded one.
        non_zero_entries: Non-zero matrix entry count, if the solver's
            diagnostics recorded one (sparse solvers only).
    """

    solver_name: str
    matrix_type: str
    preconditioner: str | None
    iterations: int | None
    final_residual: float
    relative_residual: float
    converged: bool
    tolerance: float | None
    solve_time: float
    degrees_of_freedom: int | None
    non_zero_entries: int | None


def solver_convergence_record(solver_result: SolverResult) -> SolverConvergenceRecord:
    """Build a :class:`SolverConvergenceRecord` from an existing :class:`SolverResult`.

    Args:
        solver_result: The Version 26
            :class:`~femtoolkit.solvers.results.SolverResult` to record.

    Returns:
        A :class:`SolverConvergenceRecord` summarizing it.
    """
    diagnostics = solver_result.diagnostics
    matrix_type = "sparse" if "nnz" in diagnostics else "dense"
    configured_tolerance = diagnostics.get("tolerance")

    return SolverConvergenceRecord(
        solver_name=solver_result.solver_name,
        matrix_type=matrix_type,
        preconditioner=None,
        iterations=solver_result.iterations,
        final_residual=solver_result.residual_norm,
        relative_residual=solver_result.relative_residual,
        converged=solver_result.converged,
        tolerance=configured_tolerance,
        solve_time=solver_result.solve_time,
        degrees_of_freedom=diagnostics.get("dofs"),
        non_zero_entries=diagnostics.get("nnz"),
    )


def compare_solver_solutions(
    name: str,
    reference_solution: np.ndarray,
    comparison_solution: np.ndarray,
    tolerance: Tolerance = _DEFAULT_SOLUTION_TOLERANCE,
    reference_label: str = "direct solver",
    comparison_label: str = "iterative solver",
) -> VerificationResult:
    """Compare two solvers' solution vectors for the same problem.

    Verifies that switching solver strategy (e.g. from a direct solver
    to an iterative one, or between matrix representations) does not
    change the physical answer beyond numerical tolerance -- exactly
    the check spec section 8 asks for ("verify that the resulting
    solutions are numerically consistent").

    Args:
        name: A short label for this comparison (e.g. ``"Cantilever
            beam: dense vs. sparse solve"``).
        reference_solution: The baseline solution vector (typically
            from a direct solver).
        comparison_solution: The solution vector to compare against it
            (typically from an iterative solver, or a different matrix
            representation).
        tolerance: The combined absolute/relative tolerance the two
            solutions' L2 difference must be within.
        reference_label: Display label for ``reference_solution``.
        comparison_label: Display label for ``comparison_solution``.

    Returns:
        A :class:`~femtoolkit.verification.cases.VerificationResult`
        (reusing the same reporting shape every other verification
        comparison in this framework uses)
        with :attr:`~femtoolkit.verification.status.VerificationStatus.PASS`
        if the solutions agree within ``tolerance``, otherwise
        :attr:`~femtoolkit.verification.status.VerificationStatus.FAIL`.
    """
    import numpy as np

    from femtoolkit.verification.metrics import l2_error, relative_l2_error

    abs_err = l2_error(comparison_solution, reference_solution)
    rel_err = relative_l2_error(comparison_solution, reference_solution)
    reference_norm = float(np.linalg.norm(np.asarray(reference_solution, dtype=float)))
    satisfied = abs_err <= tolerance.allowed_error(reference_norm)
    status = VerificationStatus.PASS if satisfied else VerificationStatus.FAIL

    message = (
        f"{comparison_label} vs. {reference_label}: absolute_error={abs_err:.6e}, "
        f"relative_error={rel_err:.6e} -> {status.value.upper()}"
    )

    return VerificationResult(
        case_name=name,
        description=f"Comparison of {comparison_label} against {reference_label} for one system.",
        analysis_type="solver_comparison",
        quantity="Solution vector (L2 norm)",
        reference_value=reference_solution,
        numerical_value=comparison_solution,
        absolute_error=abs_err,
        relative_error=rel_err,
        tolerance=tolerance,
        status=status,
        message=message,
    )


__all__ = ["SolverConvergenceRecord", "compare_solver_solutions", "solver_convergence_record"]
