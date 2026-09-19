"""The structured solver result object (Version 26, spec section 13).

Every :class:`~femtoolkit.solvers.base.LinearSolver` returns a
:class:`SolverResult` rather than a bare :class:`numpy.ndarray` -- the
solution vector is still there (``.solution``), but alongside it comes
exactly the diagnostic information spec section 14 asks a solver report
to contain: whether it converged, how many iterations it took (``None``
for a direct solver, which has no iteration concept), the achieved
residual, how long the solve took, and a free-form ``diagnostics``
dictionary for anything solver-specific (non-zero entry count, matrix
density, and so on).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SolverResult:
    """The outcome of one linear solve, with full diagnostic information.

    Attributes:
        solution: The solved vector (displacement, temperature, ...),
            in global DOF order -- the same array a caller previously
            got directly back from
            :func:`~femtoolkit.analysis.system.solve`.
        converged: Whether the solve succeeded. Always ``True`` for a
            direct solver that did not raise; for an iterative solver,
            whether the residual tolerance was reached within
            ``max_iterations``.
        iterations: Number of iterations performed, or ``None`` for a
            direct solver (which has no iteration count).
        residual_norm: ``norm(b - A @ x)`` on the reduced (free-DOF)
            system that was actually solved, in the same units as the
            right-hand side.
        relative_residual: ``residual_norm / max(norm(b), eps)`` -- a
            dimensionless measure of solution quality, independent of
            the problem's absolute force/temperature-load scale.
        solve_time: Wall-clock time spent inside the solver's own
            numerical routine, in seconds (excludes assembly).
        solver_name: A short, human-readable solver identifier (e.g.
            ``"Dense Direct"``, ``"Sparse Direct"``, ``"Conjugate
            Gradient"``), used directly in diagnostic displays.
        diagnostics: Solver-specific extra information (e.g.
            ``{"dofs": N, "free_dofs": F, "nnz": nnz, "density": rho}``
            for a sparse solver). Never fabricated -- only values the
            solver actually computed are included.
    """

    solution: np.ndarray
    converged: bool
    iterations: int | None
    residual_norm: float
    relative_residual: float
    solve_time: float
    solver_name: str
    diagnostics: dict[str, Any] = field(default_factory=dict)


__all__ = ["SolverResult"]
