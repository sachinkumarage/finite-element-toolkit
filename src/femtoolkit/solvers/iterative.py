r"""The Conjugate Gradient iterative solver (Version 26, spec section 11).

**Why Conjugate Gradient.** For a symmetric positive-definite (SPD)
matrix ``A`` (every stiffness matrix this toolkit assembles from a
properly constrained, physically valid linear elastic or thermal
conduction problem is SPD once boundary conditions are eliminated), the
linear system ``A x = b`` is exactly equivalent to minimizing the
quadratic form

.. math::

    f(\mathbf{x}) = \frac{1}{2}\mathbf{x}^T\mathbf{A}\mathbf{x} - \mathbf{b}^T\mathbf{x}

(its unique minimum occurs exactly where its gradient, ``A x - b``,
vanishes -- i.e. where ``A x = b``). Conjugate Gradient minimizes this
quadratic form by searching along a sequence of mutually
``A``-conjugate directions, converging to the exact solution in at most
``N`` iterations in exact arithmetic (and, in floating-point practice,
to a useful tolerance in far fewer iterations for a well-conditioned
system), all using only sparse matrix-vector products -- never an
explicit matrix factorization. This makes CG the standard choice for
large, sparse SPD systems where a direct factorization's fill-in would
be too expensive; it is *not* appropriate for a non-symmetric or
indefinite matrix, which is why :class:`ConjugateGradientSolver`
performs a cheap symmetry check before solving (see
:attr:`ConjugateGradientSolver.check_symmetry`).

Only one iterative method is implemented in this version -- spec
section 11 is explicit that "an unnecessarily large collection" is out
of scope, and CG alone already covers this toolkit's actual SPD
systems (linear elastic stiffness, thermal conductivity).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from femtoolkit.exceptions import InvalidSolverConfigurationError, SolverConvergenceError
from femtoolkit.solvers.base import (
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_TOLERANCE,
    LinearSolver,
    partition_dofs,
    reduced_system,
    residual_norms,
    timed,
    validate_before_solve,
    validate_solver_settings,
)
from femtoolkit.solvers.results import SolverResult

if TYPE_CHECKING:
    from femtoolkit.analysis.system import LinearSystem

_SYMMETRY_RELATIVE_TOLERANCE = 1e-8
"""Relative tolerance for the cheap ``||A - A^T|| / ||A||`` symmetry check
(spec section 15: "provide appropriate diagnostics when the matrix does
not satisfy the required assumptions"). Chosen loosely enough to accept
normal floating-point assembly round-off, tight enough to reject a
genuinely non-symmetric matrix."""


@dataclass
class ConjugateGradientSolver(LinearSolver):
    """Iterative solve via :func:`scipy.sparse.linalg.cg`.

    Attributes:
        tolerance: Relative-residual convergence tolerance (see
            :func:`~femtoolkit.solvers.base.residual_norms`). Must be
            positive and finite.
        max_iterations: Maximum CG iterations. Must be a positive
            integer.
        check_symmetry: Whether to run the cheap ``O(nnz)``
            ``||A - A^T|| / ||A||`` symmetry check before solving (CG
            requires a symmetric positive-definite matrix; an
            asymmetric matrix will not converge to a meaningful
            solution). Disable only for a very large production solve
            where the caller already knows the matrix is SPD, per spec
            section 15's "do not blindly perform expensive matrix
            checks" -- though this particular check is cheap enough
            (linear in the number of non-zero entries) to leave enabled
            in most cases.
        raise_on_non_convergence: If ``True`` (the default), a solve
            that does not reach ``tolerance`` within ``max_iterations``
            raises :class:`~femtoolkit.exceptions.SolverConvergenceError`.
            If ``False``, a non-converged solve instead returns a
            :class:`~femtoolkit.solvers.results.SolverResult` with
            ``converged=False``, for callers that want to inspect a
            partial result rather than handle an exception.
        track_residual_history: If ``True`` (default ``False``),
            records the relative residual at every iteration into
            ``diagnostics["residual_history"]`` -- used by
            :mod:`femtoolkit.verification` (Version 29) to plot solver
            convergence. Disabled by default because computing a
            residual at every iteration costs one extra sparse
            matrix-vector product per iteration, roughly doubling a
            solve's cost; every existing caller that does not opt in
            pays nothing extra.

    Raises:
        InvalidSolverConfigurationError: If ``tolerance`` or
            ``max_iterations`` is invalid.

    Example:
        >>> result = ConjugateGradientSolver(tolerance=1e-8, max_iterations=1000).solve(system)
        >>> result.iterations, result.converged
        (146, True)
    """

    MATRIX_TYPE: ClassVar[str] = "sparse"

    tolerance: float = DEFAULT_TOLERANCE
    max_iterations: int = DEFAULT_MAX_ITERATIONS
    check_symmetry: bool = True
    raise_on_non_convergence: bool = True
    track_residual_history: bool = False
    _last_iteration_count: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate solver settings immediately after construction.

        Raises:
            InvalidSolverConfigurationError: If ``tolerance`` or
                ``max_iterations`` is invalid.
        """
        validate_solver_settings(self.tolerance, self.max_iterations)

    def solve(self, system: LinearSystem) -> SolverResult:
        """Solve ``system`` with Conjugate Gradient.

        Args:
            system: The linear system to solve. ``system.stiffness``
                must be a :class:`scipy.sparse.spmatrix`.

        Returns:
            A :class:`~femtoolkit.solvers.results.SolverResult` with
            ``iterations`` set to the actual number of CG iterations
            performed.

        Raises:
            InvalidSolverConfigurationError: If ``check_symmetry`` is
                enabled and the matrix is not (to within tolerance)
                symmetric.
            SolverConvergenceError: If :attr:`raise_on_non_convergence`
                is ``True`` and the solve does not reach ``tolerance``
                within ``max_iterations``.
        """
        validate_before_solve(system)
        free, constrained, displacements = partition_dofs(system)

        n = system.dof_map.total_dofs
        nnz = int(system.stiffness.nnz) if sp.issparse(system.stiffness) else 0
        density = nnz / (n * n) if n > 0 else 0.0

        if free.size == 0:
            return SolverResult(
                solution=displacements,
                converged=True,
                iterations=0,
                residual_norm=0.0,
                relative_residual=0.0,
                solve_time=0.0,
                solver_name="Conjugate Gradient",
                diagnostics={"dofs": n, "free_dofs": 0, "nnz": nnz, "density": density},
            )

        k_free_free, reduced_forces = reduced_system(system, free, constrained, displacements)
        k_free_free_csr = k_free_free.tocsr()

        if self.check_symmetry:
            self._validate_symmetric(k_free_free_csr)

        self._last_iteration_count = 0
        residual_history: list[float] = []

        def _count_iteration(xk: np.ndarray) -> None:
            self._last_iteration_count += 1
            if self.track_residual_history:
                _, relative = residual_norms(k_free_free_csr, reduced_forces, xk)
                residual_history.append(relative)

        def _run_cg() -> tuple[np.ndarray, int]:
            return spla.cg(
                k_free_free_csr,
                reduced_forces,
                rtol=self.tolerance,
                maxiter=self.max_iterations,
                callback=_count_iteration,
            )

        (solution_free, info), solve_time = timed(_run_cg)
        iterations = self._last_iteration_count
        residual_norm, relative_residual = residual_norms(
            k_free_free_csr, reduced_forces, solution_free
        )
        converged = info == 0 and relative_residual <= self.tolerance

        if not converged and self.raise_on_non_convergence:
            raise SolverConvergenceError(
                f"Conjugate Gradient failed to converge within {self.max_iterations} "
                f"iterations (reached {iterations} iterations, relative residual "
                f"{relative_residual:.3e}, tolerance {self.tolerance:.3e})."
            )

        displacements[free] = solution_free

        diagnostics = {
            "dofs": n,
            "free_dofs": int(free.size),
            "nnz": nnz,
            "density": density,
        }
        if self.track_residual_history:
            diagnostics["residual_history"] = residual_history

        return SolverResult(
            solution=displacements,
            converged=converged,
            iterations=iterations,
            residual_norm=residual_norm,
            relative_residual=relative_residual,
            solve_time=solve_time,
            solver_name="Conjugate Gradient",
            diagnostics=diagnostics,
        )

    def _validate_symmetric(self, matrix: sp.csr_matrix) -> None:
        """Raise if ``matrix`` is not symmetric to within :data:`_SYMMETRY_RELATIVE_TOLERANCE`.

        Raises:
            InvalidSolverConfigurationError: If the matrix fails the
                symmetry check.
        """
        asymmetry = matrix - matrix.T
        asymmetry_norm = spla.norm(asymmetry) if asymmetry.nnz > 0 else 0.0
        matrix_norm = spla.norm(matrix)
        relative_asymmetry = asymmetry_norm / max(matrix_norm, 1e-30)
        if relative_asymmetry > _SYMMETRY_RELATIVE_TOLERANCE:
            raise InvalidSolverConfigurationError(
                "Conjugate Gradient requires a symmetric positive-definite matrix, "
                f"but the reduced system is not symmetric (relative asymmetry "
                f"{relative_asymmetry:.3e} exceeds {_SYMMETRY_RELATIVE_TOLERANCE:.3e}). "
                "Use SparseDirectSolver or DenseDirectSolver instead, or disable this "
                "check with check_symmetry=False if the matrix is known to be SPD."
            )


__all__ = ["ConjugateGradientSolver"]
