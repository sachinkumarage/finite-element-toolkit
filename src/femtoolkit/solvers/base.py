"""The solver abstraction: a common interface over dense, sparse-direct, and
iterative linear solvers (Version 26, spec section 8).

.. code-block:: text

    LinearSolver
    |-- DenseDirectSolver        (femtoolkit.solvers.dense)
    |-- SparseDirectSolver       (femtoolkit.solvers.sparse)
    `-- ConjugateGradientSolver  (femtoolkit.solvers.iterative)

Every concrete solver implements one method, ``solve(system) ->
SolverResult``, operating directly on the existing
:class:`~femtoolkit.analysis.system.LinearSystem` -- reusing it rather
than inventing a second "matrix + RHS + boundary conditions" container.
``LinearSystem.stiffness`` may be a dense :class:`numpy.ndarray` (as it
always was through Version 25) or, since this version, a
:class:`scipy.sparse.spmatrix`; :class:`LinearSystem` itself needed no
changes to support this, since its own validation only inspects
``.shape``, which both dense and sparse matrices expose identically.

This module holds the logic every concrete solver shares: partitioning
DOFs into free/constrained sets from a system's boundary conditions
(the same elimination approach
:func:`~femtoolkit.analysis.system.solve` already uses), slicing out the
free-free reduced system (dispatching on whether the matrix is dense or
sparse), computing the residual of the system that was actually solved,
and a few cheap pre-solve sanity checks. Concrete solvers call these
helpers rather than repeating the partition/reduce/residual logic three
times.
"""

from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, ClassVar, TypeVar

import numpy as np
import scipy.sparse as sp

from femtoolkit.exceptions import (
    InvalidSolverConfigurationError,
    SingularSystemError,
    SolverError,
)
from femtoolkit.solvers.results import SolverResult

if TYPE_CHECKING:
    from femtoolkit.analysis.system import LinearSystem

_RESIDUAL_FLOOR = 1e-30
"""Smallest denominator used in a relative-residual computation, avoiding
division by zero for a (degenerate) all-zero right-hand side."""

DEFAULT_TOLERANCE: float = 1e-8
"""Default relative-residual convergence tolerance for an iterative solver."""

DEFAULT_MAX_ITERATIONS: int = 1000
"""Default maximum iteration count for an iterative solver."""

_MatrixLike = np.ndarray | sp.spmatrix
_T = TypeVar("_T")


class LinearSolver(ABC):
    """Common interface for every linear-system solver strategy.

    Attributes:
        MATRIX_TYPE: ``"dense"`` or ``"sparse"`` -- which matrix
            representation this solver expects
            ``system.stiffness`` to already be in. A caller assembling
            the system (e.g.
            :class:`~femtoolkit.analysis.static_linear.StaticLinearAnalysis`)
            reads this to decide whether to call
            :func:`~femtoolkit.analysis.assembly.assemble_global_stiffness`
            or
            :func:`~femtoolkit.analysis.sparse_assembly.assemble_global_stiffness_sparse`.
    """

    MATRIX_TYPE: ClassVar[str]

    @abstractmethod
    def solve(self, system: LinearSystem) -> SolverResult:
        """Solve ``system`` and return a full diagnostic result.

        Args:
            system: The linear system to solve, including its boundary
                conditions.

        Returns:
            A :class:`~femtoolkit.solvers.results.SolverResult`.

        Raises:
            SingularSystemError: If the reduced free-free system is
                singular.
            SolverConvergenceError: If an iterative solver fails to
                converge within its configured iteration limit.
        """


def partition_dofs(system: LinearSystem) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split a system's global DOFs into free and constrained sets.

    The same elimination approach
    :func:`~femtoolkit.analysis.system.solve` uses: every boundary
    condition names one constrained (prescribed-value) DOF; every other
    DOF is free (unknown).

    Args:
        system: The linear system to partition.

    Returns:
        ``(free_indices, constrained_indices, displacements)`` --
        ``displacements`` is a full-length vector with every
        constrained entry already filled in with its prescribed value
        and every free entry still zero (to be overwritten by the
        solve).
    """
    n = system.dof_map.total_dofs
    displacements = np.zeros(n)

    constrained_indices = []
    for boundary_condition in system.boundary_conditions:
        global_index = system.dof_map.global_index(
            boundary_condition.node_id, boundary_condition.dof
        )
        constrained_indices.append(global_index)
        displacements[global_index] = boundary_condition.value

    constrained = np.array(constrained_indices, dtype=int)
    free = np.array([i for i in range(n) if i not in set(constrained_indices)], dtype=int)
    return free, constrained, displacements


def reduced_system(
    system: LinearSystem, free: np.ndarray, constrained: np.ndarray, displacements: np.ndarray
) -> tuple[_MatrixLike, np.ndarray]:
    """Slice out the free-free reduced stiffness matrix and reduced force vector.

    Dispatches on whether ``system.stiffness`` is a dense
    :class:`numpy.ndarray` or a :class:`scipy.sparse.spmatrix`; either
    way, the eliminated (constrained) DOFs' known contribution is moved
    to the right-hand side, exactly as
    :func:`~femtoolkit.analysis.system.solve` already does for the
    dense case.

    Args:
        system: The linear system being solved.
        free: Free (unknown) global DOF indices.
        constrained: Constrained (prescribed-value) global DOF indices.
        displacements: Full-length vector with constrained entries
            already filled in (see :func:`partition_dofs`).

    Returns:
        ``(k_free_free, reduced_forces)``, with ``k_free_free`` the same
        type (dense or sparse) as ``system.stiffness``.
    """
    if sp.issparse(system.stiffness):
        stiffness_csr = system.stiffness.tocsr()
        k_free_free = stiffness_csr[free, :][:, free]
        k_free_constrained = stiffness_csr[free, :][:, constrained]
    else:
        k_free_free = system.stiffness[np.ix_(free, free)]
        k_free_constrained = system.stiffness[np.ix_(free, constrained)]

    reduced_forces = system.forces[free] - k_free_constrained @ displacements[constrained]
    return k_free_free, reduced_forces


def residual_norms(
    k_free_free: _MatrixLike, reduced_forces: np.ndarray, solution_free: np.ndarray
) -> tuple[float, float]:
    r"""Compute the residual of the reduced system that was actually solved.

    .. math::

        \mathbf{r} = \mathbf{b} - \mathbf{A}\mathbf{x}, \qquad
        r_{rel} = \frac{\lVert \mathbf{r} \rVert}{\max(\lVert \mathbf{b} \rVert, \epsilon)}

    Args:
        k_free_free: The reduced (free-free) matrix that was solved,
            dense or sparse.
        reduced_forces: The reduced right-hand side ``b``.
        solution_free: The computed solution ``x`` on the free DOFs.

    Returns:
        ``(residual_norm, relative_residual)``.
    """
    residual = reduced_forces - k_free_free @ solution_free
    residual_norm = float(np.linalg.norm(residual))
    rhs_norm = float(np.linalg.norm(reduced_forces))
    relative_residual = residual_norm / max(rhs_norm, _RESIDUAL_FLOOR)
    return residual_norm, relative_residual


_DIRECT_SOLVE_SINGULARITY_THRESHOLD = 1e-4
"""Relative-residual threshold above which a direct solver's own result is
treated as evidence of a singular (or numerically indistinguishable from
singular) reduced system.

A dense LU factorization (:func:`numpy.linalg.solve`) or a sparse LU
factorization (:func:`scipy.sparse.linalg.spsolve`) only raises an
exception for an *exactly* singular matrix (an exact zero pivot); a
matrix that is singular in exact arithmetic but has merely a very small
(rather than exactly zero) pivot after floating-point round-off --
exactly what a free rigid-body mechanism with more than one
under-constrained DOF produces -- factorizes and "solves" without
raising, returning a physically meaningless huge-magnitude vector. A
genuinely solved system has a relative residual at or near machine
epsilon (``1e-10`` to ``1e-15``); this threshold is chosen many orders
of magnitude looser than that, so it never misfires on a legitimately
solved, merely ill-conditioned system, while still reliably catching
the "huge garbage solution" signature of an undetected singular solve
(verified directly against a real under-constrained mesh in
``tests/test_solvers.py``)."""


def check_direct_solve_residual(relative_residual: float) -> None:
    """Raise if a direct solve's own relative residual indicates an undetected singularity.

    Call this after :func:`residual_norms` in any *direct* solver (not
    an iterative one, which already has its own convergence check) --
    see :data:`_DIRECT_SOLVE_SINGULARITY_THRESHOLD` for why this
    catches cases :func:`numpy.linalg.solve`/:func:`scipy.sparse.linalg.spsolve`
    do not reliably raise on themselves.

    Args:
        relative_residual: The relative residual computed by
            :func:`residual_norms` for the direct solve just performed.

    Raises:
        SingularSystemError: If ``relative_residual`` exceeds
            :data:`_DIRECT_SOLVE_SINGULARITY_THRESHOLD`.
    """
    if relative_residual > _DIRECT_SOLVE_SINGULARITY_THRESHOLD:
        raise SingularSystemError(
            "The reduced free-free stiffness matrix is singular (or numerically "
            "indistinguishable from singular): the direct solve produced a relative "
            f"residual of {relative_residual:.3e}, far larger than a correctly solved "
            "system's. The structure is likely insufficiently constrained (e.g. a "
            "free mechanism) even though boundary conditions were provided."
        )


def validate_before_solve(system: LinearSystem) -> None:
    """Run cheap pre-solve sanity checks common to every solver (spec section 15).

    Deliberately limited to checks that cost ``O(nnz)`` or less (never a
    full-matrix scan, condition-number estimate, or factorization
    attempt) -- spec section 15 explicitly warns against "blindly
    perform[ing] expensive matrix checks on every large production
    solve." Singular-system detection is left to each solver's own
    numerical routine (the standard way a direct/iterative solver
    discovers singularity is by attempting the solve).

    Args:
        system: The linear system to check.

    Raises:
        SolverError: If the stiffness matrix or force vector contains a
            NaN or infinite value, or the system has zero degrees of
            freedom.
    """
    if system.dof_map.total_dofs == 0:
        raise SolverError("Cannot solve a system with zero degrees of freedom.")

    if sp.issparse(system.stiffness):
        matrix_finite = bool(np.all(np.isfinite(system.stiffness.data)))
    else:
        matrix_finite = bool(np.all(np.isfinite(system.stiffness)))
    if not matrix_finite:
        raise SolverError("The stiffness matrix contains a NaN or infinite value.")

    if not np.all(np.isfinite(system.forces)):
        raise SolverError("The force vector contains a NaN or infinite value.")


def validate_solver_settings(tolerance: float, max_iterations: int) -> None:
    """Validate common iterative-solver settings (spec section 15).

    Args:
        tolerance: The requested convergence tolerance. Must be positive
            and finite.
        max_iterations: The requested maximum iteration count. Must be a
            positive integer.

    Raises:
        InvalidSolverConfigurationError: If either setting is invalid.
    """
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise InvalidSolverConfigurationError(
            f"Solver tolerance must be positive and finite, got {tolerance}."
        )
    invalid_type = not isinstance(max_iterations, int) or isinstance(max_iterations, bool)
    if invalid_type or max_iterations < 1:
        raise InvalidSolverConfigurationError(
            f"Solver max_iterations must be a positive integer, got {max_iterations!r}."
        )


def timed(func: Callable[[], _T]) -> tuple[_T, float]:
    """Return ``(result, elapsed_seconds)`` for a zero-argument callable.

    A tiny shared timing helper so every concrete solver measures its
    own numerical routine's wall-clock time the same way (spec section
    14's "Solve time" diagnostic).
    """
    start = time.perf_counter()
    result = func()
    elapsed = time.perf_counter() - start
    return result, elapsed


__all__ = [
    "DEFAULT_MAX_ITERATIONS",
    "DEFAULT_TOLERANCE",
    "LinearSolver",
    "check_direct_solve_residual",
    "partition_dofs",
    "reduced_system",
    "residual_norms",
    "timed",
    "validate_before_solve",
    "validate_solver_settings",
]
