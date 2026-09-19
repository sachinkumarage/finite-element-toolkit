"""Linear solver infrastructure: dense, sparse-direct, and iterative (Version 26).

.. code-block:: text

    Element Matrices
          |
    Global Sparse Assembly    (femtoolkit.analysis.sparse_assembly)
          |
    Boundary Conditions        (femtoolkit.analysis.system.LinearSystem, unchanged)
          |
    Sparse Linear System
          |
    Linear Solver               (this package)
          |
    Solution
          |
    Post-processing              (femtoolkit.postprocessing, unchanged)

Every solver implements the same interface, :meth:`~femtoolkit.solvers.base.LinearSolver.solve`,
operating directly on the existing
:class:`~femtoolkit.analysis.system.LinearSystem`:

.. code-block:: text

    LinearSolver
    |-- DenseDirectSolver        small systems, debugging, verification
    |-- SparseDirectSolver       large sparse systems, exact factorization
    `-- ConjugateGradientSolver  large sparse SPD systems, iterative

:func:`create_solver` is the solver-selection entry point (spec section
21): given a matrix representation (``"dense"``/``"sparse"``) and a
solver name (``"direct"``/``"conjugate_gradient"``), it returns the
matching, already-validated solver instance, raising
:class:`~femtoolkit.exceptions.UnsupportedSolverError` for an unknown or
incompatible combination.
"""

from __future__ import annotations

from femtoolkit.exceptions import UnsupportedSolverError
from femtoolkit.solvers.base import DEFAULT_MAX_ITERATIONS, DEFAULT_TOLERANCE, LinearSolver
from femtoolkit.solvers.dense import DenseDirectSolver
from femtoolkit.solvers.iterative import ConjugateGradientSolver
from femtoolkit.solvers.results import SolverResult
from femtoolkit.solvers.sparse import SparseDirectSolver

MATRIX_TYPES: tuple[str, ...] = ("dense", "sparse")
"""Matrix representations :func:`create_solver` accepts for ``matrix_type``."""

SOLVER_TYPES: tuple[str, ...] = ("direct", "conjugate_gradient")
"""Solver names :func:`create_solver` accepts for ``solver_type``."""


def create_solver(
    matrix_type: str = "dense",
    solver_type: str = "direct",
    tolerance: float = DEFAULT_TOLERANCE,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
) -> LinearSolver:
    """Build a solver instance from a matrix representation and solver name.

    The solver-selection mechanism spec section 21 asks for: a matrix
    type paired with a compatible solver, plus tolerance/max-iterations
    settings for the iterative case (ignored for a direct solve, which
    has no iteration concept).

    Args:
        matrix_type: ``"dense"`` or ``"sparse"``.
        solver_type: ``"direct"`` or ``"conjugate_gradient"``.
        tolerance: Convergence tolerance, used only by
            ``"conjugate_gradient"``.
        max_iterations: Maximum iterations, used only by
            ``"conjugate_gradient"``.

    Returns:
        A :class:`~femtoolkit.solvers.base.LinearSolver` instance:
        :class:`~femtoolkit.solvers.dense.DenseDirectSolver` for
        ``("dense", "direct")``,
        :class:`~femtoolkit.solvers.sparse.SparseDirectSolver` for
        ``("sparse", "direct")``, or
        :class:`~femtoolkit.solvers.iterative.ConjugateGradientSolver`
        for ``("sparse", "conjugate_gradient")``.

    Raises:
        UnsupportedSolverError: If ``matrix_type``/``solver_type`` is
            not one of the values listed above -- in particular,
            ``("dense", "conjugate_gradient")`` is rejected outright:
            this architecture only offers Conjugate Gradient over a
            sparse matrix representation (see
            :mod:`femtoolkit.solvers.iterative`'s module docstring for
            why CG needs a sparse, not dense, matrix to be worthwhile).
        InvalidSolverConfigurationError: If ``tolerance``/``max_iterations``
            is invalid (only checked when ``solver_type`` is
            ``"conjugate_gradient"``).
    """
    if matrix_type not in MATRIX_TYPES:
        raise UnsupportedSolverError(
            f"Unknown matrix_type {matrix_type!r}; expected one of {MATRIX_TYPES}."
        )
    if solver_type not in SOLVER_TYPES:
        raise UnsupportedSolverError(
            f"Unknown solver_type {solver_type!r}; expected one of {SOLVER_TYPES}."
        )

    if matrix_type == "dense" and solver_type == "direct":
        return DenseDirectSolver()
    if matrix_type == "sparse" and solver_type == "direct":
        return SparseDirectSolver()
    if matrix_type == "sparse" and solver_type == "conjugate_gradient":
        return ConjugateGradientSolver(tolerance=tolerance, max_iterations=max_iterations)

    raise UnsupportedSolverError(
        f"Unsupported solver combination: matrix_type={matrix_type!r}, "
        f"solver_type={solver_type!r}. Conjugate Gradient requires a sparse matrix "
        "representation in this toolkit's architecture."
    )


__all__ = [
    "DEFAULT_MAX_ITERATIONS",
    "DEFAULT_TOLERANCE",
    "MATRIX_TYPES",
    "SOLVER_TYPES",
    "ConjugateGradientSolver",
    "DenseDirectSolver",
    "LinearSolver",
    "SolverResult",
    "SparseDirectSolver",
    "create_solver",
]
