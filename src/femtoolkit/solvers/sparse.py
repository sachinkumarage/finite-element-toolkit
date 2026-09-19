"""The sparse direct linear solver (Version 26, spec section 10).

Solves ``K u = F`` via :func:`scipy.sparse.linalg.spsolve` (SciPy's
sparse LU factorization, SuperLU under the hood) on the free-free
reduced system, **without ever converting the matrix to dense form**.
For a large, sparse system, this is the direct-solve counterpart to
:class:`~femtoolkit.solvers.dense.DenseDirectSolver` -- same
"factorize, then back-substitute" strategy, sized to the matrix's
actual non-zero structure rather than its full ``N x N`` extent.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from femtoolkit.exceptions import SingularSystemError
from femtoolkit.solvers.base import (
    LinearSolver,
    check_direct_solve_residual,
    partition_dofs,
    reduced_system,
    residual_norms,
    timed,
    validate_before_solve,
)
from femtoolkit.solvers.results import SolverResult

if TYPE_CHECKING:
    from femtoolkit.analysis.system import LinearSystem


class SparseDirectSolver(LinearSolver):
    """Direct solve via :func:`scipy.sparse.linalg.spsolve` on a sparse reduced system.

    Example:
        >>> result = SparseDirectSolver().solve(system)
        >>> result.diagnostics["nnz"], result.diagnostics["density"]
        (85420, 0.00043)
    """

    MATRIX_TYPE: ClassVar[str] = "sparse"

    def solve(self, system: LinearSystem) -> SolverResult:
        """Solve ``system`` with a sparse LU-based direct solve.

        Args:
            system: The linear system to solve. ``system.stiffness``
                must be a :class:`scipy.sparse.spmatrix`.

        Returns:
            A :class:`~femtoolkit.solvers.results.SolverResult` with
            ``converged=True`` and ``iterations=None``, plus
            ``diagnostics["nnz"]``/``diagnostics["density"]`` describing
            the assembled matrix's sparsity.

        Raises:
            SingularSystemError: If the reduced free-free stiffness
                matrix is singular (detected as a non-finite solution,
                since SuperLU does not always raise outright on a
                singular matrix).
        """
        validate_before_solve(system)
        free, constrained, displacements = partition_dofs(system)

        nnz = int(system.stiffness.nnz) if sp.issparse(system.stiffness) else 0
        n = system.dof_map.total_dofs
        density = nnz / (n * n) if n > 0 else 0.0

        if free.size == 0:
            return SolverResult(
                solution=displacements,
                converged=True,
                iterations=None,
                residual_norm=0.0,
                relative_residual=0.0,
                solve_time=0.0,
                solver_name="Sparse Direct",
                diagnostics={"dofs": n, "free_dofs": 0, "nnz": nnz, "density": density},
            )

        k_free_free, reduced_forces = reduced_system(system, free, constrained, displacements)
        k_free_free_csc = k_free_free.tocsc()

        solution_free, solve_time = timed(lambda: spla.spsolve(k_free_free_csc, reduced_forces))

        if not np.all(np.isfinite(solution_free)):
            raise SingularSystemError(
                "The reduced free-free stiffness matrix is singular: the structure "
                "is insufficiently constrained (e.g. a free mechanism) even though "
                "boundary conditions were provided."
            )

        residual_norm, relative_residual = residual_norms(
            k_free_free_csc, reduced_forces, solution_free
        )
        check_direct_solve_residual(relative_residual)
        displacements[free] = solution_free

        return SolverResult(
            solution=displacements,
            converged=True,
            iterations=None,
            residual_norm=residual_norm,
            relative_residual=relative_residual,
            solve_time=solve_time,
            solver_name="Sparse Direct",
            diagnostics={
                "dofs": n,
                "free_dofs": int(free.size),
                "nnz": nnz,
                "density": density,
            },
        )


__all__ = ["SparseDirectSolver"]
