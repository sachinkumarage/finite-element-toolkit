"""The dense direct linear solver (Version 26, spec section 9).

Wraps :func:`numpy.linalg.solve` on the free-free reduced system --
numerically the exact same computation
:func:`~femtoolkit.analysis.system.solve` (Version 2, unchanged) has
always performed. :class:`DenseDirectSolver` is the toolkit's default
and remains fully available for small systems, debugging, verification
(every sparse/iterative result in this version is checked against it),
and the educational examples that came before Version 26 -- introducing
sparse solvers is additive, not a replacement.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import numpy as np

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


class DenseDirectSolver(LinearSolver):
    """Direct solve via :func:`numpy.linalg.solve` on a dense reduced system.

    Example:
        >>> result = DenseDirectSolver().solve(system)
        >>> result.solution
        array([...])
    """

    MATRIX_TYPE: ClassVar[str] = "dense"

    def solve(self, system: LinearSystem) -> SolverResult:
        """Solve ``system`` with a dense LU-based direct solve.

        Args:
            system: The linear system to solve. ``system.stiffness``
                must be a dense :class:`numpy.ndarray`.

        Returns:
            A :class:`~femtoolkit.solvers.results.SolverResult` with
            ``converged=True`` and ``iterations=None`` (a direct solve
            has no iteration count).

        Raises:
            SingularSystemError: If the reduced free-free stiffness
                matrix is singular.
        """
        validate_before_solve(system)
        free, constrained, displacements = partition_dofs(system)

        if free.size == 0:
            return SolverResult(
                solution=displacements,
                converged=True,
                iterations=None,
                residual_norm=0.0,
                relative_residual=0.0,
                solve_time=0.0,
                solver_name="Dense Direct",
                diagnostics={"dofs": system.dof_map.total_dofs, "free_dofs": 0},
            )

        k_free_free, reduced_forces = reduced_system(system, free, constrained, displacements)

        try:
            solution_free, solve_time = timed(lambda: np.linalg.solve(k_free_free, reduced_forces))
        except np.linalg.LinAlgError as error:
            raise SingularSystemError(
                "The reduced free-free stiffness matrix is singular: the structure "
                "is insufficiently constrained (e.g. a free mechanism) even though "
                "boundary conditions were provided."
            ) from error

        residual_norm, relative_residual = residual_norms(
            k_free_free, reduced_forces, solution_free
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
            solver_name="Dense Direct",
            diagnostics={
                "dofs": system.dof_map.total_dofs,
                "free_dofs": int(free.size),
            },
        )


__all__ = ["DenseDirectSolver"]
