"""Dynamic time-history analysis workflow: ``DynamicAnalysis``.

Mirrors :class:`~femtoolkit.analysis.static_linear.StaticLinearAnalysis`'s
shape deliberately: build one against a mesh, register boundary
conditions and (now time-dependent) loads with ``add_*`` methods, then
call ``solve()``. The static analysis's DOF mapping, stiffness assembly,
and free/constrained reduction strategy (see
:mod:`femtoolkit.analysis.system`) are reused as directly as possible:
this module adds *only* what a static analysis cannot express --
inertia, damping, and time -- rather than a parallel, independent
implementation.

.. code-block:: text

    >>> analysis = DynamicAnalysis(mesh, damping=RayleighDamping(alpha=0.01, beta=0.0001))
    >>> analysis.add_boundary_condition(BoundaryCondition(1, TranslationDOF.X, 0.0))
    >>> analysis.add_time_dependent_load(
    ...     TimeDependentNodalLoad(5, TranslationDOF.Y, SinusoidalLoad(1000.0, 50.0))
    ... )
    >>> result = analysis.solve(time_step=0.001, total_time=1.0)
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from femtoolkit.analysis.boundary_conditions import BoundaryCondition
from femtoolkit.analysis.damping import RayleighDamping
from femtoolkit.analysis.dynamic_loads import TimeDependentNodalLoad
from femtoolkit.analysis.dynamic_system import DynamicSystem, build_dynamic_system
from femtoolkit.analysis.mass import MassMatrixType
from femtoolkit.analysis.newmark import (
    DEFAULT_BETA,
    DEFAULT_GAMMA,
    effective_force,
    effective_stiffness,
    update_velocity_acceleration,
)
from femtoolkit.analysis.system import build_force_vector
from femtoolkit.exceptions import SingularSystemError, ValidationError

if TYPE_CHECKING:
    from femtoolkit.mesh.mesh import Mesh
    from femtoolkit.results.dynamic_result import DynamicResult


class DynamicAnalysis:
    """A Newmark-beta time-history analysis over a mesh of mass-capable elements.

    Example:
        >>> analysis = DynamicAnalysis(mesh, damping=RayleighDamping(alpha=0.01, beta=0.0001))
        >>> bc = BoundaryCondition(node_id=1, dof=TranslationDOF.X, value=0.0)
        >>> analysis.add_boundary_condition(bc)
        >>> load = TimeDependentNodalLoad(node_id=5, dof=TranslationDOF.Y, load=ConstantLoad(1e3))
        >>> analysis.add_time_dependent_load(load)
        >>> result = analysis.solve(time_step=0.001, total_time=1.0)
    """

    def __init__(
        self,
        mesh: Mesh,
        mass_matrix_type: MassMatrixType = "consistent",
        damping: RayleighDamping | None = None,
    ) -> None:
        """Create a dynamic analysis for the given mesh.

        Args:
            mesh: The mesh to analyze. Every element must be
                mass-capable (see :mod:`femtoolkit.analysis.mass`); this
                is checked when :meth:`solve` is called.
            mass_matrix_type: ``"consistent"`` (default) or ``"lumped"``.
            damping: Rayleigh damping to apply, or ``None`` (the
                default) for an undamped system.
        """
        self._mesh = mesh
        self._mass_matrix_type = mass_matrix_type
        self._damping = damping
        self._boundary_conditions: list[BoundaryCondition] = []
        self._time_dependent_loads: list[TimeDependentNodalLoad] = []

    def add_boundary_condition(self, boundary_condition: BoundaryCondition) -> None:
        """Add a prescribed-displacement boundary condition to the analysis.

        The prescribed value is held constant for the entire simulation
        (velocity and acceleration are zero at this DOF throughout) --
        time-varying prescribed displacements (support motion) are out
        of scope for this version.

        Args:
            boundary_condition: The boundary condition to apply.
        """
        self._boundary_conditions.append(boundary_condition)

    def add_time_dependent_load(self, load: TimeDependentNodalLoad) -> None:
        """Add a time-dependent nodal load to the analysis.

        Args:
            load: The time-dependent load to apply.
        """
        self._time_dependent_loads.append(load)

    def build_system(self) -> DynamicSystem:
        """Assemble this analysis's :class:`~femtoolkit.analysis.dynamic_system.DynamicSystem`.

        Returns:
            The assembled :class:`~femtoolkit.analysis.dynamic_system.DynamicSystem`.

        Raises:
            InvalidAnalysisError: If the mesh has no nodes or no elements.
            InvalidElementError: If the mesh contains a non-mass-capable
                element, or elements with inconsistent ``dofs_per_node``.
            ValidationError: If no boundary conditions have been added,
                or an element's material has no density set.
        """
        return build_dynamic_system(
            self._mesh,
            self._boundary_conditions,
            mass_matrix_type=self._mass_matrix_type,
            damping=self._damping,
        )

    def solve(
        self,
        time_step: float,
        total_time: float,
        beta: float = DEFAULT_BETA,
        gamma: float = DEFAULT_GAMMA,
        initial_displacement: np.ndarray | None = None,
        initial_velocity: np.ndarray | None = None,
    ) -> DynamicResult:
        """Run a Newmark-beta time-history simulation.

        Args:
            time_step: Fixed time step, in seconds. Must be positive.
            total_time: Total simulated duration, in seconds. Must be
                positive; the number of steps is ``round(total_time / time_step)``.
            beta: Newmark beta parameter. Defaults to ``0.25`` (average
                acceleration method).
            gamma: Newmark gamma parameter. Defaults to ``0.5``.
            initial_displacement: Initial displacement vector, in global
                DOF order (``dof_map.total_dofs`` entries), or ``None``
                for all-zero initial displacement. Values at constrained
                DOFs are overwritten with their prescribed value.
            initial_velocity: Initial velocity vector, or ``None`` for
                all-zero initial velocity. Values at constrained DOFs are
                overwritten with zero.

        Returns:
            The :class:`~femtoolkit.results.dynamic_result.DynamicResult`.

        Raises:
            InvalidAnalysisError: If the mesh has no nodes or no elements.
            InvalidElementError: If the mesh contains a non-mass-capable element.
            ValidationError: If ``time_step``/``total_time`` are not
                positive, no boundary conditions have been added, or
                every DOF is constrained.
            SingularSystemError: If the Newmark effective stiffness
                matrix is singular.
        """
        # Imported locally to avoid the circular import described in
        # femtoolkit.analysis.static_linear (results -> analysis at
        # module level would create a cycle).
        from femtoolkit.results.dynamic_result import DynamicResult

        if not math.isfinite(time_step) or time_step <= 0:
            raise ValidationError(f"time_step must be positive, got {time_step}.")
        if not math.isfinite(total_time) or total_time <= 0:
            raise ValidationError(f"total_time must be positive, got {total_time}.")

        system = self.build_system()
        dof_map = system.dof_map
        total_dofs = dof_map.total_dofs

        constrained_indices: list[int] = []
        constrained_values: list[float] = []
        for bc in system.boundary_conditions:
            constrained_indices.append(dof_map.global_index(bc.node_id, bc.dof))
            constrained_values.append(bc.value)
        constrained = np.array(constrained_indices, dtype=int)
        constrained_set = set(constrained_indices)
        free = np.array([i for i in range(total_dofs) if i not in constrained_set], dtype=int)

        if free.size == 0:
            raise ValidationError("DynamicAnalysis requires at least one free (unconstrained) DOF.")

        u_c = np.array(constrained_values, dtype=float)
        mass_ff = system.mass[np.ix_(free, free)]
        damping_ff = system.damping[np.ix_(free, free)]
        stiffness_ff = system.stiffness[np.ix_(free, free)]
        stiffness_fc = system.stiffness[np.ix_(free, constrained)] if constrained.size else None
        coupling_force = stiffness_fc @ u_c if constrained.size else 0.0

        n_steps = round(total_time / time_step)
        time = np.array([step * time_step for step in range(n_steps + 1)])

        def force_at(t: float) -> np.ndarray:
            loads = [load.nodal_load_at(t) for load in self._time_dependent_loads]
            return build_force_vector(dof_map, loads)

        u_full = np.zeros(total_dofs) if initial_displacement is None else np.array(
            initial_displacement, dtype=float
        )
        v_full = np.zeros(total_dofs) if initial_velocity is None else np.array(
            initial_velocity, dtype=float
        )
        u_full[constrained] = u_c
        v_full[constrained] = 0.0
        u_f = u_full[free]
        v_f = v_full[free]

        force_0 = force_at(0.0)
        rhs_0 = (force_0[free] - coupling_force) - damping_ff @ v_f - stiffness_ff @ u_f
        a_f = np.linalg.solve(mass_ff, rhs_0)

        displacement_history = np.zeros((n_steps + 1, total_dofs))
        velocity_history = np.zeros((n_steps + 1, total_dofs))
        acceleration_history = np.zeros((n_steps + 1, total_dofs))
        reaction_history = np.zeros((n_steps + 1, total_dofs))

        def record(
            step: int, u_f_vec: np.ndarray, v_f_vec: np.ndarray, a_f_vec: np.ndarray, t: float
        ) -> None:
            full_u = np.zeros(total_dofs)
            full_v = np.zeros(total_dofs)
            full_a = np.zeros(total_dofs)
            full_u[free] = u_f_vec
            full_u[constrained] = u_c
            full_v[free] = v_f_vec
            full_a[free] = a_f_vec
            force_full = force_at(t)
            reaction = (
                system.mass @ full_a + system.damping @ full_v + system.stiffness @ full_u
                - force_full
            )
            displacement_history[step] = full_u
            velocity_history[step] = full_v
            acceleration_history[step] = full_a
            reaction_history[step] = reaction

        record(0, u_f, v_f, a_f, 0.0)

        k_eff_ff = effective_stiffness(mass_ff, damping_ff, stiffness_ff, time_step, beta, gamma)

        for step in range(1, n_steps + 1):
            t_next = time[step]
            force_next = force_at(t_next)
            force_next_f = force_next[free] - coupling_force
            f_eff = effective_force(
                mass_ff, damping_ff, force_next_f, u_f, v_f, a_f, time_step, beta, gamma
            )
            try:
                u_f_next = np.linalg.solve(k_eff_ff, f_eff)
            except np.linalg.LinAlgError as error:
                raise SingularSystemError(
                    "The Newmark effective stiffness matrix is singular; check that "
                    "the dynamic system is properly constrained."
                ) from error

            v_f_next, a_f_next = update_velocity_acceleration(
                u_f, v_f, a_f, u_f_next, time_step, beta, gamma
            )
            record(step, u_f_next, v_f_next, a_f_next, t_next)
            u_f, v_f, a_f = u_f_next, v_f_next, a_f_next

        return DynamicResult(
            dof_map=dof_map,
            time=time,
            displacement_history=displacement_history,
            velocity_history=velocity_history,
            acceleration_history=acceleration_history,
            reaction_history=reaction_history,
        )
