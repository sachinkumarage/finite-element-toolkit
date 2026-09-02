"""Nonlinear (Newton-Raphson) structural analysis workflow.

Linear analysis (:class:`~femtoolkit.analysis.static_linear.StaticLinearAnalysis`)
solves ``[K]{u} = {F}`` directly: stiffness is fixed, so a single linear
solve gives the exact answer. A nonlinear material breaks that
assumption -- stress (and therefore the element's resisting force) is no
longer a fixed matrix times displacement. The equilibrium statement
becomes a *residual* that must be driven to zero:

.. code-block:: text

    R(u) = F_ext - F_int(u) = 0

which has no closed-form solution in general. :class:`NonlinearAnalysis`
solves it with **Newton-Raphson iteration**: linearize ``F_int`` about
the current trial displacement using the **tangent stiffness**
``K_t = dF_int/du``, solve the linear correction ``K_t @ du = R``, and
repeat until the residual (or the correction itself) is small enough:

.. code-block:: text

    K_t(u_i) @ du = R(u_i)
    u_(i+1) = u_i + du

Because Newton-Raphson's linearization is only valid for a small step,
the total load is applied gradually across several **load steps**
(``NonlinearSolverSettings.load_steps``): each step scales both the
external force and any prescribed boundary-condition displacement by a
load factor in ``(0, 1]`` and re-runs Newton-Raphson from the previous
step's *converged* displacement. See :mod:`femtoolkit.analysis.convergence`
for the two supported convergence criteria.

**Trial vs. committed state, at the orchestration level.** Every
Newton-Raphson iteration evaluates each element's (and each element's
material's) response at a *trial* displacement via
:func:`~femtoolkit.analysis.nonlinear_elements.element_internal_force_and_tangent`,
always measured from the load step's *committed* (start-of-step) state
-- never from the previous iteration's trial. Only once a load step's
iterations satisfy the convergence criterion does this module overwrite
its committed displacement and committed element states with that
step's trial values (see the module docstring for
:mod:`femtoolkit.materials.nonlinear` for why this is safe by
construction). If a load step exhausts its iteration budget without
converging, nothing is committed: :meth:`NonlinearAnalysis.solve` raises
:class:`~femtoolkit.exceptions.NonlinearConvergenceError`, carrying every
step result completed so far (including the failed step's own final,
uncommitted attempt) as its ``step_results`` attribute. Per this
version's scope, no automatic load-step cutback is attempted on
failure -- catching the exception and retrying with a
:class:`NonlinearSolverSettings` built with more ``load_steps`` (a
smaller load increment) is a clean, available extension point without
this module needing to implement adaptive stepping itself.

**Materials are supplied separately from the mesh.** Unlike
``StaticLinearAnalysis``, which reads each element's own (linear)
``material`` attribute, ``NonlinearAnalysis`` takes an explicit
``{element_id: NonlinearMaterial}`` mapping. This keeps a mesh reusable
for both a linear and a nonlinear analysis at once (the element's own
linear ``material`` is never read here) and matches
:mod:`femtoolkit.analysis.nonlinear_elements`'s CST/Q4 support, which
only accepts a :class:`~femtoolkit.materials.nonlinear.NonlinearMaterial`.

Reused, not reimplemented: DOF numbering (:class:`~femtoolkit.analysis.dof.DOFMap`),
boundary conditions (:class:`~femtoolkit.analysis.boundary_conditions.BoundaryCondition`),
loads (:class:`~femtoolkit.analysis.loads.NodalLoad`,
:func:`~femtoolkit.analysis.system.build_force_vector`), and the
free/constrained stiffness partition idea from
:func:`~femtoolkit.analysis.system.solve` -- only the assembled matrix
changes every iteration here, not the partition strategy.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from femtoolkit.analysis.assembly import (
    ElementForceContribution,
    ElementStiffnessContribution,
    assemble_global_internal_force,
    assemble_global_stiffness,
)
from femtoolkit.analysis.boundary_conditions import BoundaryCondition
from femtoolkit.analysis.convergence import (
    ConvergenceCriterion,
    displacement_correction_ratio,
    residual_norm_ratio,
)
from femtoolkit.analysis.dof import DOFMap
from femtoolkit.analysis.loads import NodalLoad
from femtoolkit.analysis.nonlinear_elements import (
    NONLINEAR_CAPABLE_ELEMENT_TYPES,
    NonlinearElementState,
    element_internal_force_and_tangent,
    initial_element_state,
)
from femtoolkit.analysis.system import build_force_vector
from femtoolkit.exceptions import (
    InsufficientConstraintsError,
    InvalidAnalysisError,
    InvalidElementError,
    NonlinearConvergenceError,
    SingularSystemError,
    ValidationError,
)
from femtoolkit.materials.nonlinear import NonlinearMaterial

if TYPE_CHECKING:
    # Imported only for type checking, to avoid circular imports at
    # runtime -- see the same pattern in
    # femtoolkit.analysis.static_linear.
    from femtoolkit.analysis.nonlinear_elements import NonlinearCapableElement
    from femtoolkit.mesh.mesh import Mesh
    from femtoolkit.results.nonlinear_result import LoadStepResult, NonlinearAnalysisResult


@dataclass(frozen=True)
class NonlinearSolverSettings:
    """Configuration for a :class:`NonlinearAnalysis` Newton-Raphson solve.

    Attributes:
        max_iterations: Maximum Newton-Raphson iterations allowed per
            load step before it is reported as a convergence failure.
        tolerance: Convergence tolerance for the chosen ``convergence``
            criterion (a dimensionless ratio, see
            :mod:`femtoolkit.analysis.convergence`).
        load_steps: Number of equal load increments the total applied
            load (and any nonzero prescribed boundary-condition value)
            is divided into.
        convergence: Which criterion decides convergence: ``"residual"``
            (default, ``||R||/||F_ext|| < tolerance``) or
            ``"displacement"`` (``||du||/||u|| < tolerance``).

    Raises:
        ValidationError: If any field fails validation.

    Example:
        >>> settings = NonlinearSolverSettings(
        ...     max_iterations=30, tolerance=1e-8, load_steps=20, convergence="residual"
        ... )
    """

    max_iterations: int = 25
    tolerance: float = 1e-8
    load_steps: int = 10
    convergence: ConvergenceCriterion = "residual"

    def __post_init__(self) -> None:
        """Validate every setting immediately after construction.

        Raises:
            ValidationError: If ``max_iterations`` or ``load_steps`` is
                not a positive integer, ``tolerance`` is not a positive
                finite number, or ``convergence`` is not ``"residual"``
                or ``"displacement"``.
        """
        if (
            not isinstance(self.max_iterations, int)
            or isinstance(self.max_iterations, bool)
            or self.max_iterations <= 0
        ):
            raise ValidationError(
                "NonlinearSolverSettings max_iterations must be a positive integer, got "
                f"{self.max_iterations!r}."
            )
        if not math.isfinite(self.tolerance) or self.tolerance <= 0:
            raise ValidationError(
                f"NonlinearSolverSettings tolerance must be positive, got {self.tolerance}."
            )
        if (
            not isinstance(self.load_steps, int)
            or isinstance(self.load_steps, bool)
            or self.load_steps <= 0
        ):
            raise ValidationError(
                "NonlinearSolverSettings load_steps must be a positive integer, got "
                f"{self.load_steps!r}."
            )
        if self.convergence not in ("residual", "displacement"):
            raise ValidationError(
                'NonlinearSolverSettings convergence must be "residual" or "displacement", '
                f"got {self.convergence!r}."
            )


class NonlinearAnalysis:
    """A Newton-Raphson nonlinear structural analysis over a mesh of continuum elements.

    Supports :class:`~femtoolkit.mesh.cst_element.CSTElement2D`,
    :class:`~femtoolkit.mesh.quad_element.QuadElement2D`,
    :class:`~femtoolkit.mesh.tet4_element.Tet4Element3D`, and
    :class:`~femtoolkit.mesh.hex8_element.Hex8Element3D` elements (see
    :mod:`femtoolkit.analysis.nonlinear_elements`), each driven by its
    own :class:`~femtoolkit.materials.nonlinear.NonlinearMaterial`
    supplied through the ``materials`` mapping -- independent of, and
    never reading, the element's own linear ``material`` attribute.

    Example:
        >>> materials = {
        ...     element.id: ElasticPerfectlyPlasticMaterial1D(2e11, 2.5e8)
        ...     for element in mesh.elements
        ... }
        >>> analysis = NonlinearAnalysis(mesh, materials, NonlinearSolverSettings(load_steps=10))
        >>> analysis.add_boundary_condition(BoundaryCondition(node_id=1, dof=0, value=0.0))
        >>> analysis.add_load(NodalLoad(node_id=2, dof=0, value=1.0e5))
        >>> result = analysis.solve()
    """

    def __init__(
        self,
        mesh: Mesh,
        materials: Mapping[int, NonlinearMaterial],
        settings: NonlinearSolverSettings | None = None,
    ) -> None:
        """Create a nonlinear analysis for the given mesh.

        Args:
            mesh: The mesh to analyze. Its elements must all be
                CST or Q4 continuum elements and share the same
                ``dofs_per_node``; this is checked when :meth:`solve` is
                called.
            materials: Maps each element's ``id`` to the
                :class:`~femtoolkit.materials.nonlinear.NonlinearMaterial`
                it should be analyzed with. Every element in ``mesh``
                must have a matching entry.
            settings: Newton-Raphson solver configuration. Defaults to
                :class:`NonlinearSolverSettings`'s defaults.
        """
        self._mesh = mesh
        self._materials = dict(materials)
        self._settings = settings if settings is not None else NonlinearSolverSettings()
        self._loads: list[NodalLoad] = []
        self._boundary_conditions: list[BoundaryCondition] = []

    def add_load(self, load: NodalLoad) -> None:
        """Add a nodal load (applied at full load factor 1.0) to the analysis.

        Args:
            load: The nodal load to apply.
        """
        self._loads.append(load)

    def add_boundary_condition(self, boundary_condition: BoundaryCondition) -> None:
        """Add a prescribed-displacement boundary condition to the analysis.

        A nonzero ``boundary_condition.value`` is scaled by the same load
        factor as external loads at each load step, so it is reached
        gradually rather than applied all at once.

        Args:
            boundary_condition: The boundary condition to apply.
        """
        self._boundary_conditions.append(boundary_condition)

    def _assemble(
        self,
        dof_map: DOFMap,
        elements: Sequence[NonlinearCapableElement],
        displacements: np.ndarray,
        committed_states: dict[int, NonlinearElementState],
    ) -> tuple[np.ndarray, np.ndarray, dict[int, NonlinearElementState]]:
        """Assemble the global internal-force vector and tangent stiffness at a trial displacement.

        Args:
            dof_map: The analysis's DOF map.
            elements: The mesh's elements.
            displacements: The current trial global displacement vector.
            committed_states: Each element's state at the start of the
                current load step (read-only; never mutated).

        Returns:
            ``(f_int_global, k_t_global, trial_states)``: the global
            internal-force vector, the global tangent stiffness matrix,
            and each element's new (not yet committed) state.
        """
        force_contributions = []
        stiffness_contributions = []
        trial_states: dict[int, NonlinearElementState] = {}

        for element in elements:
            dof_keys = element.dof_keys()
            local_indices = [dof_map.global_index(node_id, dof) for node_id, dof in dof_keys]
            local_displacements = displacements[local_indices]

            f_int_local, k_t_local, trial_state = element_internal_force_and_tangent(
                element,
                self._materials[element.id],
                local_displacements,
                committed_states[element.id],
            )
            force_contributions.append(ElementForceContribution(dof_keys, f_int_local))
            stiffness_contributions.append(ElementStiffnessContribution(dof_keys, k_t_local))
            trial_states[element.id] = trial_state

        f_int_global = assemble_global_internal_force(dof_map, force_contributions)
        k_t_global = assemble_global_stiffness(dof_map, stiffness_contributions)
        return f_int_global, k_t_global, trial_states

    def solve(self) -> NonlinearAnalysisResult:
        """Run the incremental Newton-Raphson solve for this analysis.

        Returns:
            The :class:`~femtoolkit.results.nonlinear_result.NonlinearAnalysisResult`
            containing every load step's displacement, reactions,
            residual norm, iteration count, convergence status, and
            per-element material state.

        Raises:
            InvalidAnalysisError: If the mesh has no nodes or no elements.
            InvalidElementError: If the mesh contains an element type
                other than CST/Q4, or elements with inconsistent
                ``dofs_per_node``.
            ValidationError: If an element has no matching entry in the
                ``materials`` mapping, or two boundary conditions target
                the same DOF.
            InsufficientConstraintsError: If no boundary conditions have
                been added.
            SingularSystemError: If a load step's reduced tangent
                stiffness matrix is singular.
            NonlinearConvergenceError: If a load step fails to converge
                within ``settings.max_iterations``. The exception's
                ``step_results`` attribute holds every step result
                completed so far, including the failed step's own final
                attempt.
        """
        from femtoolkit.results.nonlinear_result import LoadStepResult, NonlinearAnalysisResult

        nodes = self._mesh.nodes
        elements = self._mesh.elements

        if not nodes:
            raise InvalidAnalysisError("Cannot solve a nonlinear analysis whose mesh has no nodes.")
        if not elements:
            raise InvalidAnalysisError(
                "Cannot solve a nonlinear analysis whose mesh has no elements."
            )

        for element in elements:
            if not isinstance(element, NONLINEAR_CAPABLE_ELEMENT_TYPES):
                raise InvalidElementError(
                    "NonlinearAnalysis only supports CSTElement2D, QuadElement2D, "
                    "Tet4Element3D, and Hex8Element3D elements, "
                    f"got {type(element).__name__} (id={element.id})."
                )
            if element.id not in self._materials:
                raise ValidationError(
                    f"No nonlinear material assigned for element id {element.id}: every "
                    "element in the mesh must have a matching entry in the `materials` "
                    "mapping passed to NonlinearAnalysis."
                )

        dofs_per_node_values = {element.dofs_per_node for element in elements}
        if len(dofs_per_node_values) > 1:
            raise InvalidElementError(
                "NonlinearAnalysis requires all elements in a mesh to use the same number "
                f"of DOFs per node, got: {sorted(dofs_per_node_values)}."
            )
        dofs_per_node = dofs_per_node_values.pop()

        if not self._boundary_conditions:
            raise InsufficientConstraintsError(
                "NonlinearAnalysis requires at least one boundary condition."
            )

        dof_map = DOFMap(node_ids=[node.id for node in nodes], dofs_per_node=dofs_per_node)
        total_dofs = dof_map.total_dofs

        constrained_targets: dict[int, float] = {}
        for boundary_condition in self._boundary_conditions:
            global_index = dof_map.global_index(
                boundary_condition.node_id, boundary_condition.dof
            )
            if global_index in constrained_targets:
                raise ValidationError(
                    "Multiple boundary conditions target the same DOF "
                    f"(node_id={boundary_condition.node_id}, dof={boundary_condition.dof})."
                )
            constrained_targets[global_index] = boundary_condition.value

        free = np.array(
            [i for i in range(total_dofs) if i not in constrained_targets], dtype=int
        )
        external_force_total = build_force_vector(dof_map, self._loads)

        u_committed = np.zeros(total_dofs)
        committed_states: dict[int, NonlinearElementState] = {
            element.id: initial_element_state(element, self._materials[element.id])
            for element in elements
        }

        settings = self._settings
        step_results: list[LoadStepResult] = []

        for step in range(1, settings.load_steps + 1):
            load_factor = step / settings.load_steps
            external_force = load_factor * external_force_total

            u_trial = u_committed.copy()
            for global_index, target_value in constrained_targets.items():
                u_trial[global_index] = load_factor * target_value

            trial_states = dict(committed_states)
            converged = free.size == 0
            residual_ratio = 0.0
            iterations_used = 0
            f_int_global = np.zeros(total_dofs)

            for iteration in range(1, settings.max_iterations + 1):
                if free.size == 0:
                    break

                iterations_used = iteration
                f_int_global, k_t_global, trial_states = self._assemble(
                    dof_map, elements, u_trial, committed_states
                )
                residual = external_force - f_int_global
                residual_ratio = residual_norm_ratio(residual[free], external_force[free])

                if settings.convergence == "residual" and residual_ratio < settings.tolerance:
                    converged = True
                    break

                k_t_free_free = k_t_global[np.ix_(free, free)]
                try:
                    delta_u_free = np.linalg.solve(k_t_free_free, residual[free])
                except np.linalg.LinAlgError as error:
                    raise SingularSystemError(
                        f"The tangent stiffness matrix is singular at load step {step}, "
                        f"iteration {iteration}: the structure is insufficiently "
                        "constrained, or every active material point has fully yielded."
                    ) from error

                u_trial[free] += delta_u_free

                if settings.convergence == "displacement":
                    displacement_ratio = displacement_correction_ratio(
                        delta_u_free, u_trial[free]
                    )
                    if displacement_ratio < settings.tolerance:
                        converged = True
                        break

            if free.size == 0:
                f_int_global, _, trial_states = self._assemble(
                    dof_map, elements, u_trial, committed_states
                )

            reactions = f_int_global - external_force

            step_result = LoadStepResult(
                load_factor=load_factor,
                displacement=u_trial.copy(),
                reactions=reactions,
                residual_norm=residual_ratio,
                iterations=iterations_used,
                converged=converged,
                element_states=dict(trial_states),
            )
            step_results.append(step_result)

            if not converged:
                raise NonlinearConvergenceError(
                    f"Newton-Raphson failed to converge at load step {step} "
                    f"(load factor {load_factor:.6g}) within {settings.max_iterations} "
                    f"iterations (final {settings.convergence} ratio: {residual_ratio:.3e}, "
                    f"tolerance: {settings.tolerance:.3e}).",
                    step_results=tuple(step_results),
                )

            u_committed = u_trial
            committed_states = trial_states

        return NonlinearAnalysisResult(dof_map=dof_map, step_results=tuple(step_results))
