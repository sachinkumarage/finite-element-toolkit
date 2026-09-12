"""Steady-state and transient heat-conduction solvers (Version 20-21).

**Steady-state.** Solves ``K_T @ T = F_T`` directly -- exactly the same
linear-algebra problem (a stiffness matrix, a load vector, Dirichlet
boundary conditions eliminated before solving) as every static mechanical
analysis in this toolkit, so this reuses
:class:`~femtoolkit.analysis.system.LinearSystem` and
:func:`~femtoolkit.analysis.system.solve` **directly, unmodified** --
``K_T``/``F_T`` are just a differently-named stiffness matrix and load
vector to that machinery, which has no notion of what physical quantity
its ``[K]{x}={F}`` system represents.

**Transient (Backward Euler).** Solves ``C_T @ dT/dt + K_T @ T = F_T``.
Approximating ``dT/dt`` at time ``t_(n+1)`` by the backward
(implicit) difference ``(T_(n+1) - T_n) / dt`` and evaluating every other
term at ``t_(n+1)`` gives:

.. code-block:: text

    C_T @ (T_(n+1) - T_n)/dt + K_T @ T_(n+1) = F_(n+1)
    (C_T/dt + K_T) @ T_(n+1) = F_(n+1) + (C_T/dt) @ T_n

Backward Euler is **unconditionally stable** (no matter how large ``dt``
is, the solution cannot blow up -- unlike forward/explicit Euler, which
requires a small enough time step to remain stable): exactly the
"robust, simple" first method this version's brief asks for. Each time
step is, again, just a linear solve with an *effective* stiffness
(``C_T/dt + K_T``) and an *effective* load
(``F_(n+1) + (C_T/dt)@T_n``) -- so this reuses the exact same
:class:`~femtoolkit.analysis.system.LinearSystem`/:func:`~femtoolkit.analysis.system.solve`
machinery, once per step -- **when every boundary condition is linear**
(see below).

**Convection (Version 21), the linear case.** A convective boundary with
a *constant* coefficient ``h`` is exactly linear in the unknown
temperature (see
:mod:`femtoolkit.thermal.thermal_boundary_conditions`'s module
docstring for the weak-form derivation): its conductance matrix
``K_conv`` and load contribution ``F_conv`` are fixed, known quantities,
folded directly into ``K_T``/``F_T`` alongside conduction and heat
generation/flux. This is by far the common case (most convection
problems use one lumped, constant ``h``), and it costs nothing extra --
the direct one-shot linear solve above still applies unchanged.

**Nonlinear thermal boundary conditions (Version 21).** Temperature-
dependent convection (``h = h(T)``) and radiation (``q ~ T^4``) make the
thermal system genuinely nonlinear in the unknown temperature. When
either is present, this module switches to **Newton-Raphson**
iteration, mirroring
:class:`~femtoolkit.analysis.nonlinear_analysis.NonlinearAnalysis`'s
exact pattern (``R(T) = F_ext(T) - F_int(T)``, linearize, solve
``K_t @ dT = R`` for the free DOFs, repeat) -- see :func:`_solve_nonlinear_system`.
Every *linear* boundary condition (conduction, heat generation, prescribed
flux) is assembled once, outside the iteration; the nonlinear surface
terms (every convection and radiation surface, evaluated at a single
representative -- mean-of-face-nodes -- temperature, per
:func:`femtoolkit.thermal.thermal_surfaces.surface_mean_temperature`) are
recomputed, with their exact tangent, at every iteration.

**Time-dependent ambient temperature.** A convection boundary's ambient
temperature can vary with time (see
:class:`~femtoolkit.thermal.thermal_boundary_conditions.ConvectionBoundaryCondition`).
:class:`TransientThermalAnalysis` resolves it at each time step's current
time before assembling that step's system, exactly like
:class:`TimeDependentThermalLoad` already resolves a time-varying nodal
load. :class:`SteadyStateThermalAnalysis` has no time axis, so it
resolves any time-dependent ambient condition at ``t = 0``.

**Sequential thermomechanical workflow (Version 19 connection).** This
module deliberately does *not* implement fully coupled nonlinear
thermoplasticity (explicitly out of scope) -- instead, a clean two-stage
pipeline:

.. code-block:: text

    Thermal analysis (this module)
        -> SteadyStateThermalResult / TransientThermalResult
        -> .to_temperature_field() -> TemperatureField (Version 19)
        -> thermoelastic_materials_for_mesh(...) -> per-element ThermoelasticMaterialAtTemperature
        -> NonlinearAnalysis (mechanical, Version 13-19, completely unmodified)

Version 19's :class:`~femtoolkit.analysis.temperature_field.TemperatureField`
was *designed* for exactly this connection (its own module docstring
says so explicitly); this version's results simply produce one, letting
the entire Version 19 thermomechanical consumer side work unmodified.
Version 21 does not change this connection at all -- it only makes the
*thermal* side able to reach a richer variety of realistic temperature
fields (via convection/radiation equilibrium) before handing one off.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from femtoolkit.analysis.assembly import (
    ElementStiffnessContribution,
    assemble_global_mass,
    assemble_global_stiffness,
)
from femtoolkit.analysis.boundary_conditions import BoundaryCondition
from femtoolkit.analysis.convergence import residual_norm_ratio
from femtoolkit.analysis.dof import DOFMap, TranslationDOF
from femtoolkit.analysis.dynamic_loads import TimeDependentLoad
from femtoolkit.analysis.system import LinearSystem, solve
from femtoolkit.exceptions import NonlinearConvergenceError, SingularSystemError, ValidationError
from femtoolkit.mesh.mesh import Mesh
from femtoolkit.thermal.thermal_boundary_conditions import (
    STEFAN_BOLTZMANN_CONSTANT,
    ConvectionBoundaryCondition,
    PrescribedHeatFlux,
    PrescribedTemperature,
    RadiationBoundaryCondition,
)
from femtoolkit.thermal.thermal_elements import capacity_contribution, conductivity_contribution
from femtoolkit.thermal.thermal_loads import ThermalLoad
from femtoolkit.thermal.thermal_material import ThermalMaterial
from femtoolkit.thermal.thermal_result import SteadyStateThermalResult, TransientThermalResult
from femtoolkit.thermal.thermal_surfaces import (
    surface_conductance_contribution,
    surface_node_ids,
    surface_uniform_load,
)

_TEMPERATURE_DOF = TranslationDOF.X
"""The single scalar DOF slot used for temperature -- see
:data:`femtoolkit.thermal.thermal_elements._TEMPERATURE_DOF` for why."""

_DEFAULT_NONLINEAR_MAX_ITERATIONS = 30
_DEFAULT_NONLINEAR_TOLERANCE = 1e-8
_FINITE_DIFFERENCE_STEP = 1e-3


def build_thermal_force_vector(dof_map: DOFMap, thermal_loads: Sequence[ThermalLoad]) -> np.ndarray:
    """Assemble a global thermal load vector from a list of thermal loads.

    The thermal analogue of
    :func:`~femtoolkit.analysis.system.build_force_vector`.

    Args:
        dof_map: DOF map defining the global DOF numbering.
        thermal_loads: Thermal loads to place into the vector. Loads on
            the same node are summed.

    Returns:
        The global thermal load vector ``{F_T}``, of shape
        ``(dof_map.total_dofs,)``.
    """
    forces = np.zeros(dof_map.total_dofs)
    for load in thermal_loads:
        global_index = dof_map.global_index(load.node_id, _TEMPERATURE_DOF)
        forces[global_index] += load.value
    return forces


@dataclass(frozen=True)
class TimeDependentThermalLoad:
    """A time-dependent heat-flow load applied to one node.

    The thermal analogue of
    :class:`~femtoolkit.analysis.dynamic_loads.TimeDependentNodalLoad`,
    reusing the exact same
    :class:`~femtoolkit.analysis.dynamic_loads.TimeDependentLoad`
    protocol and its ``ConstantLoad``/``StepLoad``/``SinusoidalLoad``
    implementations directly -- a time-history shape has no notion of
    "newtons" vs. "watts" baked into it, so nothing here needs
    reinventing.

    Attributes:
        node_id: The node this load applies to.
        load: The time-dependent load history, e.g. a
            :class:`~femtoolkit.analysis.dynamic_loads.StepLoad`.

    Example:
        >>> from femtoolkit.analysis.dynamic_loads import StepLoad
        >>> heater = TimeDependentThermalLoad(node_id=3, load=StepLoad(500.0, step_time=10.0))
        >>> heater.thermal_load_at(t=20.0)
        ThermalLoad(node_id=3, value=500.0)
    """

    node_id: int
    load: TimeDependentLoad

    def thermal_load_at(self, t: float) -> ThermalLoad:
        """Return the equivalent :class:`~femtoolkit.thermal.thermal_loads.ThermalLoad` at ``t``."""
        return ThermalLoad(self.node_id, self.load.value_at(t))


def _require_materials_for_every_element(mesh: Mesh, materials: dict[int, ThermalMaterial]) -> None:
    missing = [element.id for element in mesh.elements if element.id not in materials]
    if missing:
        raise ValidationError(
            f"No ThermalMaterial supplied for element(s) {missing}; every element in the "
            "mesh must have a corresponding entry in `materials`."
        )


def _validate_nonlinear_settings(max_iterations: int, tolerance: float) -> None:
    if (
        not isinstance(max_iterations, int)
        or isinstance(max_iterations, bool)
        or max_iterations <= 0
    ):
        raise ValidationError(
            f"nonlinear_max_iterations must be a positive integer, got {max_iterations!r}."
        )
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValidationError(f"nonlinear_tolerance must be positive, got {tolerance}.")


def _requires_nonlinear_solve(
    convections: Sequence[ConvectionBoundaryCondition],
    radiations: Sequence[RadiationBoundaryCondition],
) -> bool:
    """Whether any boundary condition makes the thermal problem nonlinear.

    True if any radiation boundary condition is present (radiation is
    always nonlinear, ``q ~ T^4``), or any convection boundary condition
    has a temperature-dependent coefficient. A convective boundary with
    a *constant* coefficient stays exactly linear and does not trigger
    this.
    """
    return bool(radiations) or any(bc.is_temperature_dependent for bc in convections)


def _linear_convection_contributions(
    mesh: Mesh, convections: Sequence[ConvectionBoundaryCondition], time: float
) -> tuple[list[ElementStiffnessContribution], list[ThermalLoad]]:
    """Fold every (necessarily constant-``h``) convection surface directly into K_T/F_T.

    Only called when :func:`_requires_nonlinear_solve` is ``False`` --
    every convection boundary condition present is guaranteed to have a
    constant coefficient, so evaluating it at any temperature gives the
    same value.
    """
    stiffness_contributions = []
    thermal_loads: list[ThermalLoad] = []
    for bc in convections:
        coefficient = bc.convection_coefficient_at(temperature=0.0)
        ambient = bc.ambient_temperature_at(time)
        stiffness_contributions.append(
            surface_conductance_contribution(mesh, bc.surface, coefficient)
        )
        thermal_loads.extend(surface_uniform_load(mesh, bc.surface, coefficient * ambient))
    return stiffness_contributions, thermal_loads


def _nonlinear_surface_terms(
    mesh: Mesh,
    convections: Sequence[ConvectionBoundaryCondition],
    radiations: Sequence[RadiationBoundaryCondition],
    dof_map: DOFMap,
    temperatures: np.ndarray,
    time: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the nonlinear residual contribution and tangent from every surface B.C.

    Every convection and radiation surface is evaluated at a single
    representative temperature (the mean of its own face nodes' current
    trial temperatures) -- see
    :func:`~femtoolkit.thermal.thermal_surfaces.surface_mean_temperature`.
    The *spatial* distribution over the face still uses full shape-
    function interpolation (the same ``integral(N^T N)``/``integral(N^T)``
    matrices used everywhere else in this toolkit); only the nonlinear
    *coefficient* (``h(T)`` or the ``T^4`` radiative term) is evaluated
    at that single representative value, exactly like Version 19/20's
    "one fixed evaluation temperature" simplification.

    Args:
        mesh: The mesh being solved.
        convections: Every convection boundary condition.
        radiations: Every radiation boundary condition.
        dof_map: The global DOF map.
        temperatures: The current trial nodal temperature vector.
        time: The current time (seconds) -- used to resolve a
            time-dependent ambient temperature.

    Returns:
        ``(residual, tangent)``: a length-``total_dofs`` residual
        contribution vector and a ``(total_dofs, total_dofs)`` tangent
        contribution matrix (``-d(residual)/dT``, ready to be *added*
        to a linear tangent stiffness).
    """
    residual_loads: list[ThermalLoad] = []
    tangent_contributions: list[ElementStiffnessContribution] = []

    for bc in convections:
        node_ids = surface_node_ids(mesh, bc.surface)
        indices = [dof_map.global_index(node_id, _TEMPERATURE_DOF) for node_id in node_ids]
        face_temperatures = temperatures[indices]
        mean_temperature = float(np.mean(face_temperatures))
        coefficient = bc.convection_coefficient_at(mean_temperature)
        ambient = bc.ambient_temperature_at(time)

        n = len(node_ids)
        shape_matrix = surface_conductance_contribution(mesh, bc.surface, 1.0).stiffness
        shape_vector = np.array(
            [load.value for load in surface_uniform_load(mesh, bc.surface, 1.0)]
        )

        residual_face = coefficient * (ambient * shape_vector - shape_matrix @ face_temperatures)
        residual_loads.extend(
            ThermalLoad(node_id, float(value))
            for node_id, value in zip(node_ids, residual_face, strict=True)
        )

        if bc.is_temperature_dependent:
            step = max(_FINITE_DIFFERENCE_STEP, abs(mean_temperature) * 1e-6)
            derivative = (
                bc.convection_coefficient_at(mean_temperature + step)
                - bc.convection_coefficient_at(mean_temperature - step)
            ) / (2.0 * step)
        else:
            derivative = 0.0

        base_vector = shape_matrix @ face_temperatures - ambient * shape_vector
        tangent_block = coefficient * shape_matrix + (derivative / n) * np.outer(
            base_vector, np.ones(n)
        )
        dof_keys = tuple((node_id, _TEMPERATURE_DOF) for node_id in node_ids)
        tangent_contributions.append(ElementStiffnessContribution(dof_keys, tangent_block))

    for bc in radiations:
        node_ids = surface_node_ids(mesh, bc.surface)
        indices = [dof_map.global_index(node_id, _TEMPERATURE_DOF) for node_id in node_ids]
        face_temperatures = temperatures[indices]
        mean_temperature = float(np.mean(face_temperatures))
        coefficient = bc.emissivity * STEFAN_BOLTZMANN_CONSTANT

        n = len(node_ids)
        shape_vector = np.array(
            [load.value for load in surface_uniform_load(mesh, bc.surface, 1.0)]
        )

        residual_face = coefficient * (
            bc.surrounding_temperature**4 - mean_temperature**4
        ) * shape_vector
        residual_loads.extend(
            ThermalLoad(node_id, float(value))
            for node_id, value in zip(node_ids, residual_face, strict=True)
        )

        tangent_scale = coefficient * 4.0 * mean_temperature**3 / n
        tangent_block = tangent_scale * np.outer(shape_vector, np.ones(n))
        dof_keys = tuple((node_id, _TEMPERATURE_DOF) for node_id in node_ids)
        tangent_contributions.append(ElementStiffnessContribution(dof_keys, tangent_block))

    residual = build_thermal_force_vector(dof_map, residual_loads)
    tangent = (
        assemble_global_stiffness(dof_map, tangent_contributions)
        if tangent_contributions
        else np.zeros((dof_map.total_dofs, dof_map.total_dofs))
    )
    return residual, tangent


def _solve_nonlinear_system(
    dof_map: DOFMap,
    mesh: Mesh,
    k_lin: np.ndarray,
    f_lin: np.ndarray,
    boundary_conditions: Sequence[PrescribedTemperature],
    convections: Sequence[ConvectionBoundaryCondition],
    radiations: Sequence[RadiationBoundaryCondition],
    time: float,
    initial_temperatures: np.ndarray,
    max_iterations: int,
    tolerance: float,
) -> np.ndarray:
    """Solve ``k_lin @ T + [nonlinear surface terms](T) = f_lin`` via Newton-Raphson.

    Mirrors :class:`~femtoolkit.analysis.nonlinear_analysis.NonlinearAnalysis`'s
    exact free/constrained-DOF partition and iteration structure (see
    that module's docstring), applied to the scalar thermal DOF instead
    of a vector displacement DOF.

    Args:
        dof_map: The global DOF map.
        mesh: The mesh being solved.
        k_lin: The fixed (conduction, or Backward-Euler effective)
            linear stiffness matrix.
        f_lin: The fixed linear forcing vector (heat generation +
            prescribed flux, or its transient effective-load
            counterpart).
        boundary_conditions: Prescribed-temperature boundary conditions.
        convections: Every convection boundary condition (linear and
            nonlinear alike -- see the module docstring).
        radiations: Every radiation boundary condition.
        time: The current time (seconds), for resolving a time-dependent
            ambient temperature.
        initial_temperatures: The Newton-Raphson starting temperature
            vector (constrained entries are overwritten with their
            prescribed value).
        max_iterations: Maximum Newton-Raphson iterations.
        tolerance: Residual-ratio convergence tolerance.

    Returns:
        The converged nodal temperature vector.

    Raises:
        SingularSystemError: If the reduced tangent matrix is singular.
        NonlinearConvergenceError: If the iteration does not converge
            within ``max_iterations``.
    """
    total_dofs = dof_map.total_dofs
    constrained_targets: dict[int, float] = {}
    for bc in boundary_conditions:
        index = dof_map.global_index(bc.node_id, _TEMPERATURE_DOF)
        if index in constrained_targets:
            raise ValidationError(
                f"Multiple boundary conditions target the same DOF (node_id={bc.node_id})."
            )
        constrained_targets[index] = bc.value

    free = np.array([i for i in range(total_dofs) if i not in constrained_targets], dtype=int)

    temperatures = initial_temperatures.copy()
    for index, value in constrained_targets.items():
        temperatures[index] = value

    if free.size == 0:
        return temperatures

    residual_ratio = 0.0
    for iteration in range(1, max_iterations + 1):
        residual_nl, tangent_nl = _nonlinear_surface_terms(
            mesh, convections, radiations, dof_map, temperatures, time
        )
        residual = f_lin + residual_nl - k_lin @ temperatures
        residual_ratio = residual_norm_ratio(residual[free], f_lin[free])
        if residual_ratio < tolerance:
            return temperatures

        tangent = k_lin + tangent_nl
        tangent_free_free = tangent[np.ix_(free, free)]
        try:
            delta_free = np.linalg.solve(tangent_free_free, residual[free])
        except np.linalg.LinAlgError as error:
            raise SingularSystemError(
                f"The thermal tangent matrix is singular at Newton-Raphson iteration "
                f"{iteration}: the model is insufficiently constrained."
            ) from error
        temperatures[free] += delta_free

    raise NonlinearConvergenceError(
        f"Thermal Newton-Raphson failed to converge within {max_iterations} iterations "
        f"(final residual ratio {residual_ratio})."
    )


def steady_state_energy_balance(
    analysis: SteadyStateThermalAnalysis, result: SteadyStateThermalResult
) -> tuple[float, float]:
    """Return ``(q_supplied, q_removed)`` for a solved steady-state analysis.

    Computes the same net heat flow through two independent paths, which
    must agree at a correctly converged steady state: ``q_supplied`` sums
    every explicitly applied source (heat generation, prescribed flux,
    and the net convective/radiative exchange, evaluated at the actual
    solved temperature); ``q_removed`` sums the implied heat flow at the
    prescribed-temperature boundaries needed to sustain them (the
    thermal analogue of a mechanical support reaction). At an exact
    steady state, ``Q_in ~= Q_out`` (this version's brief), so
    ``q_supplied`` should equal ``q_removed`` to within numerical
    precision -- a mismatch signals a modeling error (e.g. a boundary
    condition applied to the wrong node), since the underlying finite
    element formulation conserves energy exactly at its own converged
    solution.

    Args:
        analysis: The (already solved) steady-state analysis.
        result: The result returned by ``analysis.solve()``.

    Returns:
        ``(q_supplied, q_removed)``, both in watts, computed
        independently and expected to agree.
    """
    dof_map = result.dof_map
    conductivity_contributions = [
        conductivity_contribution(
            element, analysis.materials[element.id], analysis.evaluation_temperature
        )
        for element in analysis.mesh.elements
    ]
    k_conduction = assemble_global_stiffness(dof_map, conductivity_contributions)

    flux_loads = [ThermalLoad(bc.node_id, bc.value) for bc in analysis._heat_fluxes]
    f_lin = build_thermal_force_vector(dof_map, [*analysis._thermal_loads, *flux_loads])

    nonlinear_residual, _ = _nonlinear_surface_terms(
        analysis.mesh,
        analysis._convections,
        analysis._radiations,
        dof_map,
        result.temperatures,
        time=0.0,
    )

    full_residual = f_lin + nonlinear_residual - k_conduction @ result.temperatures
    constrained_indices = [
        dof_map.global_index(bc.node_id, _TEMPERATURE_DOF) for bc in analysis._boundary_conditions
    ]

    q_supplied = float(np.sum(f_lin) + np.sum(nonlinear_residual))
    q_removed = float(np.sum(full_residual[constrained_indices]))
    return q_supplied, q_removed


@dataclass
class SteadyStateThermalAnalysis:
    """Solves the steady-state heat-conduction problem ``K_T @ T = F_T``.

    Attributes:
        mesh: The mesh to analyze (nodes and thermally-capable elements).
        materials: Maps each element's ``id`` to the
            :class:`~femtoolkit.thermal.thermal_material.ThermalMaterial`
            it is solved with. Every element in ``mesh`` must have an entry.
        evaluation_temperature: The (fixed) temperature used to evaluate
            temperature-dependent conductivity, in kelvin. Irrelevant if
            every material's conductivity is a constant.
        nonlinear_max_iterations: Maximum Newton-Raphson iterations, used
            only when a temperature-dependent convection coefficient or
            a radiation boundary condition is present (see the module
            docstring).
        nonlinear_tolerance: Newton-Raphson residual-ratio convergence
            tolerance, used only in the nonlinear case.

    Raises:
        ValidationError: If ``materials`` is missing an entry for any
            element in ``mesh``, or ``nonlinear_max_iterations``/
            ``nonlinear_tolerance`` is invalid.

    Example:
        >>> analysis = SteadyStateThermalAnalysis(mesh, {1: steel_thermal})
        >>> analysis.add_boundary_condition(PrescribedTemperature(node_id=1, value=373.15))
        >>> analysis.add_boundary_condition(PrescribedTemperature(node_id=2, value=293.15))
        >>> result = analysis.solve()
    """

    mesh: Mesh
    materials: dict[int, ThermalMaterial]
    evaluation_temperature: float = 293.15
    nonlinear_max_iterations: int = _DEFAULT_NONLINEAR_MAX_ITERATIONS
    nonlinear_tolerance: float = _DEFAULT_NONLINEAR_TOLERANCE
    _boundary_conditions: list[PrescribedTemperature] = field(default_factory=list, init=False)
    _heat_fluxes: list[PrescribedHeatFlux] = field(default_factory=list, init=False)
    _thermal_loads: list[ThermalLoad] = field(default_factory=list, init=False)
    _convections: list[ConvectionBoundaryCondition] = field(default_factory=list, init=False)
    _radiations: list[RadiationBoundaryCondition] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        """Validate that every mesh element has an assigned thermal material.

        Raises:
            ValidationError: If ``materials`` is missing an entry for
                any element in ``mesh``, or ``nonlinear_max_iterations``/
                ``nonlinear_tolerance`` is invalid.
        """
        _require_materials_for_every_element(self.mesh, self.materials)
        _validate_nonlinear_settings(self.nonlinear_max_iterations, self.nonlinear_tolerance)

    def add_boundary_condition(self, boundary_condition: PrescribedTemperature) -> None:
        """Add a prescribed-temperature boundary condition."""
        self._boundary_conditions.append(boundary_condition)

    def add_heat_flux(self, heat_flux: PrescribedHeatFlux) -> None:
        """Add a prescribed heat-flux (Neumann) boundary condition."""
        self._heat_fluxes.append(heat_flux)

    def add_thermal_load(self, thermal_load: ThermalLoad) -> None:
        """Add a single nodal thermal load (e.g. one contribution from heat generation)."""
        self._thermal_loads.append(thermal_load)

    def add_thermal_loads(self, thermal_loads: Sequence[ThermalLoad]) -> None:
        """Add several nodal thermal loads at once (e.g. from
        :func:`~femtoolkit.thermal.thermal_loads.heat_generation_to_thermal_loads`)."""
        self._thermal_loads.extend(thermal_loads)

    def add_convection(self, convection: ConvectionBoundaryCondition) -> None:
        """Add a convective (Robin) boundary condition."""
        self._convections.append(convection)

    def add_radiation(self, radiation: RadiationBoundaryCondition) -> None:
        """Add a radiative boundary condition."""
        self._radiations.append(radiation)

    def solve(self) -> SteadyStateThermalResult:
        """Assemble and solve the steady-state thermal system.

        1. Assembles the global conductivity matrix ``K_T`` (conduction,
           plus any constant-coefficient convection).
        2. Assembles the global thermal load vector ``F_T`` (heat
           generation, prescribed heat flux, and constant-coefficient
           convection contributions).
        3. Applies prescribed-temperature boundary conditions.
        4. If a temperature-dependent convection coefficient or a
           radiation boundary condition is present, solves the resulting
           nonlinear system via Newton-Raphson (see the module
           docstring); otherwise solves the linear system directly.

        Returns:
            The solved :class:`~femtoolkit.thermal.thermal_result.SteadyStateThermalResult`.

        Raises:
            ValidationError: If no prescribed-temperature boundary
                condition was added.
            SingularSystemError: If the (reduced) linear/tangent system
                is singular.
            NonlinearConvergenceError: If the nonlinear system fails to
                converge within :attr:`nonlinear_max_iterations`.
        """
        if not self._boundary_conditions:
            raise ValidationError(
                "SteadyStateThermalAnalysis requires at least one PrescribedTemperature "
                "boundary condition."
            )

        dof_map = DOFMap(node_ids=[node.id for node in self.mesh.nodes], dofs_per_node=1)

        conductivity_contributions = [
            conductivity_contribution(
                element, self.materials[element.id], self.evaluation_temperature
            )
            for element in self.mesh.elements
        ]

        flux_loads = [ThermalLoad(bc.node_id, bc.value) for bc in self._heat_fluxes]

        boundary_conditions = list(self._boundary_conditions)

        if _requires_nonlinear_solve(self._convections, self._radiations):
            k_t = assemble_global_stiffness(dof_map, conductivity_contributions)
            f_t = build_thermal_force_vector(dof_map, [*self._thermal_loads, *flux_loads])
            initial_temperatures = np.full(dof_map.total_dofs, self.evaluation_temperature)
            temperatures = _solve_nonlinear_system(
                dof_map,
                self.mesh,
                k_t,
                f_t,
                boundary_conditions,
                self._convections,
                self._radiations,
                time=0.0,
                initial_temperatures=initial_temperatures,
                max_iterations=self.nonlinear_max_iterations,
                tolerance=self.nonlinear_tolerance,
            )
        else:
            conv_stiffness, conv_loads = _linear_convection_contributions(
                self.mesh, self._convections, time=0.0
            )
            k_t = assemble_global_stiffness(dof_map, [*conductivity_contributions, *conv_stiffness])
            f_t = build_thermal_force_vector(
                dof_map, [*self._thermal_loads, *flux_loads, *conv_loads]
            )
            system = LinearSystem(
                dof_map=dof_map,
                stiffness=k_t,
                forces=f_t,
                boundary_conditions=[
                    BoundaryCondition(bc.node_id, _TEMPERATURE_DOF, bc.value)
                    for bc in boundary_conditions
                ],
            )
            temperatures = solve(system)

        return SteadyStateThermalResult(
            dof_map=dof_map,
            temperatures=temperatures,
            mesh=self.mesh,
            materials=self.materials,
            evaluation_temperature=self.evaluation_temperature,
        )


@dataclass
class TransientThermalAnalysis:
    """Solves the transient problem ``C_T @ dT/dt + K_T @ T = F_T`` via Backward Euler.

    Attributes:
        mesh: The mesh to analyze.
        materials: Maps each element's ``id`` to its
            :class:`~femtoolkit.thermal.thermal_material.ThermalMaterial`.
            Every element in ``mesh`` must have an entry.
        time_step: The (fixed) time step ``dt``, in seconds. Must be positive.
        num_steps: The number of time steps to take. Must be positive.
        initial_temperature: The temperature every node starts at, in
            kelvin (a single value applied uniformly), or a mapping from
            node ID to its own initial temperature.
        evaluation_temperature: The (fixed) temperature used to evaluate
            temperature-dependent conductivity/specific heat, in kelvin.
        nonlinear_max_iterations: Maximum Newton-Raphson iterations per
            time step, used only when a temperature-dependent convection
            coefficient or a radiation boundary condition is present.
        nonlinear_tolerance: Newton-Raphson residual-ratio convergence
            tolerance, used only in the nonlinear case.

    Raises:
        ValidationError: If ``materials`` is missing an entry for any
            element in ``mesh``, ``time_step`` is not positive and
            finite, ``num_steps`` is not a positive integer, or
            ``nonlinear_max_iterations``/``nonlinear_tolerance`` is invalid.

    Example:
        >>> analysis = TransientThermalAnalysis(
        ...     mesh, {1: steel_thermal}, time_step=1.0, num_steps=50, initial_temperature=293.15
        ... )
        >>> analysis.add_boundary_condition(PrescribedTemperature(node_id=1, value=373.15))
        >>> result = analysis.solve()
    """

    mesh: Mesh
    materials: dict[int, ThermalMaterial]
    time_step: float
    num_steps: int
    initial_temperature: float | dict[int, float] = 293.15
    evaluation_temperature: float = 293.15
    nonlinear_max_iterations: int = _DEFAULT_NONLINEAR_MAX_ITERATIONS
    nonlinear_tolerance: float = _DEFAULT_NONLINEAR_TOLERANCE
    _boundary_conditions: list[PrescribedTemperature] = field(default_factory=list, init=False)
    _heat_fluxes: list[PrescribedHeatFlux] = field(default_factory=list, init=False)
    _thermal_loads: list[ThermalLoad] = field(default_factory=list, init=False)
    _time_dependent_loads: list[TimeDependentThermalLoad] = field(default_factory=list, init=False)
    _convections: list[ConvectionBoundaryCondition] = field(default_factory=list, init=False)
    _radiations: list[RadiationBoundaryCondition] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        """Validate the time-stepping parameters and material assignments.

        Raises:
            ValidationError: If ``materials`` is missing an entry for
                any element in ``mesh``, ``time_step`` is not positive
                and finite, ``num_steps`` is not a positive integer, or
                ``nonlinear_max_iterations``/``nonlinear_tolerance`` is
                invalid.
        """
        _require_materials_for_every_element(self.mesh, self.materials)
        _validate_nonlinear_settings(self.nonlinear_max_iterations, self.nonlinear_tolerance)
        if not np.isfinite(self.time_step) or self.time_step <= 0:
            raise ValidationError(
                f"TransientThermalAnalysis time_step must be positive, got {self.time_step}."
            )
        if not isinstance(self.num_steps, int) or self.num_steps <= 0:
            raise ValidationError(
                f"TransientThermalAnalysis num_steps must be a positive integer, got "
                f"{self.num_steps!r}."
            )

    def add_boundary_condition(self, boundary_condition: PrescribedTemperature) -> None:
        """Add a prescribed-temperature boundary condition (held fixed across every time step)."""
        self._boundary_conditions.append(boundary_condition)

    def add_heat_flux(self, heat_flux: PrescribedHeatFlux) -> None:
        """Add a prescribed heat-flux boundary condition (held fixed across every time step)."""
        self._heat_fluxes.append(heat_flux)

    def add_thermal_load(self, thermal_load: ThermalLoad) -> None:
        """Add a single nodal thermal load, applied unchanged at every time step."""
        self._thermal_loads.append(thermal_load)

    def add_thermal_loads(self, thermal_loads: Sequence[ThermalLoad]) -> None:
        """Add several nodal thermal loads, each applied unchanged at every time step."""
        self._thermal_loads.extend(thermal_loads)

    def add_time_dependent_load(self, time_dependent_load: TimeDependentThermalLoad) -> None:
        """Add a time-varying thermal load (see :class:`TimeDependentThermalLoad`)."""
        self._time_dependent_loads.append(time_dependent_load)

    def add_convection(self, convection: ConvectionBoundaryCondition) -> None:
        """Add a convective (Robin) boundary condition.

        Its ambient temperature is re-resolved at every time step's
        current time (see
        :meth:`~femtoolkit.thermal.thermal_boundary_conditions.ConvectionBoundaryCondition.ambient_temperature_at`).
        """
        self._convections.append(convection)

    def add_radiation(self, radiation: RadiationBoundaryCondition) -> None:
        """Add a radiative boundary condition."""
        self._radiations.append(radiation)

    def _initial_temperature_vector(self, dof_map: DOFMap) -> np.ndarray:
        if isinstance(self.initial_temperature, dict):
            temperatures = np.zeros(dof_map.total_dofs)
            for node in self.mesh.nodes:
                if node.id not in self.initial_temperature:
                    raise ValidationError(
                        f"No initial temperature supplied for node {node.id}."
                    )
                temperatures[dof_map.global_index(node.id, _TEMPERATURE_DOF)] = (
                    self.initial_temperature[node.id]
                )
            return temperatures
        return np.full(dof_map.total_dofs, float(self.initial_temperature))

    def solve(self) -> TransientThermalResult:
        """Time-step the transient thermal system via Backward Euler.

        Assembles ``K_T`` (conduction) and ``C_T`` once (material
        properties are evaluated at :attr:`evaluation_temperature`
        throughout, kept fixed across time steps -- see the module
        docstring). At each step, if a temperature-dependent convection
        coefficient or a radiation boundary condition is present, solves
        that step's nonlinear system via Newton-Raphson (starting from
        the previous step's converged temperature); otherwise folds any
        constant-coefficient convection directly into that step's
        effective stiffness/load and solves directly, reusing
        :class:`~femtoolkit.analysis.system.LinearSystem`/:func:`~femtoolkit.analysis.system.solve`.

        Returns:
            The solved :class:`~femtoolkit.thermal.thermal_result.TransientThermalResult`,
            including the initial condition as its first stored step.

        Raises:
            ValidationError: If no prescribed-temperature boundary
                condition was added.
            SingularSystemError: If a step's (reduced) linear/tangent
                system is singular.
            NonlinearConvergenceError: If a step's nonlinear system
                fails to converge within :attr:`nonlinear_max_iterations`.
        """
        if not self._boundary_conditions:
            raise ValidationError(
                "TransientThermalAnalysis requires at least one PrescribedTemperature "
                "boundary condition."
            )

        dof_map = DOFMap(node_ids=[node.id for node in self.mesh.nodes], dofs_per_node=1)

        conductivity_contributions = [
            conductivity_contribution(
                element, self.materials[element.id], self.evaluation_temperature
            )
            for element in self.mesh.elements
        ]
        k_t = assemble_global_stiffness(dof_map, conductivity_contributions)

        capacity_contributions = [
            capacity_contribution(element, self.materials[element.id], self.evaluation_temperature)
            for element in self.mesh.elements
        ]
        c_t = assemble_global_mass(dof_map, capacity_contributions)

        flux_loads = [ThermalLoad(bc.node_id, bc.value) for bc in self._heat_fluxes]
        f_static = build_thermal_force_vector(dof_map, [*self._thermal_loads, *flux_loads])

        boundary_conditions = list(self._boundary_conditions)
        mechanical_boundary_conditions = [
            BoundaryCondition(bc.node_id, _TEMPERATURE_DOF, bc.value) for bc in boundary_conditions
        ]

        temperatures = self._initial_temperature_vector(dof_map)
        for boundary_condition in mechanical_boundary_conditions:
            index = dof_map.global_index(boundary_condition.node_id, boundary_condition.dof)
            temperatures[index] = boundary_condition.value

        effective_stiffness = c_t / self.time_step + k_t
        nonlinear = _requires_nonlinear_solve(self._convections, self._radiations)

        times = [0.0]
        temperature_history = [temperatures.copy()]

        for step in range(1, self.num_steps + 1):
            current_time = step * self.time_step
            dynamic_loads = [
                load.thermal_load_at(current_time) for load in self._time_dependent_loads
            ]
            f_dynamic = build_thermal_force_vector(dof_map, dynamic_loads)
            effective_forces = f_static + f_dynamic + (c_t @ temperatures) / self.time_step

            if nonlinear:
                temperatures = _solve_nonlinear_system(
                    dof_map,
                    self.mesh,
                    effective_stiffness,
                    effective_forces,
                    boundary_conditions,
                    self._convections,
                    self._radiations,
                    time=current_time,
                    initial_temperatures=temperatures,
                    max_iterations=self.nonlinear_max_iterations,
                    tolerance=self.nonlinear_tolerance,
                )
            else:
                conv_stiffness, conv_loads = _linear_convection_contributions(
                    self.mesh, self._convections, time=current_time
                )
                step_stiffness = effective_stiffness + (
                    assemble_global_stiffness(dof_map, conv_stiffness)
                    if conv_stiffness
                    else 0.0
                )
                step_forces = effective_forces + build_thermal_force_vector(dof_map, conv_loads)
                system = LinearSystem(
                    dof_map=dof_map,
                    stiffness=step_stiffness,
                    forces=step_forces,
                    boundary_conditions=mechanical_boundary_conditions,
                )
                temperatures = solve(system)

            times.append(current_time)
            temperature_history.append(temperatures.copy())

        return TransientThermalResult(
            dof_map=dof_map,
            times=np.array(times),
            temperature_history=temperature_history,
            mesh=self.mesh,
            materials=self.materials,
            evaluation_temperature=self.evaluation_temperature,
        )
