"""Steady-state and transient heat-conduction solvers (Version 20).

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
machinery, once per step.

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
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from femtoolkit.analysis.assembly import assemble_global_mass, assemble_global_stiffness
from femtoolkit.analysis.boundary_conditions import BoundaryCondition
from femtoolkit.analysis.dof import DOFMap, TranslationDOF
from femtoolkit.analysis.dynamic_loads import TimeDependentLoad
from femtoolkit.analysis.system import LinearSystem, solve
from femtoolkit.exceptions import ValidationError
from femtoolkit.mesh.mesh import Mesh
from femtoolkit.thermal.thermal_boundary_conditions import PrescribedHeatFlux, PrescribedTemperature
from femtoolkit.thermal.thermal_elements import capacity_contribution, conductivity_contribution
from femtoolkit.thermal.thermal_loads import ThermalLoad
from femtoolkit.thermal.thermal_material import ThermalMaterial
from femtoolkit.thermal.thermal_result import SteadyStateThermalResult, TransientThermalResult

_TEMPERATURE_DOF = TranslationDOF.X
"""The single scalar DOF slot used for temperature -- see
:data:`femtoolkit.thermal.thermal_elements._TEMPERATURE_DOF` for why."""


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

    Raises:
        ValidationError: If ``materials`` is missing an entry for any
            element in ``mesh``.

    Example:
        >>> analysis = SteadyStateThermalAnalysis(mesh, {1: steel_thermal})
        >>> analysis.add_boundary_condition(PrescribedTemperature(node_id=1, value=373.15))
        >>> analysis.add_boundary_condition(PrescribedTemperature(node_id=2, value=293.15))
        >>> result = analysis.solve()
    """

    mesh: Mesh
    materials: dict[int, ThermalMaterial]
    evaluation_temperature: float = 293.15
    _boundary_conditions: list[PrescribedTemperature] = field(default_factory=list, init=False)
    _heat_fluxes: list[PrescribedHeatFlux] = field(default_factory=list, init=False)
    _thermal_loads: list[ThermalLoad] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        """Validate that every mesh element has an assigned thermal material.

        Raises:
            ValidationError: If ``materials`` is missing an entry for
                any element in ``mesh``.
        """
        _require_materials_for_every_element(self.mesh, self.materials)

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

    def solve(self) -> SteadyStateThermalResult:
        """Assemble and solve the steady-state thermal system.

        1. Assembles the global conductivity matrix ``K_T``.
        2. Assembles the global thermal load vector ``F_T`` (heat
           generation + prescribed heat flux contributions).
        3. Applies prescribed-temperature boundary conditions.
        4. Solves for the nodal temperature vector.

        Returns:
            The solved :class:`~femtoolkit.thermal.thermal_result.SteadyStateThermalResult`.

        Raises:
            ValidationError: If no prescribed-temperature boundary
                condition was added (the system would be singular).
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
        k_t = assemble_global_stiffness(dof_map, conductivity_contributions)

        flux_loads = [ThermalLoad(bc.node_id, bc.value) for bc in self._heat_fluxes]
        f_t = build_thermal_force_vector(dof_map, [*self._thermal_loads, *flux_loads])

        boundary_conditions = [
            BoundaryCondition(bc.node_id, _TEMPERATURE_DOF, bc.value)
            for bc in self._boundary_conditions
        ]
        system = LinearSystem(
            dof_map=dof_map, stiffness=k_t, forces=f_t, boundary_conditions=boundary_conditions
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

    Raises:
        ValidationError: If ``materials`` is missing an entry for any
            element in ``mesh``, ``time_step`` is not positive and
            finite, or ``num_steps`` is not a positive integer.

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
    _boundary_conditions: list[PrescribedTemperature] = field(default_factory=list, init=False)
    _heat_fluxes: list[PrescribedHeatFlux] = field(default_factory=list, init=False)
    _thermal_loads: list[ThermalLoad] = field(default_factory=list, init=False)
    _time_dependent_loads: list[TimeDependentThermalLoad] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        """Validate the time-stepping parameters and material assignments.

        Raises:
            ValidationError: If ``materials`` is missing an entry for
                any element in ``mesh``, ``time_step`` is not positive
                and finite, or ``num_steps`` is not a positive integer.
        """
        _require_materials_for_every_element(self.mesh, self.materials)
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

        Assembles ``K_T`` and ``C_T`` once (material properties are
        evaluated at :attr:`evaluation_temperature` throughout, kept
        fixed across time steps -- see the module docstring), then
        solves ``(C_T/dt + K_T) @ T_(n+1) = F_(n+1) + (C_T/dt) @ T_n``
        once per step, reusing
        :class:`~femtoolkit.analysis.system.LinearSystem`/:func:`~femtoolkit.analysis.system.solve`.

        Returns:
            The solved :class:`~femtoolkit.thermal.thermal_result.TransientThermalResult`,
            including the initial condition as its first stored step.

        Raises:
            ValidationError: If no prescribed-temperature boundary
                condition was added.
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

        boundary_conditions = [
            BoundaryCondition(bc.node_id, _TEMPERATURE_DOF, bc.value)
            for bc in self._boundary_conditions
        ]

        temperatures = self._initial_temperature_vector(dof_map)
        for boundary_condition in boundary_conditions:
            index = dof_map.global_index(boundary_condition.node_id, boundary_condition.dof)
            temperatures[index] = boundary_condition.value

        effective_stiffness = c_t / self.time_step + k_t

        times = [0.0]
        temperature_history = [temperatures.copy()]

        for step in range(1, self.num_steps + 1):
            current_time = step * self.time_step
            dynamic_loads = [
                load.thermal_load_at(current_time) for load in self._time_dependent_loads
            ]
            f_dynamic = build_thermal_force_vector(dof_map, dynamic_loads)
            effective_forces = f_static + f_dynamic + (c_t @ temperatures) / self.time_step

            system = LinearSystem(
                dof_map=dof_map,
                stiffness=effective_stiffness,
                forces=effective_forces,
                boundary_conditions=boundary_conditions,
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
