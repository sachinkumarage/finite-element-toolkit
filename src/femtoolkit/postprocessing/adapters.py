"""Bridges from every existing result class into a
:class:`~femtoolkit.postprocessing.result_model.SimulationResult`.

This module performs **no calculation of its own** -- every value it
stores was already computed by a solver or an existing result class
(:class:`~femtoolkit.results.analysis_result.AnalysisResult`,
:class:`~femtoolkit.results.nonlinear_result.NonlinearAnalysisResult`,
:class:`~femtoolkit.results.dynamic_result.DynamicResult`,
:class:`~femtoolkit.thermal.thermal_result.SteadyStateThermalResult`,
:class:`~femtoolkit.thermal.thermal_result.TransientThermalResult`). Its
only job is *reshaping*: pulling values out through each class's own
public query methods and placing them into the uniform step/field shape
:mod:`femtoolkit.postprocessing.result_model` defines.

The one partial exception is transient thermal element fields
(temperature gradient, heat flux):
:class:`~femtoolkit.thermal.thermal_result.TransientThermalResult`
does not itself expose a per-step element accessor for them (only
:class:`~femtoolkit.thermal.thermal_result.SteadyStateThermalResult`
does), so :func:`from_thermal_transient` calls the same *public* element-
level functions those accessors call internally
(:func:`~femtoolkit.thermal.thermal_elements.element_temperature_gradient`/
:func:`~femtoolkit.thermal.thermal_elements.element_heat_flux`) directly,
once per stored time step -- reusing the exact same formulas, not
re-deriving them.

**Multi-Gauss-point elements.** A Q4/HEX8 element's stress/strain state
is generally different at each of its Gauss points (Version 13/15); this
module reports the state at **Gauss point 0** as that element's single
representative value, mirroring the same simplified single-value
reporting convention :class:`~femtoolkit.mesh.hex8_element.Hex8Element3D`/
:class:`~femtoolkit.mesh.quad_element.QuadElement2D` already use for
their own ``strain_from_dofs``/``stress_from_dofs`` (evaluated at the
element's natural-coordinate center) -- not a new simplification
invented here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from femtoolkit.exceptions import InvalidElementError
from femtoolkit.postprocessing.result_model import MeshTopology, ResultStep, SimulationResult
from femtoolkit.thermal.thermal_elements import (
    element_heat_flux,
    element_temperature_gradient,
    thermal_dof_keys,
)

if TYPE_CHECKING:
    from femtoolkit.mesh.mesh import Mesh
    from femtoolkit.results.analysis_result import AnalysisResult
    from femtoolkit.results.dynamic_result import DynamicResult
    from femtoolkit.results.nonlinear_result import NonlinearAnalysisResult
    from femtoolkit.thermal.thermal_result import SteadyStateThermalResult, TransientThermalResult

_GAUSS_POINT = 0
"""The representative Gauss point used for a multi-point element's stress/strain -- see
the module docstring."""


def from_thermal_steady_state(result: SteadyStateThermalResult) -> SimulationResult:
    """Build a :class:`~femtoolkit.postprocessing.result_model.SimulationResult` from a
    solved :class:`~femtoolkit.thermal.thermal_result.SteadyStateThermalResult`.

    Args:
        result: The solved steady-state thermal result.

    Returns:
        A single-step :class:`~femtoolkit.postprocessing.result_model.SimulationResult`.
    """
    topology = MeshTopology.from_mesh(result.mesh)
    nodal_fields = {
        "temperature": {node.id: result.node_temperature(node.id) for node in result.mesh.nodes}
    }
    element_fields = {
        "temperature_gradient": {
            element.id: result.element_temperature_gradient(element.id)
            for element in result.mesh.elements
        },
        "heat_flux": {
            element.id: result.element_heat_flux(element.id) for element in result.mesh.elements
        },
    }
    step = ResultStep(index=0, time=0.0, nodal_fields=nodal_fields, element_fields=element_fields)
    return SimulationResult(
        topology,
        steps=(step,),
        field_units={"temperature": "K", "temperature_gradient": "K/m", "heat_flux": "W/m^2"},
    )


def _element_nodal_temperatures(
    mesh: Mesh, element_id: int, temperatures: np.ndarray, dof_map
) -> tuple[np.ndarray, object]:
    """The transient-result analogue of ``SteadyStateThermalResult``'s internal helper."""
    element = mesh.get_element(element_id)
    indices = [dof_map.global_index(node_id, dof) for node_id, dof in thermal_dof_keys(element)]
    return temperatures[indices], element


def from_thermal_transient(result: TransientThermalResult) -> SimulationResult:
    """Build a :class:`~femtoolkit.postprocessing.result_model.SimulationResult` from a
    solved :class:`~femtoolkit.thermal.thermal_result.TransientThermalResult`.

    Args:
        result: The solved transient thermal result.

    Returns:
        A :class:`~femtoolkit.postprocessing.result_model.SimulationResult` with one
        step per stored time step, including the initial condition.
    """
    topology = MeshTopology.from_mesh(result.mesh)
    steps = []
    for step_index, time in enumerate(result.times):
        temperatures = result.temperature_history[step_index]
        nodal_fields = {
            "temperature": {
                node.id: result.node_temperature(node.id, step=step_index)
                for node in result.mesh.nodes
            }
        }
        gradients: dict[int, np.ndarray] = {}
        fluxes: dict[int, np.ndarray] = {}
        for element in result.mesh.elements:
            nodal_temperatures, resolved_element = _element_nodal_temperatures(
                result.mesh, element.id, temperatures, result.dof_map
            )
            gradients[element.id] = element_temperature_gradient(
                resolved_element, nodal_temperatures
            )
            fluxes[element.id] = element_heat_flux(
                resolved_element,
                result.materials[element.id],
                nodal_temperatures,
                result.evaluation_temperature,
            )
        element_fields = {"temperature_gradient": gradients, "heat_flux": fluxes}
        steps.append(
            ResultStep(
                index=step_index,
                time=float(time),
                nodal_fields=nodal_fields,
                element_fields=element_fields,
            )
        )
    return SimulationResult(
        topology,
        steps=tuple(steps),
        field_units={"temperature": "K", "temperature_gradient": "K/m", "heat_flux": "W/m^2"},
    )


def from_static_linear(result: AnalysisResult) -> SimulationResult:
    """Build a :class:`~femtoolkit.postprocessing.result_model.SimulationResult` from a
    solved :class:`~femtoolkit.results.analysis_result.AnalysisResult`.

    Args:
        result: The solved static linear result.

    Returns:
        A single-step :class:`~femtoolkit.postprocessing.result_model.SimulationResult`.
    """
    topology = MeshTopology.from_elements(result.elements)
    dofs_per_node = result.dof_map.dofs_per_node

    nodal_fields: dict[str, dict[int, np.ndarray]] = {
        "displacement": {}, "reaction": {}
    }
    for node_id in topology.node_ids:
        nodal_fields["displacement"][node_id] = np.array(
            [result.displacement(node_id, dof) for dof in range(dofs_per_node)]
        )
        nodal_fields["reaction"][node_id] = np.array(
            [result.reaction(node_id, dof) for dof in range(dofs_per_node)]
        )

    stresses: dict[int, np.ndarray] = {}
    strains: dict[int, np.ndarray] = {}
    for element in result.elements:
        try:
            stresses[element.id] = np.atleast_1d(result.element_stress(element.id))
            strains[element.id] = np.atleast_1d(result.element_strain(element.id))
        except InvalidElementError:
            continue

    step = ResultStep(
        index=0,
        time=0.0,
        nodal_fields=nodal_fields,
        element_fields={"stress": stresses, "strain": strains},
    )
    return SimulationResult(
        topology,
        steps=(step,),
        field_units={"displacement": "m", "reaction": "N", "stress": "Pa", "strain": ""},
    )


def from_nonlinear(result: NonlinearAnalysisResult, mesh: Mesh) -> SimulationResult:
    """Build a :class:`~femtoolkit.postprocessing.result_model.SimulationResult` from a
    solved :class:`~femtoolkit.results.nonlinear_result.NonlinearAnalysisResult`.

    Args:
        result: The solved nonlinear (load-stepped) result.
        mesh: The mesh the analysis was solved on (``NonlinearAnalysisResult``
            does not itself store node coordinates/connectivity).

    Returns:
        A :class:`~femtoolkit.postprocessing.result_model.SimulationResult`
        with one step per load increment, ``time`` set to each step's
        load factor.
    """
    topology = MeshTopology.from_mesh(mesh)

    steps = []
    for step_index, load_step in enumerate(result.step_results):
        nodal_fields: dict[str, dict[int, np.ndarray]] = {"displacement": {}, "reaction": {}}
        for node_id in topology.node_ids:
            nodal_fields["displacement"][node_id] = np.array(
                result.node_displacement(node_id, step=step_index)
            )
            nodal_fields["reaction"][node_id] = np.array(
                result.node_reaction(node_id, step=step_index)
            )

        stresses = {}
        strains = {}
        plastic_strains = {}
        for element_id in load_step.element_states:
            stresses[element_id] = np.atleast_1d(
                result.element_stress(element_id, step=step_index, gauss_point=_GAUSS_POINT)
            )
            strains[element_id] = np.atleast_1d(
                result.element_strain(element_id, step=step_index, gauss_point=_GAUSS_POINT)
            )
            plastic_strains[element_id] = np.atleast_1d(
                result.element_plastic_strain(
                    element_id, step=step_index, gauss_point=_GAUSS_POINT
                )
            )

        steps.append(
            ResultStep(
                index=step_index,
                time=float(load_step.load_factor),
                nodal_fields=nodal_fields,
                element_fields={
                    "stress": stresses,
                    "strain": strains,
                    "plastic_strain": plastic_strains,
                },
            )
        )
    return SimulationResult(
        topology,
        steps=tuple(steps),
        field_units={
            "displacement": "m",
            "reaction": "N",
            "stress": "Pa",
            "strain": "",
            "plastic_strain": "",
        },
    )


def from_dynamic(result: DynamicResult, mesh: Mesh) -> SimulationResult:
    """Build a :class:`~femtoolkit.postprocessing.result_model.SimulationResult` from a
    solved :class:`~femtoolkit.results.dynamic_result.DynamicResult`.

    Args:
        result: The solved dynamic (time-history) result.
        mesh: The mesh the analysis was solved on (``DynamicResult`` does
            not itself store node coordinates/connectivity or elements).

    Returns:
        A :class:`~femtoolkit.postprocessing.result_model.SimulationResult`
        with one step per stored time instant. Element stress/strain are
        recomputed at each time step from that step's displacement,
        reusing each element's own (already-tested)
        ``stress_from_dofs``/``strain_from_dofs`` methods -- exactly what
        :class:`~femtoolkit.results.analysis_result.AnalysisResult`
        already does for a single, static snapshot.
    """
    topology = MeshTopology.from_mesh(mesh)
    node_displacements = {
        node_id: result.node_displacement(node_id) for node_id in topology.node_ids
    }

    steps = []
    for step_index, time in enumerate(result.time):
        nodal_fields = {
            "displacement": {
                node_id: history[step_index] for node_id, history in node_displacements.items()
            }
        }

        stresses = {}
        strains = {}
        for element in mesh.elements:
            dof_values = [
                result.displacement_history[step_index, result.dof_map.global_index(node_id, dof)]
                for node_id, dof in element.dof_keys()
            ]
            try:
                stresses[element.id] = np.atleast_1d(element.stress_from_dofs(dof_values))
                strains[element.id] = np.atleast_1d(element.strain_from_dofs(dof_values))
            except AttributeError:
                continue

        steps.append(
            ResultStep(
                index=step_index,
                time=float(time),
                nodal_fields=nodal_fields,
                element_fields={"stress": stresses, "strain": strains},
            )
        )
    return SimulationResult(
        topology,
        steps=tuple(steps),
        field_units={"displacement": "m", "stress": "Pa", "strain": ""},
    )


def merge_thermomechanical(
    thermal: SimulationResult, mechanical: SimulationResult
) -> SimulationResult:
    """Correlate a thermal result with the mechanical result it drove (Version 22, section 10).

    For the sequential thermomechanical workflow (Version 19-21: thermal
    analysis -> temperature field -> thermal expansion -> mechanical
    analysis), this produces one result that carries **both** the solved
    temperature field and the resulting displacement/stress/strain, so a
    caller can correlate them directly (e.g. plot displacement against
    temperature at the same node) without re-deriving anything. The
    thermal result's *final* step's nodal temperature (and, where
    present, element temperature gradient/heat flux) is copied into
    every step of the mechanical result -- appropriate for a sequential
    workflow, where the temperature field is solved once and held fixed
    while the mechanical solve proceeds.

    This does **not** separately expose the thermal/mechanical/total
    strain decomposition (``epsilon = epsilon_mechanical +
    epsilon_thermal``, Version 19) -- that machinery already exists on
    :class:`~femtoolkit.materials.thermoelastic.ThermoelasticMaterial3D`
    (``thermal_strain_voigt``/``mechanical_strain_voigt``) and is
    directly callable by a caller who has the base material and the
    correlated temperature this function provides; duplicating it here
    would need the base material passed through this call for no benefit.

    Args:
        thermal: The thermal result (from :func:`from_thermal_steady_state`
            or :func:`from_thermal_transient`).
        mechanical: The mechanical result solved from ``thermal``'s
            temperature field (from :func:`from_nonlinear`, typically).

    Returns:
        A new :class:`~femtoolkit.postprocessing.result_model.SimulationResult`,
        using ``mechanical``'s topology and steps, with thermal fields merged in.
    """
    thermal_step = thermal.final_step
    merged_steps = []
    for step in mechanical.steps:
        nodal_fields = dict(step.nodal_fields)
        if "temperature" in thermal_step.nodal_fields:
            nodal_fields["temperature"] = dict(thermal_step.nodal_fields["temperature"])

        element_fields = dict(step.element_fields)
        for name in ("temperature_gradient", "heat_flux"):
            if name in thermal_step.element_fields:
                element_fields[name] = dict(thermal_step.element_fields[name])

        merged_steps.append(
            ResultStep(
                step.index, step.time, nodal_fields=nodal_fields, element_fields=element_fields
            )
        )

    field_units = {**thermal.field_units, **mechanical.field_units}
    return SimulationResult(mechanical.topology, tuple(merged_steps), field_units=field_units)
