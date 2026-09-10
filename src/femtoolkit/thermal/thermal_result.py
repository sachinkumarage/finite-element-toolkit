"""Results of a heat-conduction analysis (Version 20).

Mirrors :class:`~femtoolkit.results.analysis_result.AnalysisResult` (the
static mechanical result) and
:class:`~femtoolkit.results.dynamic_result.DynamicResult` (the time-
history mechanical result): a read-only view over the solved nodal
temperature(s), with convenience accessors for per-element temperature
gradient and heat flux (see
:mod:`femtoolkit.thermal.thermal_elements`), and a bridge back into
Version 19's thermomechanical world via :meth:`SteadyStateThermalResult.to_temperature_field`/
:meth:`TransientThermalResult.to_temperature_field` -- see
:mod:`femtoolkit.thermal.thermal_analysis`'s module docstring for the
full sequential thermal -> mechanical workflow this exists for.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from femtoolkit.analysis.dof import DOFMap
from femtoolkit.analysis.temperature_field import TemperatureField
from femtoolkit.mesh.mesh import Mesh
from femtoolkit.thermal.thermal_elements import (
    element_heat_flux,
    element_temperature_gradient,
    thermal_dof_keys,
)
from femtoolkit.thermal.thermal_material import ThermalMaterial


def _element_nodal_temperatures(
    mesh: Mesh, element_id: int, temperatures: np.ndarray, dof_map: DOFMap
) -> tuple[np.ndarray, object]:
    element = mesh.get_element(element_id)
    indices = [dof_map.global_index(node_id, dof) for node_id, dof in thermal_dof_keys(element)]
    return temperatures[indices], element


@dataclass(frozen=True)
class SteadyStateThermalResult:
    """Read-only results of a steady-state heat-conduction analysis.

    Instances are produced by
    :meth:`~femtoolkit.thermal.thermal_analysis.SteadyStateThermalAnalysis.solve`
    and should not be constructed directly by application code.

    Attributes:
        dof_map: DOF map used for the analysis (one DOF per node).
        temperatures: The solved global nodal temperature vector, in
            kelvin, ordered per ``dof_map``.
        mesh: The mesh the analysis was solved on.
        materials: Maps each element's ID to the
            :class:`~femtoolkit.thermal.thermal_material.ThermalMaterial`
            it was solved with.
        evaluation_temperature: The (fixed) temperature used to evaluate
            temperature-dependent conductivity for this solve.
    """

    dof_map: DOFMap
    temperatures: np.ndarray
    mesh: Mesh
    materials: dict[int, ThermalMaterial]
    evaluation_temperature: float

    def node_temperature(self, node_id: int) -> float:
        """Return the solved temperature at ``node_id``, in kelvin."""
        return float(self.temperatures[self.dof_map.global_index(node_id, 0)])

    def element_temperature_gradient(self, element_id: int) -> np.ndarray:
        """Return an element's temperature gradient, ``grad(T)``, in K/m.

        See :func:`~femtoolkit.thermal.thermal_elements.element_temperature_gradient`.
        """
        nodal_temperatures, element = _element_nodal_temperatures(
            self.mesh, element_id, self.temperatures, self.dof_map
        )
        return element_temperature_gradient(element, nodal_temperatures)

    def element_heat_flux(self, element_id: int) -> np.ndarray:
        """Return an element's heat flux, ``q = -k*grad(T)`` (Fourier's law), in W/m^2.

        See :func:`~femtoolkit.thermal.thermal_elements.element_heat_flux`.
        """
        nodal_temperatures, element = _element_nodal_temperatures(
            self.mesh, element_id, self.temperatures, self.dof_map
        )
        material = self.materials[element_id]
        return element_heat_flux(element, material, nodal_temperatures, self.evaluation_temperature)

    def to_temperature_field(self) -> TemperatureField:
        """Return this result as a :class:`~femtoolkit.analysis.temperature_field.TemperatureField`.

        The bridge into Version 19's thermomechanical workflow: the
        solved nodal temperatures become a nodal
        :class:`~femtoolkit.analysis.temperature_field.TemperatureField`,
        directly usable with
        :func:`~femtoolkit.analysis.temperature_field.thermoelastic_materials_for_mesh`.
        """
        temperatures_by_node = {
            node.id: self.node_temperature(node.id) for node in self.mesh.nodes
        }
        return TemperatureField.nodal(temperatures_by_node)


@dataclass(frozen=True)
class TransientThermalResult:
    """Read-only results of a transient heat-conduction analysis.

    Instances are produced by
    :meth:`~femtoolkit.thermal.thermal_analysis.TransientThermalAnalysis.solve`
    and should not be constructed directly by application code.

    Attributes:
        dof_map: DOF map used for the analysis (one DOF per node).
        times: The time (seconds) at the end of each stored step,
            including the initial condition at ``times[0] == 0.0``.
        temperature_history: One global nodal temperature vector (kelvin)
            per entry of ``times``, in matching order.
        mesh: The mesh the analysis was solved on.
        materials: Maps each element's ID to the
            :class:`~femtoolkit.thermal.thermal_material.ThermalMaterial`
            it was solved with.
        evaluation_temperature: The (fixed) temperature used to evaluate
            temperature-dependent conductivity/capacity for this solve.
    """

    dof_map: DOFMap
    times: np.ndarray
    temperature_history: list[np.ndarray]
    mesh: Mesh
    materials: dict[int, ThermalMaterial]
    evaluation_temperature: float

    def node_temperature(self, node_id: int, step: int = -1) -> float:
        """Return the solved temperature at ``node_id`` at ``step`` (default: the last)."""
        return float(self.temperature_history[step][self.dof_map.global_index(node_id, 0)])

    def node_temperature_history(self, node_id: int) -> np.ndarray:
        """Return the solved temperature at ``node_id`` across every stored step."""
        index = self.dof_map.global_index(node_id, 0)
        return np.array([temperatures[index] for temperatures in self.temperature_history])

    def to_temperature_field(self, step: int = -1) -> TemperatureField:
        """Return one time step's result as a :class:`TemperatureField`.

        Args:
            step: Which stored step to convert (default: the last, the
                final steady/transient state reached).
        """
        temperatures_by_node = {
            node.id: self.node_temperature(node.id, step) for node in self.mesh.nodes
        }
        return TemperatureField.nodal(temperatures_by_node)
