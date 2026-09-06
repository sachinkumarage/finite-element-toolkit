"""A prescribed temperature distribution over a mesh (Version 19).

Section 8's brief asks for a thermal-load representation that can express
a uniform temperature change, nodal temperature values, or
element/integration-point temperature -- and, notably, asks for the
architecture to be reusable by a *future* heat-conduction solver
(explicitly out of scope for this version; see the Version 20 preview).
:class:`TemperatureField` is that single, minimal representation: either
a **uniform** temperature (the whole mesh at one value) or **nodal**
temperatures (one value per node, interpolated to each element the same
way displacement is). This is deliberately the *output* shape a future
heat-conduction solve would naturally produce (a solved nodal
temperature field) -- so that, when Version 20 adds one, its result
becomes a drop-in :class:`TemperatureField`, immediately consumable by
every mechanical/thermoelastic consumer already built against this class,
with no changes needed on this side.

This module intentionally does **not** solve for temperature -- it only
represents a *given* one, exactly like
:class:`~femtoolkit.analysis.loads.NodalLoad` represents a given force,
not something the toolkit computes from first principles.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from femtoolkit.exceptions import ValidationError
from femtoolkit.materials.thermoelastic import (
    ThermoelasticMaterial3D,
    ThermoelasticMaterialAtTemperature,
)

if TYPE_CHECKING:
    from femtoolkit.mesh.hex8_element import Hex8Element3D
    from femtoolkit.mesh.mesh import Mesh
    from femtoolkit.mesh.tet4_element import Tet4Element3D


@dataclass(frozen=True)
class TemperatureField:
    """A prescribed temperature distribution over a mesh: uniform or per-node.

    Not constructed directly -- use :meth:`uniform` or :meth:`nodal`.

    Example:
        >>> uniform = TemperatureField.uniform(373.15)
        >>> uniform.node_temperature(node_id=1)
        373.15
        >>> nodal = TemperatureField.nodal({1: 300.0, 2: 350.0})
        >>> nodal.node_temperature(node_id=2)
        350.0
    """

    _uniform_temperature: float | None = field(default=None, repr=False)
    _nodal_temperatures: Mapping[int, float] | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Validate that exactly one representation was supplied.

        Raises:
            ValidationError: If neither or both of the uniform/nodal
                representations were supplied (use :meth:`uniform` or
                :meth:`nodal` instead of constructing directly).
        """
        has_uniform = self._uniform_temperature is not None
        has_nodal = self._nodal_temperatures is not None
        if has_uniform == has_nodal:
            raise ValidationError(
                "TemperatureField must be built via TemperatureField.uniform(...) or "
                "TemperatureField.nodal(...), not constructed directly."
            )

    @classmethod
    def uniform(cls, temperature: float) -> TemperatureField:
        """Build a temperature field that is the same everywhere.

        Args:
            temperature: The temperature, in kelvin, applied to every
                node and element.

        Returns:
            A uniform :class:`TemperatureField`.

        Raises:
            ValidationError: If ``temperature`` is not finite.
        """
        if not np.isfinite(temperature):
            raise ValidationError(
                f"TemperatureField temperature must be finite, got {temperature}."
            )
        return cls(_uniform_temperature=float(temperature))

    @classmethod
    def nodal(cls, temperatures: Mapping[int, float]) -> TemperatureField:
        """Build a temperature field from per-node values.

        Args:
            temperatures: Maps each node's ``id`` to its temperature, in
                kelvin. Must not be empty.

        Returns:
            A nodal :class:`TemperatureField`.

        Raises:
            ValidationError: If ``temperatures`` is empty, or any value
                is not finite.
        """
        if not temperatures:
            raise ValidationError("TemperatureField.nodal requires at least one node temperature.")
        for node_id, value in temperatures.items():
            if not np.isfinite(value):
                raise ValidationError(
                    f"TemperatureField temperature for node {node_id} must be finite, got {value}."
                )
        return cls(_nodal_temperatures=dict(temperatures))

    def node_temperature(self, node_id: int) -> float:
        """Return the temperature prescribed at ``node_id``.

        Args:
            node_id: The node to query.

        Returns:
            The temperature, in kelvin.

        Raises:
            ValidationError: If this is a nodal field and ``node_id``
                has no prescribed temperature.
        """
        if self._uniform_temperature is not None:
            return self._uniform_temperature
        if node_id not in self._nodal_temperatures:
            raise ValidationError(f"TemperatureField has no temperature for node {node_id}.")
        return self._nodal_temperatures[node_id]

    def element_temperature(self, element: Tet4Element3D | Hex8Element3D) -> float:
        """Return a single, element-uniform temperature for ``element``.

        The mean of the element's node temperatures (exact for a
        uniform field; a simple, explicitly documented per-element
        approximation for a nodal field -- Version 19 evaluates every
        Gauss point of an element at the same temperature, matching the
        one-``ThermoelasticMaterialAtTemperature``-per-element
        architecture; genuinely per-Gauss-point temperature is a natural
        future extension, not needed for this version's scope).

        Args:
            element: The TET4 or HEX8 element to query.

        Returns:
            The element's (uniform) temperature, in kelvin.
        """
        return float(np.mean([self.node_temperature(node.id) for node in element.nodes]))


def thermoelastic_materials_for_mesh(
    mesh: Mesh, base_material: ThermoelasticMaterial3D, temperature_field: TemperatureField
) -> dict[int, ThermoelasticMaterialAtTemperature]:
    """Build a per-element material mapping, ready for
    :class:`~femtoolkit.analysis.nonlinear_analysis.NonlinearAnalysis`.

    Every element gets its own
    :class:`~femtoolkit.materials.thermoelastic.ThermoelasticMaterialAtTemperature`,
    bound to ``base_material`` at that element's temperature
    (:meth:`TemperatureField.element_temperature`) -- this is how a
    spatially-varying :class:`TemperatureField` (uniform or nodal)
    becomes usable with the existing, unmodified TET4/HEX8 dispatch (see
    :mod:`femtoolkit.materials.thermoelastic`'s module docstring for why
    a per-element material, rather than a per-call temperature argument,
    is this version's design).

    Args:
        mesh: The mesh to build materials for. Only TET4/HEX8 elements
            are supported.
        base_material: The temperature-dependent material every element
            is evaluated from.
        temperature_field: The prescribed temperature distribution.

    Returns:
        Maps each element's ``id`` to its
        :class:`~femtoolkit.materials.thermoelastic.ThermoelasticMaterialAtTemperature`.
    """
    return {
        element.id: base_material.at_temperature(temperature_field.element_temperature(element))
        for element in mesh.elements
    }
