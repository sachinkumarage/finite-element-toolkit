"""Volumetric heat generation, converted into equivalent nodal thermal loads (Version 20).

A volumetric heat source ``Q`` (W/m^3) acts throughout an element's
volume -- a body-force analogue for heat, exactly like
:class:`~femtoolkit.analysis.body_load.GravityLoad` is for mechanics.
Its equivalent nodal contribution to the thermal load vector is obtained
the same standard way, integrating against the element's own shape
functions:

.. code-block:: text

    Fe = integral( N^T * Q ) dV

For CST/TET4 (whose shape functions are the triangle/tetrahedron's
area/volume coordinates), this splits the total heat generation *evenly*
across the element's nodes, via the same closed-form area/volume-
coordinate integral identity already used throughout
:mod:`femtoolkit.continuum.mass`/:mod:`femtoolkit.analysis.body_load`
(``integral(Li) dA = A/3``, ``integral(Li) dV = V/4``). For Q4/HEX8
(bilinear/trilinear shape functions, no such closed form on a general
element), this reuses the same Gauss quadrature already used for their
conductivity/capacity matrices.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from femtoolkit.continuum.gauss import GAUSS_2X2_POINTS, GAUSS_2X2X2_POINTS
from femtoolkit.continuum.jacobian import (
    physical_shape_function_derivatives,
    physical_shape_function_derivatives_3d,
)
from femtoolkit.continuum.shape_functions import (
    hex8_shape_function_derivatives,
    hex8_shape_functions,
    quad_shape_function_derivatives,
    quad_shape_functions,
)
from femtoolkit.exceptions import ValidationError
from femtoolkit.mesh.bar_element import BarElement
from femtoolkit.mesh.cst_element import CSTElement2D
from femtoolkit.mesh.hex8_element import Hex8Element3D
from femtoolkit.mesh.mesh import Mesh
from femtoolkit.mesh.quad_element import QuadElement2D
from femtoolkit.mesh.tet4_element import Tet4Element3D


@dataclass(frozen=True)
class ThermalLoad:
    """One nodal contribution to the global thermal load vector ``F_T``.

    The thermal analogue of :class:`~femtoolkit.analysis.loads.NodalLoad`
    -- kept as a separate, dedicated type (rather than reusing
    ``NodalLoad`` directly) so a heat-flow value (watts) is never
    confused with a mechanical force (newtons), even though both are,
    mechanically speaking, "a value added to one DOF's row of a load
    vector."

    Attributes:
        node_id: The node this contribution applies to.
        value: The heat-flow value, in watts. Positive means heat
            flowing *into* the node.
    """

    node_id: int
    value: float


@dataclass(frozen=True)
class HeatGeneration:
    """A uniform volumetric heat source, ``Q`` (W/m^3), applied to every element of a mesh.

    Attributes:
        volumetric_rate: The heat generation rate per unit volume, in
            W/m^3. May be negative (a heat *sink*, e.g. representing a
            simplified cooling effect) or exactly zero.

    Raises:
        ValidationError: If ``volumetric_rate`` is not finite.

    Example:
        >>> heater = HeatGeneration(volumetric_rate=5.0e4)
    """

    volumetric_rate: float

    def __post_init__(self) -> None:
        """Validate the heat generation rate immediately after construction.

        Raises:
            ValidationError: If ``volumetric_rate`` is not finite.
        """
        if not math.isfinite(self.volumetric_rate):
            raise ValidationError(
                f"HeatGeneration volumetric_rate must be finite, got {self.volumetric_rate}."
            )


def _bar_heat_generation_loads(element: BarElement, volumetric_rate: float) -> list[ThermalLoad]:
    total = volumetric_rate * element.cross_section.area * element.length
    share = total / 2.0
    return [ThermalLoad(node.id, share) for node in element.nodes]


def _cst_heat_generation_loads(element: CSTElement2D, volumetric_rate: float) -> list[ThermalLoad]:
    total = volumetric_rate * element.area * element.thickness
    share = total / 3.0
    return [ThermalLoad(node.id, share) for node in element.nodes]


def _tet4_heat_generation_loads(
    element: Tet4Element3D, volumetric_rate: float
) -> list[ThermalLoad]:
    total = volumetric_rate * element.volume
    share = total / 4.0
    return [ThermalLoad(node.id, share) for node in element.nodes]


def _quad_heat_generation_loads(
    element: QuadElement2D, volumetric_rate: float
) -> list[ThermalLoad]:
    x_coords = tuple(node.x for node in element.nodes)
    y_coords = tuple(node.y for node in element.nodes)

    node_contributions = [0.0, 0.0, 0.0, 0.0]
    for point in GAUSS_2X2_POINTS:
        n_values = quad_shape_functions(point.xi, point.eta)
        dn_dxi, dn_deta = quad_shape_function_derivatives(point.xi, point.eta)
        _, _, det_j = physical_shape_function_derivatives(dn_dxi, dn_deta, x_coords, y_coords)
        weight_factor = point.weight * det_j * volumetric_rate * element.thickness
        for i, n_value in enumerate(n_values):
            node_contributions[i] += weight_factor * n_value

    return [
        ThermalLoad(node.id, contribution)
        for node, contribution in zip(element.nodes, node_contributions, strict=True)
    ]


def _hex8_heat_generation_loads(
    element: Hex8Element3D, volumetric_rate: float
) -> list[ThermalLoad]:
    x = tuple(node.x for node in element.nodes)
    y = tuple(node.y for node in element.nodes)
    z = tuple(node.z for node in element.nodes)

    node_contributions = [0.0] * 8
    for point in GAUSS_2X2X2_POINTS:
        n_values = hex8_shape_functions(point.xi, point.eta, point.zeta)
        dn_dxi, dn_deta, dn_dzeta = hex8_shape_function_derivatives(point.xi, point.eta, point.zeta)
        _, _, _, det_j = physical_shape_function_derivatives_3d(
            dn_dxi, dn_deta, dn_dzeta, x, y, z
        )
        weight_factor = point.weight * det_j * volumetric_rate
        for i, n_value in enumerate(n_values):
            node_contributions[i] += weight_factor * n_value

    return [
        ThermalLoad(node.id, contribution)
        for node, contribution in zip(element.nodes, node_contributions, strict=True)
    ]


def heat_generation_to_thermal_loads(
    mesh: Mesh, heat_generation: HeatGeneration
) -> list[ThermalLoad]:
    """Convert a uniform volumetric heat source into equivalent nodal thermal loads.

    Every thermally-capable element (bar, CST, Q4, TET4, HEX8) in
    ``mesh`` contributes its own share; elements sharing a node each add
    their own contribution, which
    :func:`~femtoolkit.thermal.thermal_analysis.build_thermal_force_vector`
    sums.

    Args:
        mesh: The mesh to compute heat-generation loads for. Other
            element types are skipped.
        heat_generation: The uniform volumetric heat source to convert.

    Returns:
        One :class:`ThermalLoad` per node per contributing element.
    """
    rate = heat_generation.volumetric_rate
    thermal_loads: list[ThermalLoad] = []
    for element in mesh.elements:
        if isinstance(element, BarElement):
            thermal_loads.extend(_bar_heat_generation_loads(element, rate))
        elif isinstance(element, CSTElement2D):
            thermal_loads.extend(_cst_heat_generation_loads(element, rate))
        elif isinstance(element, QuadElement2D):
            thermal_loads.extend(_quad_heat_generation_loads(element, rate))
        elif isinstance(element, Tet4Element3D):
            thermal_loads.extend(_tet4_heat_generation_loads(element, rate))
        elif isinstance(element, Hex8Element3D):
            thermal_loads.extend(_hex8_heat_generation_loads(element, rate))
    return thermal_loads


def element_heat_generation_to_thermal_loads(
    mesh: Mesh, volumetric_rates: dict[int, float]
) -> list[ThermalLoad]:
    """Convert *per-element* volumetric heat generation rates into equivalent nodal loads.

    The element-based counterpart to :func:`heat_generation_to_thermal_loads`'s
    mesh-wide uniform rate -- e.g. a localized heater represented by a
    high generation rate on just a few elements.

    Args:
        mesh: The mesh to compute heat-generation loads for.
        volumetric_rates: Maps a subset of ``mesh``'s element IDs to
            their own heat generation rate, in W/m^3. Elements not
            present in this mapping (or not thermally-capable) are
            skipped.

    Returns:
        One :class:`ThermalLoad` per node per contributing element.
    """
    thermal_loads: list[ThermalLoad] = []
    for element in mesh.elements:
        if element.id not in volumetric_rates:
            continue
        rate = volumetric_rates[element.id]
        if isinstance(element, BarElement):
            thermal_loads.extend(_bar_heat_generation_loads(element, rate))
        elif isinstance(element, CSTElement2D):
            thermal_loads.extend(_cst_heat_generation_loads(element, rate))
        elif isinstance(element, QuadElement2D):
            thermal_loads.extend(_quad_heat_generation_loads(element, rate))
        elif isinstance(element, Tet4Element3D):
            thermal_loads.extend(_tet4_heat_generation_loads(element, rate))
        elif isinstance(element, Hex8Element3D):
            thermal_loads.extend(_hex8_heat_generation_loads(element, rate))
    return thermal_loads
