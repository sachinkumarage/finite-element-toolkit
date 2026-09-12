"""Validation: convection against a known closed-form solution (spec section 20).

The mandatory "convection problem with a known simplified solution."
For a 1D slab of conductivity ``k``, length ``L``, and unit
cross-sectional area, held at ``T0`` on one face and exposed to
convection (coefficient ``h``, ambient ``T_infinity``) on the other,
steady-state energy balance -- the heat conducted through the slab must
equal the heat carried away by convection -- gives the classic
composite-thermal-resistance result:

.. code-block:: text

    (T0 - T_L) / (L / (k*A)) = h*A*(T_L - T_infinity)
    =>  T_L = (k*T0 + h*L*T_infinity) / (k + h*L)

This file embeds that 1D problem inside a unit-cube HEX8 element (one
face held at ``T0``, the opposite face exposed to convection, the
remaining four faces left unconstrained/insulated) and verifies the
finite element solution matches this closed form exactly -- since HEX8's
trilinear shape functions, and the linear edge shape functions
underlying the convection surface integral, are both exact for this
purely 1D, linear-in-x temperature field.
"""

import pytest

from femtoolkit.materials import LinearElastic3D
from femtoolkit.mesh import Hex8Element3D, Mesh, Node
from femtoolkit.thermal import PrescribedTemperature, SteadyStateThermalAnalysis, ThermalMaterial
from femtoolkit.thermal.thermal_boundary_conditions import ConvectionBoundaryCondition
from femtoolkit.thermal.thermal_surfaces import ThermalSurface

_COORDS = [
    (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0), (0.0, 1.0, 1.0),
]
_RIGHT_FACE_INDEX = 4  # xi = +1, the x = 1 face -- see HEX8_FACES


def _build_slab() -> tuple[Mesh, tuple[Node, ...]]:
    placeholder = LinearElastic3D(youngs_modulus=200e9, poisson_ratio=0.3, density=7850.0)
    nodes = tuple(Node(id=i + 1, x=x, y=y, z=z) for i, (x, y, z) in enumerate(_COORDS))
    hexa = Hex8Element3D(id=1, nodes=nodes, material=placeholder)
    mesh = Mesh()
    for node in nodes:
        mesh.add_node(node)
    mesh.add_element(hexa)
    return mesh, nodes


@pytest.mark.parametrize(
    ("conductivity", "coefficient", "hot_temperature", "ambient_temperature"),
    [
        (50.0, 10.0, 400.0, 300.0),
        (50.0, 1000.0, 400.0, 300.0),  # very large h: right face should approach ambient
        (50.0, 0.01, 400.0, 300.0),  # very small h: right face should approach the hot face
        (10.0, 25.0, 373.15, 293.15),
    ],
)
def test_conduction_convection_composite_resistance_matches_closed_form(
    conductivity: float, coefficient: float, hot_temperature: float, ambient_temperature: float
) -> None:
    mesh, nodes = _build_slab()
    thermal_material = ThermalMaterial(
        thermal_conductivity=conductivity, density=7850.0, specific_heat=460.0
    )
    analysis = SteadyStateThermalAnalysis(mesh, {1: thermal_material})
    for node in nodes:
        if node.x == 0.0:
            analysis.add_boundary_condition(PrescribedTemperature(node.id, hot_temperature))
    right_face = ThermalSurface(element_id=1, local_face_index=_RIGHT_FACE_INDEX)
    analysis.add_convection(
        ConvectionBoundaryCondition(
            surface=right_face,
            convection_coefficient=coefficient,
            ambient_temperature=ambient_temperature,
        )
    )
    result = analysis.solve()

    length = 1.0
    expected = (conductivity * hot_temperature + coefficient * length * ambient_temperature) / (
        conductivity + coefficient * length
    )
    for node in nodes:
        if node.x == 1.0:
            assert result.node_temperature(node.id) == pytest.approx(expected, rel=1e-9)


def test_convection_heat_flux_matches_conductive_flux_at_steady_state() -> None:
    """At steady state, the heat conducted through the slab equals the heat convected away."""
    mesh, nodes = _build_slab()
    conductivity, coefficient, hot_temperature, ambient_temperature = 50.0, 10.0, 400.0, 300.0
    thermal_material = ThermalMaterial(
        thermal_conductivity=conductivity, density=7850.0, specific_heat=460.0
    )
    analysis = SteadyStateThermalAnalysis(mesh, {1: thermal_material})
    for node in nodes:
        if node.x == 0.0:
            analysis.add_boundary_condition(PrescribedTemperature(node.id, hot_temperature))
    right_face = ThermalSurface(element_id=1, local_face_index=_RIGHT_FACE_INDEX)
    analysis.add_convection(
        ConvectionBoundaryCondition(
            surface=right_face,
            convection_coefficient=coefficient,
            ambient_temperature=ambient_temperature,
        )
    )
    result = analysis.solve()

    conductive_flux = result.element_heat_flux(1)[0]
    right_face_temperature = result.node_temperature(2)
    convective_flux = coefficient * (right_face_temperature - ambient_temperature)
    assert conductive_flux == pytest.approx(convective_flux, rel=1e-9)
