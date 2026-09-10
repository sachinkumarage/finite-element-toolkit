"""Validation: TET4/HEX8 steady-state conduction, and volumetric heat generation.

Embeds the same 1D linear-conduction problem
(``tests/validation/test_thermal_steady_state_conduction.py``) inside a
genuine 3D solid (a unit-cube HEX8 block, and a TET4 patch), confirming
the 3D solid thermal elements reproduce the exact analytical result --
and separately verifies the classic 1D heat-generation benchmark
(``d/dx(k dT/dx) + Q = 0``) against its closed-form parabolic solution.
"""

import numpy as np
import pytest

from femtoolkit.materials import LinearElastic3D, Material
from femtoolkit.mesh import Hex8Element3D, Node, Tet4Element3D
from femtoolkit.mesh.bar_element import BarElement
from femtoolkit.mesh.mesh import Mesh
from femtoolkit.sections import CrossSection
from femtoolkit.thermal import (
    HeatGeneration,
    PrescribedTemperature,
    SteadyStateThermalAnalysis,
    ThermalMaterial,
    heat_generation_to_thermal_loads,
)

_CONDUCTIVITY = 50.0
_T0 = 373.15
_T_L = 293.15


def test_hex8_block_reproduces_exact_1d_linear_profile() -> None:
    coords = [
        (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0), (0.0, 1.0, 1.0),
    ]
    placeholder = LinearElastic3D(youngs_modulus=200e9, poisson_ratio=0.3, density=7850.0)
    nodes = tuple(Node(id=i + 1, x=x, y=y, z=z) for i, (x, y, z) in enumerate(coords))
    hexa = Hex8Element3D(id=1, nodes=nodes, material=placeholder)
    mesh = Mesh()
    for node in nodes:
        mesh.add_node(node)
    mesh.add_element(hexa)

    thermal_material = ThermalMaterial(
        thermal_conductivity=_CONDUCTIVITY, density=7850.0, specific_heat=460.0
    )
    analysis = SteadyStateThermalAnalysis(mesh, {hexa.id: thermal_material})
    for node in nodes:
        if node.x == 0.0:
            analysis.add_boundary_condition(PrescribedTemperature(node.id, _T0))
        else:
            analysis.add_boundary_condition(PrescribedTemperature(node.id, _T_L))
    result = analysis.solve()

    for node in nodes:
        expected = _T0 + (_T_L - _T0) * node.x / 1.0
        assert result.node_temperature(node.id) == pytest.approx(expected, abs=1e-9)

    expected_flux = -_CONDUCTIVITY * (_T_L - _T0) / 1.0
    flux = result.element_heat_flux(hexa.id)
    assert flux[0] == pytest.approx(expected_flux, rel=1e-8)
    assert flux[1] == pytest.approx(0.0, abs=1e-6)
    assert flux[2] == pytest.approx(0.0, abs=1e-6)


def test_tet4_patch_reproduces_exact_linear_profile() -> None:
    """A TET4 patch test: an imposed linear temperature field must reproduce that
    exact field and its constant gradient/flux everywhere (the thermal analogue of
    the mechanical patch test in tests/validation/test_tet4_patch.py)."""
    placeholder = LinearElastic3D(youngs_modulus=200e9, poisson_ratio=0.3, density=7850.0)
    nodes = (
        Node(id=1, x=0.0, y=0.0, z=0.0),
        Node(id=2, x=1.0, y=0.0, z=0.0),
        Node(id=3, x=0.3, y=1.0, z=0.1),
        Node(id=4, x=0.2, y=0.2, z=1.0),
    )
    tet = Tet4Element3D(id=1, nodes=nodes, material=placeholder)
    mesh = Mesh()
    for node in nodes:
        mesh.add_node(node)
    mesh.add_element(tet)

    thermal_material = ThermalMaterial(
        thermal_conductivity=_CONDUCTIVITY, density=7850.0, specific_heat=460.0
    )
    analysis = SteadyStateThermalAnalysis(mesh, {tet.id: thermal_material})

    gradient = np.array([40.0, 0.0, 0.0])  # K/m, imposed field T = T0 + gradient . X
    for node in nodes:
        position = np.array([node.x, node.y, node.z])
        temperature = _T0 + float(gradient @ position)
        analysis.add_boundary_condition(PrescribedTemperature(node.id, temperature))

    result = analysis.solve()
    for node in nodes:
        position = np.array([node.x, node.y, node.z])
        expected = _T0 + float(gradient @ position)
        assert result.node_temperature(node.id) == pytest.approx(expected, abs=1e-8)

    recovered_gradient = result.element_temperature_gradient(tet.id)
    assert recovered_gradient == pytest.approx(gradient, abs=1e-8)

    flux = result.element_heat_flux(tet.id)
    assert flux == pytest.approx(-_CONDUCTIVITY * gradient, abs=1e-6)


# --- Heat generation: 1D closed-form parabolic benchmark ----------------------


def test_1d_heat_generation_matches_parabolic_analytical_solution() -> None:
    """d/dx(k dT/dx) + Q = 0, T(0)=T(L)=T0 -> T(x) = T0 + (Q/(2k))*x*(L-x)."""
    length = 2.0
    q_generation = 1.0e5
    t_boundary = 293.15
    num_elements = 20

    mechanical_material = Material(
        name="steel", density=7850.0, youngs_modulus=200e9, poissons_ratio=0.3
    )
    thermal_material = ThermalMaterial(
        thermal_conductivity=_CONDUCTIVITY, density=7850.0, specific_heat=460.0
    )
    x_coords = np.linspace(0.0, length, num_elements + 1)
    nodes = [Node(id=i + 1, x=float(x), y=0.0, z=0.0) for i, x in enumerate(x_coords)]
    mesh = Mesh()
    for node in nodes:
        mesh.add_node(node)

    materials: dict[int, ThermalMaterial] = {}
    for i in range(num_elements):
        element = BarElement(
            id=i + 1,
            nodes=(nodes[i], nodes[i + 1]),
            material=mechanical_material,
            cross_section=CrossSection(area=0.01),
        )
        mesh.add_element(element)
        materials[element.id] = thermal_material

    analysis = SteadyStateThermalAnalysis(mesh, materials)
    analysis.add_boundary_condition(PrescribedTemperature(nodes[0].id, t_boundary))
    analysis.add_boundary_condition(PrescribedTemperature(nodes[-1].id, t_boundary))
    loads = heat_generation_to_thermal_loads(mesh, HeatGeneration(volumetric_rate=q_generation))
    analysis.add_thermal_loads(loads)
    result = analysis.solve()

    for node in nodes:
        expected = t_boundary + (q_generation / (2.0 * _CONDUCTIVITY)) * node.x * (length - node.x)
        assert result.node_temperature(node.id) == pytest.approx(expected, rel=1e-3, abs=1e-2)


def test_zero_heat_generation_reduces_to_pure_conduction() -> None:
    mechanical_material = Material(
        name="steel", density=7850.0, youngs_modulus=200e9, poissons_ratio=0.3
    )
    thermal_material = ThermalMaterial(
        thermal_conductivity=_CONDUCTIVITY, density=7850.0, specific_heat=460.0
    )
    n1 = Node(id=1, x=0.0, y=0.0, z=0.0)
    n2 = Node(id=2, x=2.0, y=0.0, z=0.0)
    element = BarElement(
        id=1, nodes=(n1, n2), material=mechanical_material, cross_section=CrossSection(area=0.01)
    )
    mesh = Mesh()
    mesh.add_node(n1)
    mesh.add_node(n2)
    mesh.add_element(element)

    analysis = SteadyStateThermalAnalysis(mesh, {element.id: thermal_material})
    analysis.add_boundary_condition(PrescribedTemperature(n1.id, _T0))
    analysis.add_boundary_condition(PrescribedTemperature(n2.id, _T_L))
    loads = heat_generation_to_thermal_loads(mesh, HeatGeneration(volumetric_rate=0.0))
    analysis.add_thermal_loads(loads)
    result = analysis.solve()

    assert result.node_temperature(n1.id) == pytest.approx(_T0)
    assert result.node_temperature(n2.id) == pytest.approx(_T_L)
