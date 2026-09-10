"""Validation: steady-state heat conduction against the closed-form 1D solution (spec section 18).

For a bar with constant conductivity and no internal heat generation,
the steady-state heat equation reduces to ``d^2T/dx^2 = 0``, whose
solution is the linear profile:

.. code-block:: text

    T(x) = T0 + (TL - T0) * x / L

This is the toolkit's mandatory 1D conduction benchmark. Since the 1D
conduction element's shape functions are exactly linear, the finite
element solution should match this analytical solution to floating-
point precision, not just "converge toward it" -- verified directly
here, along with the resulting heat flux, ``q = -k * dT/dx``.
"""

import numpy as np
import pytest

from femtoolkit.materials import Material
from femtoolkit.mesh import Node
from femtoolkit.mesh.bar_element import BarElement
from femtoolkit.mesh.mesh import Mesh
from femtoolkit.sections import CrossSection
from femtoolkit.thermal import PrescribedTemperature, SteadyStateThermalAnalysis, ThermalMaterial

_CONDUCTIVITY = 50.0  # W/(m*K)
_LENGTH = 2.0  # m
_T0 = 373.15  # K
_T_L = 293.15  # K


def _build_bar_mesh(num_elements: int) -> tuple[Mesh, dict[int, ThermalMaterial], list[Node]]:
    mechanical_material = Material(
        name="steel", density=7850.0, youngs_modulus=200e9, poissons_ratio=0.3
    )
    thermal_material = ThermalMaterial(
        thermal_conductivity=_CONDUCTIVITY, density=7850.0, specific_heat=460.0
    )
    x_coords = np.linspace(0.0, _LENGTH, num_elements + 1)
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

    return mesh, materials, nodes


@pytest.mark.parametrize("num_elements", [1, 2, 5, 10])
def test_1d_conduction_matches_analytical_linear_profile_exactly(num_elements: int) -> None:
    mesh, materials, nodes = _build_bar_mesh(num_elements)
    analysis = SteadyStateThermalAnalysis(mesh, materials)
    analysis.add_boundary_condition(PrescribedTemperature(nodes[0].id, _T0))
    analysis.add_boundary_condition(PrescribedTemperature(nodes[-1].id, _T_L))
    result = analysis.solve()

    for node in nodes:
        expected = _T0 + (_T_L - _T0) * node.x / _LENGTH
        assert result.node_temperature(node.id) == pytest.approx(expected, abs=1e-9)


@pytest.mark.parametrize("num_elements", [1, 4, 10])
def test_1d_conduction_heat_flux_matches_fourier_law(num_elements: int) -> None:
    mesh, materials, nodes = _build_bar_mesh(num_elements)
    analysis = SteadyStateThermalAnalysis(mesh, materials)
    analysis.add_boundary_condition(PrescribedTemperature(nodes[0].id, _T0))
    analysis.add_boundary_condition(PrescribedTemperature(nodes[-1].id, _T_L))
    result = analysis.solve()

    expected_flux = -_CONDUCTIVITY * (_T_L - _T0) / _LENGTH
    for element_id in materials:
        flux = result.element_heat_flux(element_id)
        assert flux[0] == pytest.approx(expected_flux, rel=1e-8)


def test_1d_conduction_flux_is_uniform_across_all_elements() -> None:
    """Steady-state, no generation -> heat flux must be exactly uniform along the bar."""
    mesh, materials, nodes = _build_bar_mesh(num_elements=8)
    analysis = SteadyStateThermalAnalysis(mesh, materials)
    analysis.add_boundary_condition(PrescribedTemperature(nodes[0].id, _T0))
    analysis.add_boundary_condition(PrescribedTemperature(nodes[-1].id, _T_L))
    result = analysis.solve()

    fluxes = [result.element_heat_flux(element_id)[0] for element_id in materials]
    assert all(flux == pytest.approx(fluxes[0], rel=1e-8) for flux in fluxes)


def test_no_temperature_difference_gives_uniform_temperature_and_zero_flux() -> None:
    mesh, materials, nodes = _build_bar_mesh(num_elements=4)
    analysis = SteadyStateThermalAnalysis(mesh, materials)
    analysis.add_boundary_condition(PrescribedTemperature(nodes[0].id, 300.0))
    analysis.add_boundary_condition(PrescribedTemperature(nodes[-1].id, 300.0))
    result = analysis.solve()

    for node in nodes:
        assert result.node_temperature(node.id) == pytest.approx(300.0)
    for element_id in materials:
        assert result.element_heat_flux(element_id)[0] == pytest.approx(0.0, abs=1e-8)


def test_finer_mesh_does_not_degrade_accuracy() -> None:
    """The FE solution is exact for this problem at any mesh density (linear exact solution)."""
    errors = []
    for num_elements in (1, 2, 4, 8, 16):
        mesh, materials, nodes = _build_bar_mesh(num_elements)
        analysis = SteadyStateThermalAnalysis(mesh, materials)
        analysis.add_boundary_condition(PrescribedTemperature(nodes[0].id, _T0))
        analysis.add_boundary_condition(PrescribedTemperature(nodes[-1].id, _T_L))
        result = analysis.solve()
        max_error = max(
            abs(result.node_temperature(node.id) - (_T0 + (_T_L - _T0) * node.x / _LENGTH))
            for node in nodes
        )
        errors.append(max_error)
    assert all(error < 1e-8 for error in errors)
