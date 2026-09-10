"""Validation: multiple and invalid/incompatible thermal boundary conditions (spec section 16).

Reuses the existing, already-tested boundary-condition machinery
(:class:`~femtoolkit.analysis.system.LinearSystem`'s duplicate-DOF
detection, :class:`~femtoolkit.analysis.dof.DOFMap`'s unknown-node
detection) rather than reimplementing validation -- this file confirms
that reuse actually surfaces the right errors for thermal analyses.
"""

import pytest

from femtoolkit.exceptions import EntityNotFoundError, ValidationError
from femtoolkit.materials import Material
from femtoolkit.mesh import Node
from femtoolkit.mesh.bar_element import BarElement
from femtoolkit.mesh.mesh import Mesh
from femtoolkit.sections import CrossSection
from femtoolkit.thermal import (
    PrescribedHeatFlux,
    PrescribedTemperature,
    SteadyStateThermalAnalysis,
    ThermalMaterial,
)

_T0 = 373.15
_T_L = 293.15


def _three_node_bar_mesh() -> tuple[Mesh, dict[int, ThermalMaterial], list[Node]]:
    mechanical_material = Material(
        name="steel", density=7850.0, youngs_modulus=200e9, poissons_ratio=0.3
    )
    thermal_material = ThermalMaterial(
        thermal_conductivity=50.0, density=7850.0, specific_heat=460.0
    )
    nodes = [
        Node(id=1, x=0.0, y=0.0, z=0.0),
        Node(id=2, x=1.0, y=0.0, z=0.0),
        Node(id=3, x=2.0, y=0.0, z=0.0),
    ]
    mesh = Mesh()
    for node in nodes:
        mesh.add_node(node)
    bar1 = BarElement(
        id=1, nodes=(nodes[0], nodes[1]), material=mechanical_material,
        cross_section=CrossSection(area=0.01),
    )
    bar2 = BarElement(
        id=2, nodes=(nodes[1], nodes[2]), material=mechanical_material,
        cross_section=CrossSection(area=0.01),
    )
    mesh.add_element(bar1)
    mesh.add_element(bar2)
    return mesh, {1: thermal_material, 2: thermal_material}, nodes


def test_multiple_prescribed_temperatures_all_applied() -> None:
    mesh, materials, nodes = _three_node_bar_mesh()
    analysis = SteadyStateThermalAnalysis(mesh, materials)
    analysis.add_boundary_condition(PrescribedTemperature(nodes[0].id, _T0))
    analysis.add_boundary_condition(PrescribedTemperature(nodes[2].id, _T_L))
    result = analysis.solve()
    assert result.node_temperature(nodes[0].id) == pytest.approx(_T0)
    assert result.node_temperature(nodes[2].id) == pytest.approx(_T_L)


def test_prescribed_temperature_and_prescribed_flux_combined() -> None:
    """One end fixed at a temperature, a known heat flux prescribed at the other."""
    mesh, materials, nodes = _three_node_bar_mesh()
    analysis = SteadyStateThermalAnalysis(mesh, materials)
    analysis.add_boundary_condition(PrescribedTemperature(nodes[0].id, _T0))
    # A flux entering at the free end, in equilibrium with what conduction alone
    # would carry, should reproduce the same linear profile as the pure-Dirichlet case.
    conductivity, area, length = 50.0, 0.01, 2.0
    expected_gradient = (_T_L - _T0) / length
    flux_in = -conductivity * area * expected_gradient  # heat flowing in at node 3 (x=2)
    analysis.add_heat_flux(PrescribedHeatFlux(nodes[2].id, -flux_in))
    result = analysis.solve()
    assert result.node_temperature(nodes[2].id) == pytest.approx(_T_L, rel=1e-6)


def test_conflicting_prescribed_temperatures_on_same_node_raises() -> None:
    mesh, materials, nodes = _three_node_bar_mesh()
    analysis = SteadyStateThermalAnalysis(mesh, materials)
    analysis.add_boundary_condition(PrescribedTemperature(nodes[0].id, 300.0))
    analysis.add_boundary_condition(PrescribedTemperature(nodes[0].id, 310.0))
    analysis.add_boundary_condition(PrescribedTemperature(nodes[2].id, 290.0))
    with pytest.raises(ValidationError):
        analysis.solve()


def test_boundary_condition_referencing_unknown_node_raises() -> None:
    mesh, materials, nodes = _three_node_bar_mesh()
    analysis = SteadyStateThermalAnalysis(mesh, materials)
    analysis.add_boundary_condition(PrescribedTemperature(node_id=999, value=300.0))
    analysis.add_boundary_condition(PrescribedTemperature(nodes[2].id, 290.0))
    with pytest.raises(EntityNotFoundError):
        analysis.solve()


def test_no_boundary_condition_raises() -> None:
    mesh, materials, _nodes = _three_node_bar_mesh()
    analysis = SteadyStateThermalAnalysis(mesh, materials)
    with pytest.raises(ValidationError):
        analysis.solve()


def test_missing_thermal_material_for_an_element_raises_at_construction() -> None:
    mesh, materials, _nodes = _three_node_bar_mesh()
    incomplete_materials = {1: materials[1]}  # element 2 has no material
    with pytest.raises(ValidationError):
        SteadyStateThermalAnalysis(mesh, incomplete_materials)


def test_heat_flux_referencing_unknown_node_raises() -> None:
    mesh, materials, nodes = _three_node_bar_mesh()
    analysis = SteadyStateThermalAnalysis(mesh, materials)
    analysis.add_boundary_condition(PrescribedTemperature(nodes[0].id, _T0))
    analysis.add_boundary_condition(PrescribedTemperature(nodes[2].id, _T_L))
    analysis.add_heat_flux(PrescribedHeatFlux(node_id=999, value=10.0))
    with pytest.raises(EntityNotFoundError):
        analysis.solve()
