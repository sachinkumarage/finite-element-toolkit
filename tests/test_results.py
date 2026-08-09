"""Tests for AnalysisResult."""

import pytest
from numpy.testing import assert_allclose

from femtoolkit.analysis import BoundaryCondition, NodalLoad, StaticLinearAnalysis
from femtoolkit.exceptions import EntityNotFoundError
from femtoolkit.materials import Material
from femtoolkit.mesh import BarElement, Mesh, Node
from femtoolkit.sections import CrossSection


@pytest.fixture
def single_bar_result():
    steel = Material(name="Steel", density=7850.0, youngs_modulus=200e9, poissons_ratio=0.3)
    section = CrossSection(area=0.01)
    node_1 = Node(id=1, x=0.0, y=0.0, z=0.0)
    node_2 = Node(id=2, x=2.0, y=0.0, z=0.0)

    mesh = Mesh()
    mesh.add_node(node_1)
    mesh.add_node(node_2)
    mesh.add_element(
        BarElement(id=1, nodes=(node_1, node_2), material=steel, cross_section=section)
    )

    analysis = StaticLinearAnalysis(mesh)
    analysis.add_boundary_condition(BoundaryCondition(node_id=1, dof=0, value=0.0))
    analysis.add_load(NodalLoad(node_id=2, dof=0, value=1000.0))

    return analysis.solve(), steel, section


def test_result_displacement(single_bar_result) -> None:
    result, steel, section = single_bar_result

    assert_allclose(result.displacement(1), 0.0, atol=1e-12)
    assert_allclose(result.displacement(2), 1000.0 * 2.0 / (steel.youngs_modulus * section.area))


def test_result_reaction_matches_equilibrium(single_bar_result) -> None:
    result, _, _ = single_bar_result

    assert_allclose(result.reaction(1), -1000.0)
    assert_allclose(result.reaction(2), 0.0, atol=1e-6)


def test_result_element_strain(single_bar_result) -> None:
    result, steel, section = single_bar_result

    expected_strain = 1000.0 / (steel.youngs_modulus * section.area)
    assert_allclose(result.element_strain(1), expected_strain)


def test_result_element_stress(single_bar_result) -> None:
    result, steel, section = single_bar_result

    expected_stress = 1000.0 / section.area
    assert_allclose(result.element_stress(1), expected_stress)


def test_result_element_axial_force(single_bar_result) -> None:
    result, _, _ = single_bar_result

    assert_allclose(result.element_axial_force(1), 1000.0)


def test_result_displacement_unknown_node_raises(single_bar_result) -> None:
    result, _, _ = single_bar_result

    with pytest.raises(EntityNotFoundError):
        result.displacement(99)


def test_result_element_unknown_id_raises(single_bar_result) -> None:
    result, _, _ = single_bar_result

    with pytest.raises(EntityNotFoundError):
        result.element_stress(99)
