"""Tests for the BarElement domain object."""

import pytest
from numpy.testing import assert_allclose

from femtoolkit.exceptions import ValidationError
from femtoolkit.materials import Material
from femtoolkit.mesh import BarElement, Node
from femtoolkit.sections import CrossSection


@pytest.fixture
def steel() -> Material:
    return Material(name="Steel", density=7850.0, youngs_modulus=200e9, poissons_ratio=0.3)


@pytest.fixture
def section() -> CrossSection:
    return CrossSection(area=0.01)


@pytest.fixture
def node_pair() -> tuple[Node, Node]:
    return (Node(id=1, x=0.0, y=0.0, z=0.0), Node(id=2, x=2.0, y=0.0, z=0.0))


def test_bar_element_length(
    node_pair: tuple[Node, Node], steel: Material, section: CrossSection
) -> None:
    bar = BarElement(id=1, nodes=node_pair, material=steel, cross_section=section)

    assert bar.length == 2.0


def test_bar_element_length_ignores_y_and_z() -> None:
    n1 = Node(id=1, x=0.0, y=5.0, z=-3.0)
    n2 = Node(id=2, x=3.0, y=-9.0, z=42.0)
    steel = Material(name="Steel", density=7850.0, youngs_modulus=200e9, poissons_ratio=0.3)
    section = CrossSection(area=0.01)

    bar = BarElement(id=1, nodes=(n1, n2), material=steel, cross_section=section)

    assert bar.length == 3.0


def test_zero_length_element_raises() -> None:
    n1 = Node(id=1, x=1.0, y=0.0, z=0.0)
    n2 = Node(id=2, x=1.0, y=0.0, z=0.0)
    steel = Material(name="Steel", density=7850.0, youngs_modulus=200e9, poissons_ratio=0.3)
    section = CrossSection(area=0.01)

    with pytest.raises(ValidationError):
        BarElement(id=1, nodes=(n1, n2), material=steel, cross_section=section)


def test_bar_element_stiffness_matrix(
    node_pair: tuple[Node, Node], steel: Material, section: CrossSection
) -> None:
    bar = BarElement(id=1, nodes=node_pair, material=steel, cross_section=section)
    expected_k = steel.youngs_modulus * section.area / bar.length

    assert_allclose(bar.stiffness_matrix, [[expected_k, -expected_k], [-expected_k, expected_k]])


def test_bar_element_material_and_section(
    node_pair: tuple[Node, Node], steel: Material, section: CrossSection
) -> None:
    bar = BarElement(id=1, nodes=node_pair, material=steel, cross_section=section)

    assert bar.material is steel
    assert bar.cross_section is section


def test_bar_element_strain(
    node_pair: tuple[Node, Node], steel: Material, section: CrossSection
) -> None:
    bar = BarElement(id=1, nodes=node_pair, material=steel, cross_section=section)

    assert_allclose(bar.strain(u1=0.0, u2=0.002), 0.001)


def test_bar_element_stress(
    node_pair: tuple[Node, Node], steel: Material, section: CrossSection
) -> None:
    bar = BarElement(id=1, nodes=node_pair, material=steel, cross_section=section)

    assert_allclose(bar.stress(u1=0.0, u2=0.002), steel.youngs_modulus * 0.001)


def test_bar_element_axial_force(
    node_pair: tuple[Node, Node], steel: Material, section: CrossSection
) -> None:
    bar = BarElement(id=1, nodes=node_pair, material=steel, cross_section=section)

    expected_force = steel.youngs_modulus * 0.001 * section.area
    assert_allclose(bar.axial_force(u1=0.0, u2=0.002), expected_force)


def test_bar_element_tension_sign(
    node_pair: tuple[Node, Node], steel: Material, section: CrossSection
) -> None:
    bar = BarElement(id=1, nodes=node_pair, material=steel, cross_section=section)

    assert bar.strain(u1=0.0, u2=0.001) > 0
    assert bar.stress(u1=0.0, u2=0.001) > 0
    assert bar.axial_force(u1=0.0, u2=0.001) > 0


def test_bar_element_compression_sign(
    node_pair: tuple[Node, Node], steel: Material, section: CrossSection
) -> None:
    bar = BarElement(id=1, nodes=node_pair, material=steel, cross_section=section)

    assert bar.strain(u1=0.0, u2=-0.001) < 0
    assert bar.stress(u1=0.0, u2=-0.001) < 0
    assert bar.axial_force(u1=0.0, u2=-0.001) < 0


def test_invalid_id_raises(
    node_pair: tuple[Node, Node], steel: Material, section: CrossSection
) -> None:
    with pytest.raises(ValidationError):
        BarElement(id=0, nodes=node_pair, material=steel, cross_section=section)


def test_invalid_nodes_raises(steel: Material, section: CrossSection) -> None:
    node = Node(id=1, x=0.0, y=0.0, z=0.0)
    with pytest.raises(ValidationError):
        BarElement(id=1, nodes=(node,), material=steel, cross_section=section)  # type: ignore[arg-type]


def test_duplicate_node_raises(steel: Material, section: CrossSection) -> None:
    node = Node(id=1, x=0.0, y=0.0, z=0.0)
    with pytest.raises(ValidationError):
        BarElement(id=1, nodes=(node, node), material=steel, cross_section=section)


def test_missing_material_raises(node_pair: tuple[Node, Node], section: CrossSection) -> None:
    with pytest.raises(ValidationError):
        BarElement(id=1, nodes=node_pair, material=None, cross_section=section)  # type: ignore[arg-type]


def test_missing_cross_section_raises(node_pair: tuple[Node, Node], steel: Material) -> None:
    with pytest.raises(ValidationError):
        BarElement(id=1, nodes=node_pair, material=steel, cross_section=None)  # type: ignore[arg-type]
