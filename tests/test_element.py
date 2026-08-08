"""Tests for the Element data model."""

import pytest

from femtoolkit.exceptions import ValidationError
from femtoolkit.materials import Material
from femtoolkit.mesh import Element, Node


@pytest.fixture
def steel() -> Material:
    return Material(name="Steel", density=7850.0, youngs_modulus=200e9, poissons_ratio=0.3)


@pytest.fixture
def node_pair() -> list[Node]:
    return [Node(id=1, x=0.0, y=0.0, z=0.0), Node(id=2, x=1.0, y=0.0, z=0.0)]


def test_valid_element_creation(node_pair: list[Node], steel: Material) -> None:
    element = Element(id=1, nodes=node_pair, material=steel)

    assert element.id == 1
    assert element.nodes == node_pair
    assert element.material == steel


def test_element_node_association(node_pair: list[Node], steel: Material) -> None:
    element = Element(id=1, nodes=node_pair, material=steel)

    assert element.nodes[0].id == 1
    assert element.nodes[1].id == 2


def test_element_material_association(node_pair: list[Node], steel: Material) -> None:
    element = Element(id=1, nodes=node_pair, material=steel)

    assert element.material.name == "Steel"


def test_invalid_id_raises(node_pair: list[Node], steel: Material) -> None:
    with pytest.raises(ValidationError):
        Element(id=0, nodes=node_pair, material=steel)


def test_empty_node_list_raises(steel: Material) -> None:
    with pytest.raises(ValidationError):
        Element(id=1, nodes=[], material=steel)


def test_invalid_node_objects_raise(steel: Material) -> None:
    with pytest.raises(ValidationError):
        Element(id=1, nodes=[Node(id=1, x=0.0, y=0.0, z=0.0), "not-a-node"], material=steel)


def test_missing_material_raises(node_pair: list[Node]) -> None:
    with pytest.raises(ValidationError):
        Element(id=1, nodes=node_pair, material=None)
