"""Tests for the Mesh container."""

import pytest

from femtoolkit.exceptions import DuplicateIDError, EntityNotFoundError, ValidationError
from femtoolkit.materials import Material
from femtoolkit.mesh import Element, Mesh, Node


@pytest.fixture
def steel() -> Material:
    return Material(name="Steel", density=7850.0, youngs_modulus=200e9, poissons_ratio=0.3)


@pytest.fixture
def node_1() -> Node:
    return Node(id=1, x=0.0, y=0.0, z=0.0)


@pytest.fixture
def node_2() -> Node:
    return Node(id=2, x=1.0, y=0.0, z=0.0)


def test_add_and_get_node(node_1: Node) -> None:
    mesh = Mesh()
    mesh.add_node(node_1)

    assert mesh.get_node(1) is node_1


def test_duplicate_node_id_raises(node_1: Node) -> None:
    mesh = Mesh()
    mesh.add_node(node_1)

    with pytest.raises(DuplicateIDError):
        mesh.add_node(Node(id=1, x=9.0, y=9.0, z=9.0))


def test_get_missing_node_raises() -> None:
    mesh = Mesh()

    with pytest.raises(EntityNotFoundError):
        mesh.get_node(1)


def test_add_and_get_element(node_1: Node, node_2: Node, steel: Material) -> None:
    mesh = Mesh()
    mesh.add_node(node_1)
    mesh.add_node(node_2)
    element = Element(id=1, nodes=[node_1, node_2], material=steel)
    mesh.add_element(element)

    assert mesh.get_element(1) is element


def test_duplicate_element_id_raises(node_1: Node, node_2: Node, steel: Material) -> None:
    mesh = Mesh()
    mesh.add_node(node_1)
    mesh.add_node(node_2)
    mesh.add_element(Element(id=1, nodes=[node_1, node_2], material=steel))

    with pytest.raises(DuplicateIDError):
        mesh.add_element(Element(id=1, nodes=[node_1, node_2], material=steel))


def test_get_missing_element_raises() -> None:
    mesh = Mesh()

    with pytest.raises(EntityNotFoundError):
        mesh.get_element(1)


def test_element_referencing_unregistered_node_raises(
    node_1: Node, node_2: Node, steel: Material
) -> None:
    mesh = Mesh()
    mesh.add_node(node_1)
    # node_2 was never added to the mesh.
    element = Element(id=1, nodes=[node_1, node_2], material=steel)

    with pytest.raises(ValidationError):
        mesh.add_element(element)


def test_mesh_nodes_and_elements_properties(node_1: Node, node_2: Node, steel: Material) -> None:
    mesh = Mesh()
    mesh.add_node(node_1)
    mesh.add_node(node_2)
    element = Element(id=1, nodes=[node_1, node_2], material=steel)
    mesh.add_element(element)

    assert mesh.nodes == [node_1, node_2]
    assert mesh.elements == [element]
