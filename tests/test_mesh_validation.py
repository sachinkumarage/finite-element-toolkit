"""Tests for whole-mesh validation (femtoolkit.mesh.validation)."""

import pytest

from femtoolkit.exceptions import DuplicateIDError, DuplicateNodeCoordinatesError
from femtoolkit.materials import LinearElastic2D
from femtoolkit.mesh import CSTElement2D, Mesh, Node, create_quad_mesh, create_triangular_mesh
from femtoolkit.mesh.validation import validate_mesh


@pytest.fixture
def material() -> LinearElastic2D:
    return LinearElastic2D(youngs_modulus=200e9, poisson_ratio=0.3, formulation="plane_stress")


def test_generated_quad_mesh_is_always_valid(material: LinearElastic2D) -> None:
    mesh = create_quad_mesh(width=2.0, height=1.0, nx=4, ny=2, material=material, thickness=0.01)

    validate_mesh(mesh)  # must not raise


def test_generated_triangular_mesh_is_always_valid(material: LinearElastic2D) -> None:
    mesh = create_triangular_mesh(
        width=2.0, height=1.0, nx=4, ny=2, material=material, thickness=0.01
    )

    validate_mesh(mesh)  # must not raise


def test_duplicate_node_coordinates_raises(material: LinearElastic2D) -> None:
    mesh = Mesh()
    mesh.add_node(Node(id=1, x=0.0, y=0.0, z=0.0))
    mesh.add_node(Node(id=2, x=0.0, y=0.0, z=0.0))  # same location, different id

    with pytest.raises(DuplicateNodeCoordinatesError):
        validate_mesh(mesh)


def test_distinct_node_coordinates_do_not_raise() -> None:
    mesh = Mesh()
    mesh.add_node(Node(id=1, x=0.0, y=0.0, z=0.0))
    mesh.add_node(Node(id=2, x=1.0, y=0.0, z=0.0))

    validate_mesh(mesh)  # must not raise


def test_duplicate_node_id_rejected_at_insertion_time() -> None:
    """Mesh.add_node already enforces unique node IDs; validate_mesh does
    not need to duplicate this check.
    """
    mesh = Mesh()
    mesh.add_node(Node(id=1, x=0.0, y=0.0, z=0.0))

    with pytest.raises(DuplicateIDError):
        mesh.add_node(Node(id=1, x=1.0, y=0.0, z=0.0))


def test_duplicate_element_id_rejected_at_insertion_time(material: LinearElastic2D) -> None:
    """Mesh.add_element already enforces unique element IDs."""
    node_1 = Node(id=1, x=0.0, y=0.0, z=0.0)
    node_2 = Node(id=2, x=1.0, y=0.0, z=0.0)
    node_3 = Node(id=3, x=0.0, y=1.0, z=0.0)
    mesh = Mesh()
    mesh.add_node(node_1)
    mesh.add_node(node_2)
    mesh.add_node(node_3)
    mesh.add_element(
        CSTElement2D(id=1, nodes=(node_1, node_2, node_3), material=material, thickness=0.01)
    )

    with pytest.raises(DuplicateIDError):
        mesh.add_element(
            CSTElement2D(id=1, nodes=(node_1, node_2, node_3), material=material, thickness=0.01)
        )


def test_missing_node_reference_rejected_at_insertion_time(material: LinearElastic2D) -> None:
    """Mesh.add_element already rejects elements referencing nodes not in the mesh."""
    node_1 = Node(id=1, x=0.0, y=0.0, z=0.0)
    node_2 = Node(id=2, x=1.0, y=0.0, z=0.0)
    node_3 = Node(id=3, x=0.0, y=1.0, z=0.0)  # never added to the mesh
    mesh = Mesh()
    mesh.add_node(node_1)
    mesh.add_node(node_2)

    from femtoolkit.exceptions import ValidationError

    with pytest.raises(ValidationError):
        mesh.add_element(
            CSTElement2D(id=1, nodes=(node_1, node_2, node_3), material=material, thickness=0.01)
        )


def test_degenerate_element_rejected_at_construction_time(material: LinearElastic2D) -> None:
    """CSTElement2D already rejects collinear/degenerate geometry when constructed."""
    from femtoolkit.exceptions import DegenerateElementError

    node_1 = Node(id=1, x=0.0, y=0.0, z=0.0)
    node_2 = Node(id=2, x=1.0, y=1.0, z=0.0)
    node_3 = Node(id=3, x=2.0, y=2.0, z=0.0)

    with pytest.raises(DegenerateElementError):
        CSTElement2D(id=1, nodes=(node_1, node_2, node_3), material=material, thickness=0.01)


def test_invalid_node_coordinates_rejected_at_construction_time() -> None:
    """Node already rejects non-finite coordinates when constructed."""
    from femtoolkit.exceptions import ValidationError

    with pytest.raises(ValidationError):
        Node(id=1, x=float("nan"), y=0.0, z=0.0)
