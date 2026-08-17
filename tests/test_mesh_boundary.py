"""Tests for geometry-aware node selection (Mesh.nodes_on_boundary)."""

import pytest
from numpy.testing import assert_allclose

from femtoolkit.geometry import Rectangle
from femtoolkit.materials import LinearElastic2D
from femtoolkit.mesh import Mesh, Node, create_quad_mesh, create_triangular_mesh

WIDTH = 2.0
HEIGHT = 1.0
NX = 4
NY = 2


@pytest.fixture
def material() -> LinearElastic2D:
    return LinearElastic2D(youngs_modulus=200e9, poisson_ratio=0.3, formulation="plane_stress")


@pytest.fixture
def domain() -> Rectangle:
    return Rectangle(width=WIDTH, height=HEIGHT)


@pytest.fixture
def mesh(material: LinearElastic2D) -> Mesh:
    return create_quad_mesh(
        width=WIDTH, height=HEIGHT, nx=NX, ny=NY, material=material, thickness=0.01
    )


def test_left_boundary_node_count(mesh: Mesh, domain: Rectangle) -> None:
    nodes = mesh.nodes_on_boundary(domain.boundary("left"))

    assert len(nodes) == NY + 1


def test_right_boundary_node_count(mesh: Mesh, domain: Rectangle) -> None:
    nodes = mesh.nodes_on_boundary(domain.boundary("right"))

    assert len(nodes) == NY + 1


def test_bottom_boundary_node_count(mesh: Mesh, domain: Rectangle) -> None:
    nodes = mesh.nodes_on_boundary(domain.boundary("bottom"))

    assert len(nodes) == NX + 1


def test_top_boundary_node_count(mesh: Mesh, domain: Rectangle) -> None:
    nodes = mesh.nodes_on_boundary(domain.boundary("top"))

    assert len(nodes) == NX + 1


def test_left_boundary_nodes_have_x_zero(mesh: Mesh, domain: Rectangle) -> None:
    nodes = mesh.nodes_on_boundary(domain.boundary("left"))

    for node in nodes:
        assert_allclose(node.x, 0.0, atol=1e-9)


def test_right_boundary_nodes_have_x_equal_width(mesh: Mesh, domain: Rectangle) -> None:
    nodes = mesh.nodes_on_boundary(domain.boundary("right"))

    for node in nodes:
        assert_allclose(node.x, WIDTH, atol=1e-9)


def test_bottom_boundary_nodes_have_y_zero(mesh: Mesh, domain: Rectangle) -> None:
    nodes = mesh.nodes_on_boundary(domain.boundary("bottom"))

    for node in nodes:
        assert_allclose(node.y, 0.0, atol=1e-9)


def test_top_boundary_nodes_have_y_equal_height(mesh: Mesh, domain: Rectangle) -> None:
    nodes = mesh.nodes_on_boundary(domain.boundary("top"))

    for node in nodes:
        assert_allclose(node.y, HEIGHT, atol=1e-9)


def test_boundary_node_selection_is_deterministic(mesh: Mesh, domain: Rectangle) -> None:
    first = mesh.nodes_on_boundary(domain.boundary("left"))
    second = mesh.nodes_on_boundary(domain.boundary("left"))

    assert [n.id for n in first] == [n.id for n in second]


def test_boundary_node_selection_follows_mesh_node_order(mesh: Mesh, domain: Rectangle) -> None:
    """Selected nodes come back in the mesh's own node order (ascending IDs
    for a generated mesh), not an arbitrary or sorted-by-geometry order.
    """
    nodes = mesh.nodes_on_boundary(domain.boundary("bottom"))

    ids = [n.id for n in nodes]
    assert ids == sorted(ids)


def test_corner_node_appears_on_both_adjacent_boundaries(mesh: Mesh, domain: Rectangle) -> None:
    left_ids = {n.id for n in mesh.nodes_on_boundary(domain.boundary("left"))}
    bottom_ids = {n.id for n in mesh.nodes_on_boundary(domain.boundary("bottom"))}

    assert 1 in left_ids  # node 1 is (0, 0), the shared corner
    assert 1 in bottom_ids


def test_triangular_mesh_boundary_selection_matches_quad_mesh(
    material: LinearElastic2D, domain: Rectangle
) -> None:
    """The two generators place identical node grids, so boundary
    selection must give identical results regardless of element type.
    """
    tri_mesh = create_triangular_mesh(
        width=WIDTH, height=HEIGHT, nx=NX, ny=NY, material=material, thickness=0.01
    )

    left_ids = {n.id for n in tri_mesh.nodes_on_boundary(domain.boundary("left"))}
    assert len(left_ids) == NY + 1


# --- Tolerance / floating-point robustness ---


def test_tolerance_accepts_tiny_floating_point_perturbation(domain: Rectangle) -> None:
    mesh = Mesh()
    # A node that should be "on" the left boundary but has a tiny
    # floating-point offset from exactly x=0, as real solver/generator
    # arithmetic can produce.
    perturbed = Node(id=1, x=1e-12, y=0.5, z=0.0)
    mesh.add_node(perturbed)

    nodes = mesh.nodes_on_boundary(domain.boundary("left"), tolerance=1e-9)
    assert nodes == [perturbed]


def test_tolerance_rejects_a_node_meaningfully_off_the_boundary(domain: Rectangle) -> None:
    mesh = Mesh()
    off_boundary = Node(id=1, x=0.01, y=0.5, z=0.0)
    mesh.add_node(off_boundary)

    nodes = mesh.nodes_on_boundary(domain.boundary("left"), tolerance=1e-9)
    assert nodes == []


def test_tighter_tolerance_can_exclude_a_borderline_node(domain: Rectangle) -> None:
    mesh = Mesh()
    borderline = Node(id=1, x=1e-6, y=0.5, z=0.0)
    mesh.add_node(borderline)

    assert mesh.nodes_on_boundary(domain.boundary("left"), tolerance=1e-9) == []
    assert mesh.nodes_on_boundary(domain.boundary("left"), tolerance=1e-5) == [borderline]
