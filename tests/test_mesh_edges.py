"""Tests for element edge extraction and boundary-edge detection (femtoolkit.mesh.edges)."""

import pytest

from femtoolkit.materials import LinearElastic2D
from femtoolkit.mesh import (
    CSTElement2D,
    Mesh,
    Node,
    QuadElement2D,
    create_quad_mesh,
    create_triangular_mesh,
)
from femtoolkit.mesh.edges import element_edges, find_boundary_edges


@pytest.fixture
def material() -> LinearElastic2D:
    return LinearElastic2D(youngs_modulus=200e9, poisson_ratio=0.3, formulation="plane_stress")


def test_element_edges_cst_has_three_edges(material: LinearElastic2D) -> None:
    node_1 = Node(id=1, x=0.0, y=0.0, z=0.0)
    node_2 = Node(id=2, x=1.0, y=0.0, z=0.0)
    node_3 = Node(id=3, x=0.0, y=1.0, z=0.0)
    element = CSTElement2D(id=1, nodes=(node_1, node_2, node_3), material=material, thickness=0.01)

    edges = element_edges(element)

    assert len(edges) == 3
    assert [edge.node_ids for edge in edges] == [(1, 2), (2, 3), (3, 1)]


def test_element_edges_quad_has_four_edges(material: LinearElastic2D) -> None:
    node_1 = Node(id=1, x=0.0, y=0.0, z=0.0)
    node_2 = Node(id=2, x=1.0, y=0.0, z=0.0)
    node_3 = Node(id=3, x=1.0, y=1.0, z=0.0)
    node_4 = Node(id=4, x=0.0, y=1.0, z=0.0)
    element = QuadElement2D(
        id=1, nodes=(node_1, node_2, node_3, node_4), material=material, thickness=0.01
    )

    edges = element_edges(element)

    assert len(edges) == 4
    assert [edge.node_ids for edge in edges] == [(1, 2), (2, 3), (3, 4), (4, 1)]


def test_single_quad_element_all_edges_are_boundary_edges(material: LinearElastic2D) -> None:
    mesh = create_quad_mesh(width=1.0, height=1.0, nx=1, ny=1, material=material, thickness=0.01)

    boundary_edges = find_boundary_edges(mesh)

    assert len(boundary_edges) == 4


def test_two_element_mesh_has_one_shared_interior_edge(material: LinearElastic2D) -> None:
    """Two side-by-side Q4 cells: 8 total edges, one pair coincides (the
    shared internal edge), leaving 6 boundary edges.
    """
    mesh = create_quad_mesh(width=2.0, height=1.0, nx=2, ny=1, material=material, thickness=0.01)

    boundary_edges = find_boundary_edges(mesh)

    assert len(boundary_edges) == 6  # perimeter of a 2x1 grid: 2*(2+1)


def test_boundary_edge_count_for_rectangular_grid(material: LinearElastic2D) -> None:
    nx, ny = 4, 2
    mesh = create_quad_mesh(width=2.0, height=1.0, nx=nx, ny=ny, material=material, thickness=0.01)

    boundary_edges = find_boundary_edges(mesh)

    assert len(boundary_edges) == 2 * (nx + ny)


def test_interior_edge_is_not_a_boundary_edge(material: LinearElastic2D) -> None:
    mesh = create_quad_mesh(width=2.0, height=1.0, nx=2, ny=1, material=material, thickness=0.01)

    boundary_edges = find_boundary_edges(mesh)
    boundary_node_pairs = {frozenset(edge.node_ids) for edge in boundary_edges}

    # The shared vertical edge between the two cells (nodes 2 and 5 in the
    # row-major numbering for nx=2, ny=1) must NOT be a boundary edge.
    assert frozenset({2, 5}) not in boundary_node_pairs


def test_triangular_mesh_boundary_edge_count(material: LinearElastic2D) -> None:
    """Each cell's diagonal is shared by the two triangles that split that
    cell (both triangles reference the same pair of diagonal endpoint
    nodes), so it is an interior edge, not a boundary one -- splitting
    quads into triangles changes nothing about the mesh's outer
    perimeter edge count.
    """
    nx, ny = 4, 2
    mesh = create_triangular_mesh(
        width=2.0, height=1.0, nx=nx, ny=ny, material=material, thickness=0.01
    )

    boundary_edges = find_boundary_edges(mesh)
    assert len(boundary_edges) == 2 * (nx + ny)


def test_non_continuum_elements_are_skipped() -> None:
    from femtoolkit.materials import Material
    from femtoolkit.mesh import Element

    node_1 = Node(id=1, x=0.0, y=0.0, z=0.0)
    node_2 = Node(id=2, x=1.0, y=0.0, z=0.0)
    bar_material = Material(name="Steel", density=7850.0, youngs_modulus=200e9, poissons_ratio=0.3)
    mesh = Mesh()
    mesh.add_node(node_1)
    mesh.add_node(node_2)
    mesh.add_element(Element(id=1, nodes=[node_1, node_2], material=bar_material))

    assert find_boundary_edges(mesh) == []
