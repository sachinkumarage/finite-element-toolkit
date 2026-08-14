"""Tests for structured mesh generation (femtoolkit.mesh.generator)."""

import pytest
from numpy.testing import assert_allclose

from femtoolkit.exceptions import ValidationError
from femtoolkit.materials import LinearElastic2D
from femtoolkit.mesh import CSTElement2D, QuadElement2D, create_quad_mesh, create_triangular_mesh

WIDTH = 2.0
HEIGHT = 1.0
NX = 4
NY = 2


@pytest.fixture
def material() -> LinearElastic2D:
    return LinearElastic2D(youngs_modulus=200e9, poisson_ratio=0.3, formulation="plane_stress")


# --- Rectangular Q4 mesh ---


def test_quad_mesh_node_count(material: LinearElastic2D) -> None:
    mesh = create_quad_mesh(
        width=WIDTH, height=HEIGHT, nx=NX, ny=NY, material=material, thickness=0.01
    )

    assert len(mesh.nodes) == 15  # (4+1)*(2+1)


def test_quad_mesh_element_count(material: LinearElastic2D) -> None:
    mesh = create_quad_mesh(
        width=WIDTH, height=HEIGHT, nx=NX, ny=NY, material=material, thickness=0.01
    )

    assert len(mesh.elements) == 8  # 4*2
    assert all(isinstance(element, QuadElement2D) for element in mesh.elements)


def test_quad_mesh_total_area(material: LinearElastic2D) -> None:
    mesh = create_quad_mesh(
        width=WIDTH, height=HEIGHT, nx=NX, ny=NY, material=material, thickness=0.01
    )

    total_area = sum(element.area for element in mesh.elements)
    assert_allclose(total_area, WIDTH * HEIGHT, rtol=1e-9)


def test_quad_mesh_node_numbering_example_topology(material: LinearElastic2D) -> None:
    """nx=2, ny=1 must give the exact topology documented in the module:
    Node1..3 on the bottom row, Node4..6 on the top row.
    """
    mesh = create_quad_mesh(width=2.0, height=1.0, nx=2, ny=1, material=material, thickness=0.01)

    expected_coordinates = {
        1: (0.0, 0.0),
        2: (1.0, 0.0),
        3: (2.0, 0.0),
        4: (0.0, 1.0),
        5: (1.0, 1.0),
        6: (2.0, 1.0),
    }
    for node in mesh.nodes:
        assert_allclose((node.x, node.y), expected_coordinates[node.id])


def test_quad_mesh_element_connectivity_counter_clockwise(material: LinearElastic2D) -> None:
    mesh = create_quad_mesh(width=2.0, height=1.0, nx=2, ny=1, material=material, thickness=0.01)

    element_1 = mesh.get_element(1)
    assert tuple(node.id for node in element_1.nodes) == (1, 2, 5, 4)


def test_quad_mesh_element_numbering_left_to_right_bottom_to_top(material: LinearElastic2D) -> None:
    mesh = create_quad_mesh(width=4.0, height=2.0, nx=2, ny=2, material=material, thickness=0.01)

    # Row 0 (bottom): elements 1, 2; row 1 (top): elements 3, 4.
    element_1_nodes = {node.id for node in mesh.get_element(1).nodes}
    element_2_nodes = {node.id for node in mesh.get_element(2).nodes}
    element_3_nodes = {node.id for node in mesh.get_element(3).nodes}

    # Element 1 and 2 both lie in the bottom row (y=0 present for all their nodes' min y).
    bottom_row_node_ids = {node.id for node in mesh.nodes if node.y == 0.0}
    assert element_1_nodes & bottom_row_node_ids
    assert element_2_nodes & bottom_row_node_ids
    # Element 3 must be in the top row (shares no exclusively-bottom nodes region).
    top_row_node_ids = {node.id for node in mesh.nodes if node.y == 2.0}
    assert element_3_nodes & top_row_node_ids


def test_quad_mesh_numbering_is_deterministic(material: LinearElastic2D) -> None:
    mesh_a = create_quad_mesh(width=3.0, height=1.5, nx=3, ny=3, material=material, thickness=0.01)
    mesh_b = create_quad_mesh(width=3.0, height=1.5, nx=3, ny=3, material=material, thickness=0.01)

    for node_a, node_b in zip(mesh_a.nodes, mesh_b.nodes, strict=True):
        assert node_a.id == node_b.id
        assert_allclose((node_a.x, node_a.y), (node_b.x, node_b.y))
    for element_a, element_b in zip(mesh_a.elements, mesh_b.elements, strict=True):
        assert element_a.id == element_b.id
        assert [n.id for n in element_a.nodes] == [n.id for n in element_b.nodes]


def test_quad_mesh_all_elements_have_positive_area(material: LinearElastic2D) -> None:
    mesh = create_quad_mesh(
        width=WIDTH, height=HEIGHT, nx=NX, ny=NY, material=material, thickness=0.01
    )

    for element in mesh.elements:
        assert element.area > 0.0


def test_quad_mesh_all_elements_reference_valid_nodes(material: LinearElastic2D) -> None:
    mesh = create_quad_mesh(
        width=WIDTH, height=HEIGHT, nx=NX, ny=NY, material=material, thickness=0.01
    )

    node_ids = {node.id for node in mesh.nodes}
    for element in mesh.elements:
        for node in element.nodes:
            assert node.id in node_ids


# --- Triangular CST mesh ---


def test_triangular_mesh_node_count(material: LinearElastic2D) -> None:
    mesh = create_triangular_mesh(
        width=WIDTH, height=HEIGHT, nx=NX, ny=NY, material=material, thickness=0.01
    )

    assert len(mesh.nodes) == 15


def test_triangular_mesh_element_count(material: LinearElastic2D) -> None:
    mesh = create_triangular_mesh(
        width=WIDTH, height=HEIGHT, nx=NX, ny=NY, material=material, thickness=0.01
    )

    assert len(mesh.elements) == 16  # 2*4*2
    assert all(isinstance(element, CSTElement2D) for element in mesh.elements)


def test_triangular_mesh_total_area(material: LinearElastic2D) -> None:
    mesh = create_triangular_mesh(
        width=WIDTH, height=HEIGHT, nx=NX, ny=NY, material=material, thickness=0.01
    )

    total_area = sum(element.area for element in mesh.elements)
    assert_allclose(total_area, WIDTH * HEIGHT, rtol=1e-9)


def test_triangular_mesh_forward_diagonal_connectivity(material: LinearElastic2D) -> None:
    mesh = create_triangular_mesh(
        width=2.0, height=1.0, nx=2, ny=1, material=material, thickness=0.01, diagonal="forward"
    )

    triangle_1 = mesh.get_element(1)
    triangle_2 = mesh.get_element(2)
    assert tuple(n.id for n in triangle_1.nodes) == (1, 2, 5)
    assert tuple(n.id for n in triangle_2.nodes) == (1, 5, 4)


def test_triangular_mesh_backward_diagonal_connectivity(material: LinearElastic2D) -> None:
    mesh = create_triangular_mesh(
        width=2.0, height=1.0, nx=2, ny=1, material=material, thickness=0.01, diagonal="backward"
    )

    triangle_1 = mesh.get_element(1)
    triangle_2 = mesh.get_element(2)
    assert tuple(n.id for n in triangle_1.nodes) == (1, 2, 4)
    assert tuple(n.id for n in triangle_2.nodes) == (2, 5, 4)


def test_triangular_mesh_invalid_diagonal_raises(material: LinearElastic2D) -> None:
    with pytest.raises(ValidationError):
        create_triangular_mesh(
            width=1.0,
            height=1.0,
            nx=1,
            ny=1,
            material=material,
            thickness=0.01,
            diagonal="sideways",  # type: ignore[arg-type]
        )


def test_triangular_mesh_all_elements_have_positive_area(material: LinearElastic2D) -> None:
    mesh = create_triangular_mesh(
        width=WIDTH, height=HEIGHT, nx=NX, ny=NY, material=material, thickness=0.01
    )

    for element in mesh.elements:
        assert element.area > 0.0
        assert element.signed_area > 0.0  # counter-clockwise, never inverted


def test_triangular_mesh_numbering_is_deterministic(material: LinearElastic2D) -> None:
    mesh_a = create_triangular_mesh(
        width=3.0, height=1.5, nx=3, ny=3, material=material, thickness=0.01
    )
    mesh_b = create_triangular_mesh(
        width=3.0, height=1.5, nx=3, ny=3, material=material, thickness=0.01
    )

    for element_a, element_b in zip(mesh_a.elements, mesh_b.elements, strict=True):
        assert element_a.id == element_b.id
        assert [n.id for n in element_a.nodes] == [n.id for n in element_b.nodes]


# --- Invalid inputs (shared between both generators) ---


@pytest.mark.parametrize("width,height", [(0.0, 1.0), (-1.0, 1.0), (1.0, 0.0), (1.0, -1.0)])
def test_quad_mesh_rejects_non_positive_dimensions(
    material: LinearElastic2D, width: float, height: float
) -> None:
    with pytest.raises(ValidationError):
        create_quad_mesh(width=width, height=height, nx=2, ny=2, material=material, thickness=0.01)


@pytest.mark.parametrize("nx,ny", [(0, 2), (-1, 2), (2, 0), (2, -1)])
def test_quad_mesh_rejects_invalid_subdivisions(
    material: LinearElastic2D, nx: int, ny: int
) -> None:
    with pytest.raises(ValidationError):
        create_quad_mesh(width=1.0, height=1.0, nx=nx, ny=ny, material=material, thickness=0.01)


@pytest.mark.parametrize("width,height", [(0.0, 1.0), (1.0, 0.0), (-1.0, -1.0)])
def test_triangular_mesh_rejects_non_positive_dimensions(
    material: LinearElastic2D, width: float, height: float
) -> None:
    with pytest.raises(ValidationError):
        create_triangular_mesh(
            width=width, height=height, nx=2, ny=2, material=material, thickness=0.01
        )


@pytest.mark.parametrize("nx,ny", [(0, 2), (2, 0), (-2, -2)])
def test_triangular_mesh_rejects_invalid_subdivisions(
    material: LinearElastic2D, nx: int, ny: int
) -> None:
    with pytest.raises(ValidationError):
        create_triangular_mesh(
            width=1.0, height=1.0, nx=nx, ny=ny, material=material, thickness=0.01
        )


def test_quad_mesh_rejects_invalid_thickness(material: LinearElastic2D) -> None:
    with pytest.raises(ValidationError):
        create_quad_mesh(width=1.0, height=1.0, nx=2, ny=2, material=material, thickness=-0.01)


# --- Refinement (increasing nx/ny) ---


def test_finer_mesh_has_more_elements_same_total_area(material: LinearElastic2D) -> None:
    coarse = create_quad_mesh(width=2.0, height=1.0, nx=2, ny=1, material=material, thickness=0.01)
    fine = create_quad_mesh(width=2.0, height=1.0, nx=8, ny=4, material=material, thickness=0.01)

    assert len(fine.elements) > len(coarse.elements)
    assert_allclose(
        sum(e.area for e in fine.elements), sum(e.area for e in coarse.elements), rtol=1e-9
    )
