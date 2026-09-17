"""Tests for femtoolkit.mesh.statistics (Version 25)."""

import pytest

from femtoolkit.exceptions import ValidationError
from femtoolkit.materials import LinearElastic2D, LinearElastic3D
from femtoolkit.mesh import Hex8Element3D, Mesh, Node
from femtoolkit.mesh.generator import create_quad_mesh, create_triangular_mesh
from femtoolkit.mesh.statistics import compute_mesh_statistics


@pytest.fixture
def material_2d() -> LinearElastic2D:
    return LinearElastic2D(youngs_modulus=200e9, poisson_ratio=0.3, formulation="plane_stress")


def test_quad_mesh_statistics(material_2d: LinearElastic2D) -> None:
    mesh = create_quad_mesh(width=2.0, height=1.0, nx=4, ny=2, material=material_2d, thickness=0.01)
    stats = compute_mesh_statistics(mesh)

    assert stats.num_nodes == 15
    assert stats.num_elements == 8
    assert stats.element_type_counts == {"QuadElement2D": 8}
    assert stats.dimension == "2D"
    assert stats.bounding_box_min == pytest.approx((0.0, 0.0, 0.0))
    assert stats.bounding_box_max == pytest.approx((2.0, 1.0, 0.0))
    assert stats.characteristic_size is not None
    assert stats.num_boundary_edges == 12  # perimeter of a 4x2 grid: 2*(4+2)


def test_triangular_mesh_statistics(material_2d: LinearElastic2D) -> None:
    mesh = create_triangular_mesh(
        width=2.0, height=1.0, nx=4, ny=2, material=material_2d, thickness=0.01
    )
    stats = compute_mesh_statistics(mesh)
    assert stats.element_type_counts == {"CSTElement2D": 16}
    assert stats.num_boundary_edges == 12


def test_3d_mesh_statistics() -> None:
    material = LinearElastic3D(youngs_modulus=200e9, poisson_ratio=0.3, density=7850.0)
    coords = [
        (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0), (0.0, 1.0, 1.0),
    ]
    nodes = tuple(Node(id=i + 1, x=x, y=y, z=z) for i, (x, y, z) in enumerate(coords))
    hexa = Hex8Element3D(id=1, nodes=nodes, material=material)
    mesh = Mesh()
    for node in nodes:
        mesh.add_node(node)
    mesh.add_element(hexa)

    stats = compute_mesh_statistics(mesh)
    assert stats.dimension == "3D"
    assert stats.num_boundary_edges is None
    assert stats.element_type_counts == {"Hex8Element3D": 1}
    assert stats.characteristic_size == pytest.approx(3.0**0.5)  # bbox diagonal / 1**(1/3)


def test_statistics_raises_for_empty_mesh() -> None:
    with pytest.raises(ValidationError):
        compute_mesh_statistics(Mesh())
