"""Tests for element/mesh quality metrics (femtoolkit.mesh.quality)."""

import math

import pytest
from numpy.testing import assert_allclose

from femtoolkit.exceptions import ValidationError
from femtoolkit.materials import LinearElastic2D, Material
from femtoolkit.mesh import (
    CSTElement2D,
    Node,
    QuadElement2D,
    create_quad_mesh,
    create_triangular_mesh,
)
from femtoolkit.mesh.quality import compute_element_quality, compute_mesh_quality_summary


@pytest.fixture
def material() -> LinearElastic2D:
    return LinearElastic2D(youngs_modulus=200e9, poisson_ratio=0.3, formulation="plane_stress")


# --- Q4 element quality ---


def test_unit_square_quad_quality_is_ideal(material: LinearElastic2D) -> None:
    node_1 = Node(id=1, x=0.0, y=0.0, z=0.0)
    node_2 = Node(id=2, x=1.0, y=0.0, z=0.0)
    node_3 = Node(id=3, x=1.0, y=1.0, z=0.0)
    node_4 = Node(id=4, x=0.0, y=1.0, z=0.0)
    element = QuadElement2D(
        id=1, nodes=(node_1, node_2, node_3, node_4), material=material, thickness=0.01
    )

    quality = compute_element_quality(element)

    assert_allclose(quality.area, 1.0)
    assert_allclose(quality.min_edge_length, 1.0)
    assert_allclose(quality.max_edge_length, 1.0)
    assert_allclose(quality.aspect_ratio, 1.0)
    assert_allclose(quality.skewness, 0.0, atol=1e-12)
    assert_allclose(quality.quality, 1.0)
    assert_allclose(quality.jacobian_determinant, 0.25)


def test_elongated_quad_has_high_aspect_ratio_and_low_quality(material: LinearElastic2D) -> None:
    node_1 = Node(id=1, x=0.0, y=0.0, z=0.0)
    node_2 = Node(id=2, x=10.0, y=0.0, z=0.0)
    node_3 = Node(id=3, x=10.0, y=1.0, z=0.0)
    node_4 = Node(id=4, x=0.0, y=1.0, z=0.0)
    element = QuadElement2D(
        id=1, nodes=(node_1, node_2, node_3, node_4), material=material, thickness=0.01
    )

    quality = compute_element_quality(element)

    assert_allclose(quality.aspect_ratio, 10.0)
    assert_allclose(quality.quality, 0.1)
    assert quality.skewness == 0.0  # still rectangular, angles all 90 degrees


def test_quad_element_quality_reports_jacobian_determinant(material: LinearElastic2D) -> None:
    node_1 = Node(id=1, x=0.0, y=0.0, z=0.0)
    node_2 = Node(id=2, x=2.0, y=0.0, z=0.0)
    node_3 = Node(id=3, x=2.0, y=1.0, z=0.0)
    node_4 = Node(id=4, x=0.0, y=1.0, z=0.0)
    element = QuadElement2D(
        id=1, nodes=(node_1, node_2, node_3, node_4), material=material, thickness=0.01
    )

    quality = compute_element_quality(element)

    assert quality.jacobian_determinant is not None
    assert quality.jacobian_determinant > 0.0


# --- CST element quality ---


def test_right_isoceles_triangle_quality(material: LinearElastic2D) -> None:
    node_1 = Node(id=1, x=0.0, y=0.0, z=0.0)
    node_2 = Node(id=2, x=1.0, y=0.0, z=0.0)
    node_3 = Node(id=3, x=0.0, y=1.0, z=0.0)
    element = CSTElement2D(id=1, nodes=(node_1, node_2, node_3), material=material, thickness=0.01)

    quality = compute_element_quality(element)

    assert_allclose(quality.area, 0.5)
    assert_allclose(quality.min_edge_length, 1.0)
    assert_allclose(quality.max_edge_length, math.sqrt(2))
    assert_allclose(quality.aspect_ratio, math.sqrt(2))
    assert_allclose(quality.skewness, 0.25)  # angles 90, 45, 45; ideal 60
    assert quality.jacobian_determinant is None


def test_equilateral_triangle_has_zero_skewness(material: LinearElastic2D) -> None:
    node_1 = Node(id=1, x=0.0, y=0.0, z=0.0)
    node_2 = Node(id=2, x=1.0, y=0.0, z=0.0)
    node_3 = Node(id=3, x=0.5, y=math.sqrt(3) / 2, z=0.0)
    element = CSTElement2D(id=1, nodes=(node_1, node_2, node_3), material=material, thickness=0.01)

    quality = compute_element_quality(element)

    assert_allclose(quality.skewness, 0.0, atol=1e-9)
    assert_allclose(quality.quality, 1.0, atol=1e-9)
    assert_allclose(quality.aspect_ratio, 1.0, atol=1e-9)


def test_compute_element_quality_rejects_unsupported_element_type(
    material: LinearElastic2D,
) -> None:
    from femtoolkit.mesh import Element

    node_1 = Node(id=1, x=0.0, y=0.0, z=0.0)
    node_2 = Node(id=2, x=1.0, y=0.0, z=0.0)
    bar_material = Material(name="Steel", density=7850.0, youngs_modulus=200e9, poissons_ratio=0.3)
    element = Element(id=1, nodes=[node_1, node_2], material=bar_material)

    with pytest.raises(ValidationError):
        compute_element_quality(element)


# --- Mesh-level quality summary ---


def test_regular_quad_mesh_quality_summary(material: LinearElastic2D) -> None:
    mesh = create_quad_mesh(width=2.0, height=1.0, nx=4, ny=2, material=material, thickness=0.01)

    summary = mesh.quality_summary()

    assert summary.num_nodes == 15
    assert summary.num_elements == 8
    assert_allclose(summary.min_area, 0.25, rtol=1e-9)
    assert_allclose(summary.max_area, 0.25, rtol=1e-9)
    assert_allclose(summary.min_quality, 1.0)
    assert_allclose(summary.max_quality, 1.0)
    assert_allclose(summary.average_quality, 1.0)
    assert summary.num_invalid_elements == 0


def test_regular_triangular_mesh_quality_summary(material: LinearElastic2D) -> None:
    mesh = create_triangular_mesh(
        width=2.0, height=1.0, nx=4, ny=2, material=material, thickness=0.01
    )

    summary = mesh.quality_summary()

    assert summary.num_nodes == 15
    assert summary.num_elements == 16
    assert summary.min_quality > 0.0
    assert summary.max_quality <= 1.0
    assert summary.num_invalid_elements == 0


def test_quality_summary_module_function_matches_mesh_method(material: LinearElastic2D) -> None:
    mesh = create_quad_mesh(width=2.0, height=1.0, nx=2, ny=2, material=material, thickness=0.01)

    assert compute_mesh_quality_summary(mesh) == mesh.quality_summary()


def test_element_area_via_mesh(material: LinearElastic2D) -> None:
    mesh = create_quad_mesh(width=2.0, height=1.0, nx=4, ny=2, material=material, thickness=0.01)

    assert_allclose(mesh.element_area(1), 0.25, rtol=1e-9)


def test_element_area_rejects_non_continuum_element() -> None:
    from femtoolkit.mesh import Element, Mesh

    node_1 = Node(id=1, x=0.0, y=0.0, z=0.0)
    node_2 = Node(id=2, x=1.0, y=0.0, z=0.0)
    bar_material = Material(name="Steel", density=7850.0, youngs_modulus=200e9, poissons_ratio=0.3)
    mesh = Mesh()
    mesh.add_node(node_1)
    mesh.add_node(node_2)
    mesh.add_element(Element(id=1, nodes=[node_1, node_2], material=bar_material))

    with pytest.raises(ValidationError):
        mesh.element_area(1)


def test_element_quality_via_mesh(material: LinearElastic2D) -> None:
    mesh = create_quad_mesh(width=2.0, height=1.0, nx=4, ny=2, material=material, thickness=0.01)

    quality = mesh.element_quality(1)
    assert_allclose(quality.area, 0.25, rtol=1e-9)
