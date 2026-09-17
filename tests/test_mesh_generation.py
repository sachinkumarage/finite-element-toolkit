"""Tests for femtoolkit.mesh.generation (Version 25)."""

import pytest

from femtoolkit.geometry import Rectangle
from femtoolkit.materials import LinearElastic2D
from femtoolkit.mesh.generation import (
    MeshGenerator,
    StructuredQuadMeshGenerator,
    StructuredTriangularMeshGenerator,
)
from femtoolkit.mesh.sizing import MeshSizingParameters


@pytest.fixture
def material() -> LinearElastic2D:
    return LinearElastic2D(youngs_modulus=200e9, poisson_ratio=0.3, formulation="plane_stress")


@pytest.fixture
def geometry() -> Rectangle:
    return Rectangle(width=2.0, height=0.4)


@pytest.fixture
def sizing() -> MeshSizingParameters:
    return MeshSizingParameters(target_size=0.2)


def test_structured_quad_generator_satisfies_protocol() -> None:
    generator = StructuredQuadMeshGenerator()
    assert isinstance(generator, MeshGenerator)


def test_structured_triangular_generator_satisfies_protocol() -> None:
    generator = StructuredTriangularMeshGenerator()
    assert isinstance(generator, MeshGenerator)


def test_structured_quad_generator_produces_expected_mesh(
    geometry: Rectangle, sizing: MeshSizingParameters, material: LinearElastic2D
) -> None:
    generator = StructuredQuadMeshGenerator()
    mesh = generator.generate(geometry, sizing, material, thickness=0.02)

    assert len(mesh.nodes) == 11 * 3
    assert len(mesh.elements) == 10 * 2


def test_structured_triangular_generator_produces_expected_mesh(
    geometry: Rectangle, sizing: MeshSizingParameters, material: LinearElastic2D
) -> None:
    generator = StructuredTriangularMeshGenerator()
    mesh = generator.generate(geometry, sizing, material, thickness=0.02)

    assert len(mesh.elements) == 2 * 10 * 2


def test_structured_triangular_generator_respects_diagonal_direction(
    geometry: Rectangle, sizing: MeshSizingParameters, material: LinearElastic2D
) -> None:
    forward = StructuredTriangularMeshGenerator(diagonal="forward")
    backward = StructuredTriangularMeshGenerator(diagonal="backward")

    mesh_forward = forward.generate(geometry, sizing, material, thickness=0.02)
    mesh_backward = backward.generate(geometry, sizing, material, thickness=0.02)

    first_forward = mesh_forward.get_element(1)
    first_backward = mesh_backward.get_element(1)
    assert [node.id for node in first_forward.nodes] != [node.id for node in first_backward.nodes]


def test_generated_mesh_area_matches_domain(
    geometry: Rectangle, sizing: MeshSizingParameters, material: LinearElastic2D
) -> None:
    generator = StructuredQuadMeshGenerator()
    mesh = generator.generate(geometry, sizing, material, thickness=0.02)
    total_area = sum(element.area for element in mesh.elements)
    assert total_area == pytest.approx(geometry.width * geometry.height)
