"""Tests for femtoolkit.analysis.mass: element mass matrix dispatch."""

import pytest
from numpy.testing import assert_allclose

from femtoolkit.analysis.mass import element_mass_matrix, element_total_mass
from femtoolkit.exceptions import ValidationError
from femtoolkit.materials import LinearElastic2D
from femtoolkit.mesh import CSTElement2D, Mesh, Node, QuadElement2D

DENSITY = 1000.0
THICKNESS = 0.01


@pytest.fixture
def material() -> LinearElastic2D:
    return LinearElastic2D(
        youngs_modulus=200e9, poisson_ratio=0.3, formulation="plane_stress", density=DENSITY
    )


@pytest.fixture
def material_without_density() -> LinearElastic2D:
    return LinearElastic2D(youngs_modulus=200e9, poisson_ratio=0.3, formulation="plane_stress")


@pytest.fixture
def cst_element(material: LinearElastic2D) -> CSTElement2D:
    mesh = Mesh()
    mesh.add_node(Node(1, 0.0, 0.0, 0.0))
    mesh.add_node(Node(2, 1.0, 0.0, 0.0))
    mesh.add_node(Node(3, 0.0, 1.0, 0.0))
    element = CSTElement2D(
        id=1, nodes=(mesh.get_node(1), mesh.get_node(2), mesh.get_node(3)),
        material=material, thickness=THICKNESS,
    )
    mesh.add_element(element)
    return element


@pytest.fixture
def quad_element(material: LinearElastic2D) -> QuadElement2D:
    mesh = Mesh()
    mesh.add_node(Node(1, 0.0, 0.0, 0.0))
    mesh.add_node(Node(2, 1.0, 0.0, 0.0))
    mesh.add_node(Node(3, 1.0, 1.0, 0.0))
    mesh.add_node(Node(4, 0.0, 1.0, 0.0))
    element = QuadElement2D(
        id=1,
        nodes=(mesh.get_node(1), mesh.get_node(2), mesh.get_node(3), mesh.get_node(4)),
        material=material,
        thickness=THICKNESS,
    )
    mesh.add_element(element)
    return element


def test_element_total_mass_cst(cst_element: CSTElement2D) -> None:
    assert_allclose(element_total_mass(cst_element), DENSITY * 0.5 * THICKNESS)


def test_element_total_mass_quad(quad_element: QuadElement2D) -> None:
    assert_allclose(element_total_mass(quad_element), DENSITY * 1.0 * THICKNESS)


def test_element_mass_matrix_consistent_cst_shape(cst_element: CSTElement2D) -> None:
    m = element_mass_matrix(cst_element, "consistent")
    assert m.shape == (6, 6)


def test_element_mass_matrix_lumped_cst_is_diagonal(cst_element: CSTElement2D) -> None:
    import numpy as np

    m = element_mass_matrix(cst_element, "lumped")
    off_diagonal = m - np.diag(np.diag(m))
    assert_allclose(off_diagonal, np.zeros_like(m))


def test_element_mass_matrix_defaults_to_consistent(cst_element: CSTElement2D) -> None:
    default = element_mass_matrix(cst_element)
    explicit = element_mass_matrix(cst_element, "consistent")
    assert_allclose(default, explicit)


def test_element_mass_matrix_consistent_quad_shape(quad_element: QuadElement2D) -> None:
    m = element_mass_matrix(quad_element, "consistent")
    assert m.shape == (8, 8)


def test_element_mass_matrix_lumped_quad_shape(quad_element: QuadElement2D) -> None:
    m = element_mass_matrix(quad_element, "lumped")
    assert m.shape == (8, 8)


def test_element_mass_matrix_rejects_invalid_type(cst_element: CSTElement2D) -> None:
    with pytest.raises(ValidationError):
        element_mass_matrix(cst_element, "bogus")


def test_element_mass_matrix_requires_density(material_without_density: LinearElastic2D) -> None:
    mesh = Mesh()
    mesh.add_node(Node(1, 0.0, 0.0, 0.0))
    mesh.add_node(Node(2, 1.0, 0.0, 0.0))
    mesh.add_node(Node(3, 0.0, 1.0, 0.0))
    element = CSTElement2D(
        id=1, nodes=(mesh.get_node(1), mesh.get_node(2), mesh.get_node(3)),
        material=material_without_density, thickness=THICKNESS,
    )

    with pytest.raises(ValidationError):
        element_mass_matrix(element)

    with pytest.raises(ValidationError):
        element_total_mass(element)


def test_lumped_total_mass_matches_consistent_total_mass_cst(cst_element: CSTElement2D) -> None:
    import numpy as np

    consistent = element_mass_matrix(cst_element, "consistent")
    lumped = element_mass_matrix(cst_element, "lumped")
    x_indices = [0, 2, 4]
    assert_allclose(
        np.diag(lumped)[x_indices].sum(),
        consistent[np.ix_(x_indices, x_indices)].sum(),
    )
