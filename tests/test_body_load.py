"""Tests for GravityLoad and gravity_load_to_nodal_loads."""

import pytest
from numpy.testing import assert_allclose

from femtoolkit.analysis.body_load import GravityLoad, gravity_load_to_nodal_loads
from femtoolkit.analysis.dof import TranslationDOF
from femtoolkit.exceptions import ValidationError
from femtoolkit.materials import LinearElastic2D
from femtoolkit.mesh import CSTElement2D, Mesh, Node, QuadElement2D

DENSITY = 1000.0
THICKNESS = 0.01
G = 9.81


@pytest.fixture
def material() -> LinearElastic2D:
    return LinearElastic2D(
        youngs_modulus=200e9, poisson_ratio=0.3, formulation="plane_stress", density=DENSITY
    )


@pytest.fixture
def material_without_density() -> LinearElastic2D:
    return LinearElastic2D(youngs_modulus=200e9, poisson_ratio=0.3, formulation="plane_stress")


@pytest.fixture
def cst_mesh(material: LinearElastic2D) -> Mesh:
    mesh = Mesh()
    mesh.add_node(Node(1, 0.0, 0.0, 0.0))
    mesh.add_node(Node(2, 1.0, 0.0, 0.0))
    mesh.add_node(Node(3, 0.0, 1.0, 0.0))
    mesh.add_element(
        CSTElement2D(id=1, nodes=(mesh.get_node(1), mesh.get_node(2), mesh.get_node(3)),
                      material=material, thickness=THICKNESS)
    )
    return mesh


@pytest.fixture
def quad_mesh(material: LinearElastic2D) -> Mesh:
    mesh = Mesh()
    mesh.add_node(Node(1, 0.0, 0.0, 0.0))
    mesh.add_node(Node(2, 1.0, 0.0, 0.0))
    mesh.add_node(Node(3, 1.0, 1.0, 0.0))
    mesh.add_node(Node(4, 0.0, 1.0, 0.0))
    mesh.add_element(
        QuadElement2D(
            id=1,
            nodes=(mesh.get_node(1), mesh.get_node(2), mesh.get_node(3), mesh.get_node(4)),
            material=material,
            thickness=THICKNESS,
        )
    )
    return mesh


# --- GravityLoad validation ---


def test_gravity_load_default_direction_is_downward() -> None:
    gravity = GravityLoad(g=9.81)

    assert gravity.direction == (0.0, -1.0)
    assert_allclose(gravity.acceleration_vector(), (0.0, -9.81))


def test_gravity_load_rejects_non_positive_g() -> None:
    with pytest.raises(ValidationError):
        GravityLoad(g=0.0)


def test_gravity_load_rejects_non_unit_direction() -> None:
    with pytest.raises(ValidationError):
        GravityLoad(g=9.81, direction=(1.0, 1.0))


# --- gravity_load_to_nodal_loads: CST ---


def test_cst_gravity_splits_weight_evenly_across_three_nodes(cst_mesh: Mesh) -> None:
    loads = gravity_load_to_nodal_loads(cst_mesh, GravityLoad(g=G))

    y_loads = [load.value for load in loads if load.dof == TranslationDOF.Y]
    assert len(y_loads) == 3
    expected_share = -(DENSITY * 0.5 * THICKNESS * G) / 3.0
    for value in y_loads:
        assert_allclose(value, expected_share)


def test_cst_gravity_total_matches_weight_formula(cst_mesh: Mesh) -> None:
    loads = gravity_load_to_nodal_loads(cst_mesh, GravityLoad(g=G))

    total_fy = sum(load.value for load in loads if load.dof == TranslationDOF.Y)
    total_fx = sum(load.value for load in loads if load.dof == TranslationDOF.X)

    assert_allclose(total_fx, 0.0, atol=1e-12)
    assert_allclose(total_fy, -(DENSITY * 0.5 * THICKNESS * G))


def test_cst_gravity_requires_density(material_without_density: LinearElastic2D) -> None:
    mesh = Mesh()
    mesh.add_node(Node(1, 0.0, 0.0, 0.0))
    mesh.add_node(Node(2, 1.0, 0.0, 0.0))
    mesh.add_node(Node(3, 0.0, 1.0, 0.0))
    mesh.add_element(
        CSTElement2D(id=1, nodes=(mesh.get_node(1), mesh.get_node(2), mesh.get_node(3)),
                      material=material_without_density, thickness=THICKNESS)
    )

    with pytest.raises(ValidationError):
        gravity_load_to_nodal_loads(mesh, GravityLoad(g=G))


# --- gravity_load_to_nodal_loads: Q4 ---


def test_quad_gravity_splits_weight_evenly_for_a_square(quad_mesh: Mesh) -> None:
    loads = gravity_load_to_nodal_loads(quad_mesh, GravityLoad(g=G))

    y_loads = [load.value for load in loads if load.dof == TranslationDOF.Y]
    assert len(y_loads) == 4
    expected_share = -(DENSITY * 1.0 * THICKNESS * G) / 4.0
    for value in y_loads:
        assert_allclose(value, expected_share)


def test_quad_gravity_total_matches_weight_formula(quad_mesh: Mesh) -> None:
    loads = gravity_load_to_nodal_loads(quad_mesh, GravityLoad(g=G))

    total_fy = sum(load.value for load in loads if load.dof == TranslationDOF.Y)
    assert_allclose(total_fy, -(DENSITY * 1.0 * THICKNESS * G))


def test_gravity_direction_can_be_horizontal(cst_mesh: Mesh) -> None:
    loads = gravity_load_to_nodal_loads(cst_mesh, GravityLoad(g=G, direction=(1.0, 0.0)))

    total_fx = sum(load.value for load in loads if load.dof == TranslationDOF.X)
    total_fy = sum(load.value for load in loads if load.dof == TranslationDOF.Y)
    assert_allclose(total_fx, DENSITY * 0.5 * THICKNESS * G)
    assert_allclose(total_fy, 0.0, atol=1e-12)


def test_no_contributing_elements_returns_empty_list(material: LinearElastic2D) -> None:
    empty_mesh = Mesh()
    assert gravity_load_to_nodal_loads(empty_mesh, GravityLoad(g=G)) == []
