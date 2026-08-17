"""Tests for the BoundaryCondition data model."""

import math

import pytest

from femtoolkit.analysis import BoundaryCondition, TranslationDOF, boundary_conditions_for_region
from femtoolkit.exceptions import ValidationError
from femtoolkit.geometry import Rectangle
from femtoolkit.materials import LinearElastic2D
from femtoolkit.mesh import create_quad_mesh


def test_valid_boundary_condition_creation() -> None:
    boundary_condition = BoundaryCondition(node_id=1, dof=0, value=0.0)

    assert boundary_condition.node_id == 1
    assert boundary_condition.dof == 0
    assert boundary_condition.value == 0.0


def test_boundary_condition_supports_nonzero_prescribed_value() -> None:
    boundary_condition = BoundaryCondition(node_id=2, dof=0, value=-0.001)

    assert boundary_condition.value == -0.001


@pytest.mark.parametrize("node_id", [0, -1, 1.5, "1"])
def test_invalid_node_id_raises(node_id) -> None:
    with pytest.raises(ValidationError):
        BoundaryCondition(node_id=node_id, dof=0, value=0.0)


@pytest.mark.parametrize("dof", [-1, 3, 1.5])
def test_invalid_dof_raises(dof) -> None:
    with pytest.raises(ValidationError):
        BoundaryCondition(node_id=1, dof=dof, value=0.0)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_invalid_value_raises(value: float) -> None:
    with pytest.raises(ValidationError):
        BoundaryCondition(node_id=1, dof=0, value=value)


# --- boundary_conditions_for_region (Version 9) ---


WIDTH = 2.0
HEIGHT = 1.0


def _mesh_and_domain():
    material = LinearElastic2D(youngs_modulus=200e9, poisson_ratio=0.3, formulation="plane_stress")
    mesh = create_quad_mesh(
        width=WIDTH, height=HEIGHT, nx=4, ny=2, material=material, thickness=0.01
    )
    domain = Rectangle(width=WIDTH, height=HEIGHT)
    return mesh, domain


def test_boundary_conditions_for_region_fixes_both_dofs() -> None:
    mesh, domain = _mesh_and_domain()

    conditions = boundary_conditions_for_region(mesh, domain.boundary("left"), ux=0.0, uy=0.0)

    left_node_ids = {n.id for n in mesh.nodes_on_boundary(domain.boundary("left"))}
    assert len(conditions) == 2 * len(left_node_ids)
    for condition in conditions:
        assert condition.node_id in left_node_ids
        assert condition.value == 0.0


def test_boundary_conditions_for_region_single_dof_roller() -> None:
    mesh, domain = _mesh_and_domain()

    conditions = boundary_conditions_for_region(mesh, domain.boundary("right"), uy=0.0)

    right_node_ids = {n.id for n in mesh.nodes_on_boundary(domain.boundary("right"))}
    assert len(conditions) == len(right_node_ids)
    assert all(condition.dof == TranslationDOF.Y for condition in conditions)


def test_boundary_conditions_for_region_no_dofs_gives_empty_list() -> None:
    mesh, domain = _mesh_and_domain()

    conditions = boundary_conditions_for_region(mesh, domain.boundary("top"))

    assert conditions == []


def test_boundary_conditions_for_region_prescribed_nonzero_value() -> None:
    mesh, domain = _mesh_and_domain()

    conditions = boundary_conditions_for_region(mesh, domain.boundary("left"), ux=-0.001)

    assert all(condition.value == -0.001 for condition in conditions)
    assert all(condition.dof == TranslationDOF.X for condition in conditions)
