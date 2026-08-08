"""Tests for LinearSystem representation and the basic linear solver."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from femtoolkit.analysis import (
    BoundaryCondition,
    DOFMap,
    LinearSystem,
    NodalLoad,
    bar_element_stiffness,
    build_force_vector,
    solve,
)
from femtoolkit.exceptions import ValidationError


def test_build_force_vector() -> None:
    dof_map = DOFMap(node_ids=[1, 2], dofs_per_node=1)

    forces = build_force_vector(dof_map, [NodalLoad(node_id=2, dof=0, value=1000.0)])

    assert_allclose(forces, [0.0, 1000.0])


def test_linear_system_rejects_wrong_stiffness_shape() -> None:
    dof_map = DOFMap(node_ids=[1, 2], dofs_per_node=1)

    with pytest.raises(ValidationError):
        LinearSystem(
            dof_map=dof_map,
            stiffness=np.zeros((3, 3)),
            forces=np.zeros(2),
            boundary_conditions=[BoundaryCondition(node_id=1, dof=0, value=0.0)],
        )


def test_linear_system_rejects_wrong_forces_shape() -> None:
    dof_map = DOFMap(node_ids=[1, 2], dofs_per_node=1)

    with pytest.raises(ValidationError):
        LinearSystem(
            dof_map=dof_map,
            stiffness=np.zeros((2, 2)),
            forces=np.zeros(3),
            boundary_conditions=[BoundaryCondition(node_id=1, dof=0, value=0.0)],
        )


def test_linear_system_requires_boundary_conditions() -> None:
    dof_map = DOFMap(node_ids=[1, 2], dofs_per_node=1)

    with pytest.raises(ValidationError):
        LinearSystem(
            dof_map=dof_map,
            stiffness=np.zeros((2, 2)),
            forces=np.zeros(2),
            boundary_conditions=[],
        )


def test_linear_system_rejects_duplicate_boundary_conditions_on_same_dof() -> None:
    dof_map = DOFMap(node_ids=[1, 2], dofs_per_node=1)

    with pytest.raises(ValidationError):
        LinearSystem(
            dof_map=dof_map,
            stiffness=np.zeros((2, 2)),
            forces=np.zeros(2),
            boundary_conditions=[
                BoundaryCondition(node_id=1, dof=0, value=0.0),
                BoundaryCondition(node_id=1, dof=0, value=0.1),
            ],
        )


def test_solve_single_bar_matches_analytical_solution() -> None:
    """u = F * L / (E * A) for a fixed-free axial bar under a tip load."""
    youngs_modulus = 200e9
    area = 0.01
    length = 2.0
    applied_force = 1000.0

    dof_map = DOFMap(node_ids=[1, 2], dofs_per_node=1)
    stiffness = bar_element_stiffness(youngs_modulus=youngs_modulus, area=area, length=length)
    forces = build_force_vector(dof_map, [NodalLoad(node_id=2, dof=0, value=applied_force)])
    system = LinearSystem(
        dof_map=dof_map,
        stiffness=stiffness,
        forces=forces,
        boundary_conditions=[BoundaryCondition(node_id=1, dof=0, value=0.0)],
    )

    displacements = solve(system)

    analytical_displacement = applied_force * length / (youngs_modulus * area)
    assert_allclose(displacements[0], 0.0, atol=1e-12)
    assert_allclose(displacements[1], analytical_displacement, rtol=1e-10)


def test_solve_respects_nonzero_prescribed_displacement() -> None:
    dof_map = DOFMap(node_ids=[1, 2], dofs_per_node=1)
    stiffness = bar_element_stiffness(youngs_modulus=200e9, area=0.01, length=2.0)
    forces = build_force_vector(dof_map, [])
    system = LinearSystem(
        dof_map=dof_map,
        stiffness=stiffness,
        forces=forces,
        boundary_conditions=[
            BoundaryCondition(node_id=1, dof=0, value=0.0),
            BoundaryCondition(node_id=2, dof=0, value=0.005),
        ],
    )

    displacements = solve(system)

    assert_allclose(displacements, [0.0, 0.005])
