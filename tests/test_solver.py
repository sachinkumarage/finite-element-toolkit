"""Tests for the low-level linear solver (femtoolkit.analysis.system.solve).

StaticLinearAnalysis wraps this function's numpy.linalg.LinAlgError into a
domain-specific SingularSystemError (see tests/test_static_linear.py); this
module verifies the underlying numerical behavior it relies on.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from femtoolkit.analysis import BoundaryCondition, DOFMap, LinearSystem, solve


def test_solve_valid_system() -> None:
    dof_map = DOFMap(node_ids=[1, 2], dofs_per_node=1)
    k = 5e9
    stiffness = np.array([[k, -k], [-k, k]])
    forces = np.array([0.0, 1000.0])
    system = LinearSystem(
        dof_map=dof_map,
        stiffness=stiffness,
        forces=forces,
        boundary_conditions=[BoundaryCondition(node_id=1, dof=0, value=0.0)],
    )

    displacements = solve(system)

    assert_allclose(displacements, [0.0, 1000.0 / k])


def test_solve_fully_constrained_system_returns_prescribed_values() -> None:
    dof_map = DOFMap(node_ids=[1, 2], dofs_per_node=1)
    stiffness = np.array([[1.0, -1.0], [-1.0, 1.0]])
    forces = np.zeros(2)
    system = LinearSystem(
        dof_map=dof_map,
        stiffness=stiffness,
        forces=forces,
        boundary_conditions=[
            BoundaryCondition(node_id=1, dof=0, value=0.0),
            BoundaryCondition(node_id=2, dof=0, value=0.01),
        ],
    )

    displacements = solve(system)

    assert_allclose(displacements, [0.0, 0.01])


def test_solve_raises_on_singular_free_free_system() -> None:
    """A three-node chain where only node 1 is constrained, but node 3 has
    zero stiffness coupling: the free-free block is singular.
    """
    dof_map = DOFMap(node_ids=[1, 2, 3], dofs_per_node=1)
    k = 1e9
    # node 3 is disconnected: its row/column is all zeros.
    stiffness = np.array(
        [
            [k, -k, 0.0],
            [-k, k, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    forces = np.array([0.0, 0.0, 1000.0])
    system = LinearSystem(
        dof_map=dof_map,
        stiffness=stiffness,
        forces=forces,
        boundary_conditions=[BoundaryCondition(node_id=1, dof=0, value=0.0)],
    )

    with pytest.raises(np.linalg.LinAlgError):
        solve(system)
