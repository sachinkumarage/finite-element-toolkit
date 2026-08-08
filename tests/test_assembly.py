"""Tests for global stiffness matrix assembly."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from femtoolkit.analysis import (
    DOFMap,
    ElementStiffnessContribution,
    assemble_global_stiffness,
    bar_element_stiffness,
)
from femtoolkit.exceptions import EntityNotFoundError, ValidationError


def test_assemble_single_bar_element() -> None:
    dof_map = DOFMap(node_ids=[1, 2], dofs_per_node=1)
    local_stiffness = bar_element_stiffness(youngs_modulus=200e9, area=0.01, length=2.0)

    global_stiffness = assemble_global_stiffness(
        dof_map, [ElementStiffnessContribution((1, 2), local_stiffness)]
    )

    assert global_stiffness.shape == (2, 2)
    assert_allclose(global_stiffness, local_stiffness)


def test_assemble_two_element_chain() -> None:
    """Node 1 ---- Node 2 ---- Node 3, two equal bar elements in series."""
    dof_map = DOFMap(node_ids=[1, 2, 3], dofs_per_node=1)
    k1 = bar_element_stiffness(youngs_modulus=200e9, area=0.01, length=1.0)
    k2 = bar_element_stiffness(youngs_modulus=200e9, area=0.01, length=1.0)
    k = 200e9 * 0.01 / 1.0

    global_stiffness = assemble_global_stiffness(
        dof_map,
        [
            ElementStiffnessContribution((1, 2), k1),
            ElementStiffnessContribution((2, 3), k2),
        ],
    )

    expected = np.array(
        [
            [k, -k, 0.0],
            [-k, 2 * k, -k],
            [0.0, -k, k],
        ]
    )
    assert global_stiffness.shape == (3, 3)
    assert_allclose(global_stiffness, expected)


def test_assemble_two_element_chain_is_symmetric() -> None:
    dof_map = DOFMap(node_ids=[1, 2, 3], dofs_per_node=1)
    k1 = bar_element_stiffness(youngs_modulus=200e9, area=0.01, length=1.0)
    k2 = bar_element_stiffness(youngs_modulus=100e9, area=0.02, length=1.5)

    global_stiffness = assemble_global_stiffness(
        dof_map,
        [
            ElementStiffnessContribution((1, 2), k1),
            ElementStiffnessContribution((2, 3), k2),
        ],
    )

    assert_allclose(global_stiffness, global_stiffness.T)


def test_assemble_rejects_wrong_shaped_matrix() -> None:
    dof_map = DOFMap(node_ids=[1, 2], dofs_per_node=1)
    bad_matrix = np.zeros((3, 3))

    with pytest.raises(ValidationError):
        assemble_global_stiffness(dof_map, [ElementStiffnessContribution((1, 2), bad_matrix)])


def test_assemble_rejects_dof_map_with_multiple_dofs_per_node() -> None:
    dof_map = DOFMap(node_ids=[1, 2], dofs_per_node=2)
    local_stiffness = bar_element_stiffness(youngs_modulus=200e9, area=0.01, length=2.0)

    with pytest.raises(ValidationError):
        assemble_global_stiffness(dof_map, [ElementStiffnessContribution((1, 2), local_stiffness)])


def test_assemble_rejects_unknown_node() -> None:
    dof_map = DOFMap(node_ids=[1, 2], dofs_per_node=1)
    local_stiffness = bar_element_stiffness(youngs_modulus=200e9, area=0.01, length=2.0)

    with pytest.raises(EntityNotFoundError):
        assemble_global_stiffness(dof_map, [ElementStiffnessContribution((1, 99), local_stiffness)])
