"""Tests for MultiPointConstraint and apply_multi_point_constraints."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from femtoolkit.analysis import TranslationDOF
from femtoolkit.analysis.dof import DOFMap
from femtoolkit.analysis.multi_point_constraint import (
    PENALTY_FACTOR,
    MultiPointConstraint,
    apply_multi_point_constraints,
)
from femtoolkit.exceptions import EntityNotFoundError, ValidationError


def test_constraint_stores_nodes_and_dof() -> None:
    constraint = MultiPointConstraint(node_id_a=1, node_id_b=2, dof=TranslationDOF.X)

    assert constraint.node_id_a == 1
    assert constraint.node_id_b == 2
    assert constraint.dof == TranslationDOF.X


def test_constraint_rejects_same_node() -> None:
    with pytest.raises(ValidationError):
        MultiPointConstraint(node_id_a=1, node_id_b=1, dof=TranslationDOF.X)


def test_constraint_rejects_non_positive_node_id() -> None:
    with pytest.raises(ValidationError):
        MultiPointConstraint(node_id_a=0, node_id_b=2, dof=TranslationDOF.X)


def test_constraint_rejects_invalid_dof() -> None:
    with pytest.raises(ValidationError):
        MultiPointConstraint(node_id_a=1, node_id_b=2, dof=5)


def test_apply_with_no_constraints_returns_same_matrix() -> None:
    dof_map = DOFMap(node_ids=[1, 2], dofs_per_node=1)
    stiffness = np.eye(2)

    result = apply_multi_point_constraints(dof_map, stiffness, [])

    assert result is stiffness


def test_apply_adds_penalty_stiffness_at_expected_entries() -> None:
    dof_map = DOFMap(node_ids=[1, 2], dofs_per_node=1)
    stiffness = np.array([[10.0, 0.0], [0.0, 10.0]])
    constraint = MultiPointConstraint(node_id_a=1, node_id_b=2, dof=TranslationDOF.X)

    augmented = apply_multi_point_constraints(dof_map, stiffness, [constraint])

    penalty = PENALTY_FACTOR * 10.0
    assert_allclose(augmented[0, 0], 10.0 + penalty)
    assert_allclose(augmented[1, 1], 10.0 + penalty)
    assert_allclose(augmented[0, 1], -penalty)
    assert_allclose(augmented[1, 0], -penalty)
    # The original matrix is untouched.
    assert_allclose(stiffness, [[10.0, 0.0], [0.0, 10.0]])


def test_apply_with_unknown_node_raises() -> None:
    dof_map = DOFMap(node_ids=[1, 2], dofs_per_node=1)
    stiffness = np.eye(2)
    constraint = MultiPointConstraint(node_id_a=1, node_id_b=99, dof=TranslationDOF.X)

    with pytest.raises(EntityNotFoundError):
        apply_multi_point_constraints(dof_map, stiffness, [constraint])
