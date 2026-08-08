"""Tests for the BoundaryCondition data model."""

import math

import pytest

from femtoolkit.analysis import BoundaryCondition
from femtoolkit.exceptions import ValidationError


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
