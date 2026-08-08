"""Tests for the NodalLoad data model."""

import math

import pytest

from femtoolkit.analysis import NodalLoad
from femtoolkit.exceptions import ValidationError


def test_valid_nodal_load_creation() -> None:
    load = NodalLoad(node_id=2, dof=0, value=1000.0)

    assert load.node_id == 2
    assert load.dof == 0
    assert load.value == 1000.0


def test_nodal_load_supports_negative_value() -> None:
    load = NodalLoad(node_id=2, dof=0, value=-500.0)

    assert load.value == -500.0


@pytest.mark.parametrize("node_id", [0, -1, 1.5, "2"])
def test_invalid_node_id_raises(node_id) -> None:
    with pytest.raises(ValidationError):
        NodalLoad(node_id=node_id, dof=0, value=1000.0)


@pytest.mark.parametrize("dof", [-1, 3, 1.5])
def test_invalid_dof_raises(dof) -> None:
    with pytest.raises(ValidationError):
        NodalLoad(node_id=2, dof=dof, value=1000.0)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_invalid_value_raises(value: float) -> None:
    with pytest.raises(ValidationError):
        NodalLoad(node_id=2, dof=0, value=value)
