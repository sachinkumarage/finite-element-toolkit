"""Tests for the degree-of-freedom (DOF) representation."""

import pytest

from femtoolkit.analysis.dof import DOFMap, TranslationDOF, validate_dof
from femtoolkit.exceptions import EntityNotFoundError, ValidationError


def test_translation_dof_values() -> None:
    assert TranslationDOF.X == 0
    assert TranslationDOF.Y == 1
    assert TranslationDOF.Z == 2


@pytest.mark.parametrize("dof", [TranslationDOF.X, TranslationDOF.Y, TranslationDOF.Z, 0, 1, 2])
def test_validate_dof_accepts_valid_values(dof) -> None:
    assert validate_dof(dof) == int(dof)


@pytest.mark.parametrize("dof", [-1, 3, 1.5, "0", None])
def test_validate_dof_rejects_invalid_values(dof) -> None:
    with pytest.raises(ValidationError):
        validate_dof(dof)


def test_dof_map_global_index_single_dof_per_node() -> None:
    dof_map = DOFMap(node_ids=[1, 2, 3], dofs_per_node=1)

    assert dof_map.global_index(node_id=1, dof=TranslationDOF.X) == 0
    assert dof_map.global_index(node_id=2, dof=TranslationDOF.X) == 1
    assert dof_map.global_index(node_id=3, dof=TranslationDOF.X) == 2
    assert dof_map.total_dofs == 3


def test_dof_map_global_index_multiple_dofs_per_node() -> None:
    dof_map = DOFMap(node_ids=[10, 20], dofs_per_node=2)

    assert dof_map.global_index(node_id=10, dof=TranslationDOF.X) == 0
    assert dof_map.global_index(node_id=10, dof=TranslationDOF.Y) == 1
    assert dof_map.global_index(node_id=20, dof=TranslationDOF.X) == 2
    assert dof_map.global_index(node_id=20, dof=TranslationDOF.Y) == 3
    assert dof_map.total_dofs == 4


def test_dof_map_rejects_unknown_node() -> None:
    dof_map = DOFMap(node_ids=[1, 2], dofs_per_node=1)

    with pytest.raises(EntityNotFoundError):
        dof_map.global_index(node_id=99, dof=TranslationDOF.X)


def test_dof_map_rejects_inactive_dof() -> None:
    dof_map = DOFMap(node_ids=[1, 2], dofs_per_node=1)

    with pytest.raises(ValidationError):
        dof_map.global_index(node_id=1, dof=TranslationDOF.Y)


def test_dof_map_rejects_empty_node_ids() -> None:
    with pytest.raises(ValidationError):
        DOFMap(node_ids=[], dofs_per_node=1)


def test_dof_map_rejects_duplicate_node_ids() -> None:
    with pytest.raises(ValidationError):
        DOFMap(node_ids=[1, 1], dofs_per_node=1)


@pytest.mark.parametrize("dofs_per_node", [0, 4, -1])
def test_dof_map_rejects_invalid_dofs_per_node(dofs_per_node: int) -> None:
    with pytest.raises(ValidationError):
        DOFMap(node_ids=[1, 2], dofs_per_node=dofs_per_node)
