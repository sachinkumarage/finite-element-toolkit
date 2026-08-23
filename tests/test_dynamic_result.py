"""Tests for DynamicResult."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from femtoolkit.analysis.dof import DOFMap, TranslationDOF
from femtoolkit.exceptions import ValidationError
from femtoolkit.results.dynamic_result import DynamicResult


@pytest.fixture
def dynamic_result() -> DynamicResult:
    dof_map = DOFMap(node_ids=[1, 2], dofs_per_node=2)
    time = np.array([0.0, 0.1, 0.2])
    n = dof_map.total_dofs
    displacement = np.arange(len(time) * n, dtype=float).reshape(len(time), n)
    velocity = displacement * 10.0
    acceleration = displacement * 100.0
    reaction = displacement * -1.0
    return DynamicResult(
        dof_map=dof_map,
        time=time,
        displacement_history=displacement,
        velocity_history=velocity,
        acceleration_history=acceleration,
        reaction_history=reaction,
    )


def test_validates_history_shapes() -> None:
    dof_map = DOFMap(node_ids=[1, 2], dofs_per_node=1)
    time = np.array([0.0, 0.1])
    with pytest.raises(ValidationError):
        DynamicResult(
            dof_map=dof_map,
            time=time,
            displacement_history=np.zeros((2, 3)),  # wrong number of DOFs
            velocity_history=np.zeros((2, 2)),
            acceleration_history=np.zeros((2, 2)),
            reaction_history=np.zeros((2, 2)),
        )


def test_displacement_returns_time_series(dynamic_result: DynamicResult) -> None:
    series = dynamic_result.displacement(1, TranslationDOF.X)
    assert series.shape == (3,)
    assert_allclose(series, dynamic_result.displacement_history[:, 0])


def test_velocity_returns_time_series(dynamic_result: DynamicResult) -> None:
    series = dynamic_result.velocity(2, TranslationDOF.Y)
    index = dynamic_result.dof_map.global_index(2, TranslationDOF.Y)
    assert_allclose(series, dynamic_result.velocity_history[:, index])


def test_acceleration_returns_time_series(dynamic_result: DynamicResult) -> None:
    series = dynamic_result.acceleration(1, TranslationDOF.Y)
    index = dynamic_result.dof_map.global_index(1, TranslationDOF.Y)
    assert_allclose(series, dynamic_result.acceleration_history[:, index])


def test_reaction_returns_time_series(dynamic_result: DynamicResult) -> None:
    series = dynamic_result.reaction(2, TranslationDOF.X)
    index = dynamic_result.dof_map.global_index(2, TranslationDOF.X)
    assert_allclose(series, dynamic_result.reaction_history[:, index])


def test_node_displacement_returns_all_active_components(dynamic_result: DynamicResult) -> None:
    node_disp = dynamic_result.node_displacement(1)
    assert node_disp.shape == (3, 2)
    assert_allclose(node_disp[:, 0], dynamic_result.displacement(1, TranslationDOF.X))
    assert_allclose(node_disp[:, 1], dynamic_result.displacement(1, TranslationDOF.Y))


def test_node_reaction_returns_all_active_components(dynamic_result: DynamicResult) -> None:
    node_reaction = dynamic_result.node_reaction(2)
    assert node_reaction.shape == (3, 2)
    assert_allclose(node_reaction[:, 0], dynamic_result.reaction(2, TranslationDOF.X))
    assert_allclose(node_reaction[:, 1], dynamic_result.reaction(2, TranslationDOF.Y))
