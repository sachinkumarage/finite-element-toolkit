"""Tests for global stiffness matrix assembly."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from femtoolkit.analysis import (
    DOFMap,
    ElementStiffnessContribution,
    TranslationDOF,
    assemble_global_stiffness,
    bar_element_stiffness,
)
from femtoolkit.exceptions import EntityNotFoundError, ValidationError

X = TranslationDOF.X


def test_assemble_single_bar_element() -> None:
    dof_map = DOFMap(node_ids=[1, 2], dofs_per_node=1)
    local_stiffness = bar_element_stiffness(youngs_modulus=200e9, area=0.01, length=2.0)

    global_stiffness = assemble_global_stiffness(
        dof_map, [ElementStiffnessContribution(((1, X), (2, X)), local_stiffness)]
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
            ElementStiffnessContribution(((1, X), (2, X)), k1),
            ElementStiffnessContribution(((2, X), (3, X)), k2),
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
            ElementStiffnessContribution(((1, X), (2, X)), k1),
            ElementStiffnessContribution(((2, X), (3, X)), k2),
        ],
    )

    assert_allclose(global_stiffness, global_stiffness.T)


def test_assemble_four_dof_element() -> None:
    """A 4x4 contribution (e.g. a 2D truss element) assembles correctly too."""
    dof_map = DOFMap(node_ids=[1, 2], dofs_per_node=2)
    local_stiffness = np.arange(16, dtype=float).reshape(4, 4)
    local_stiffness = local_stiffness + local_stiffness.T  # make it symmetric

    global_stiffness = assemble_global_stiffness(
        dof_map,
        [
            ElementStiffnessContribution(
                (
                    (1, TranslationDOF.X),
                    (1, TranslationDOF.Y),
                    (2, TranslationDOF.X),
                    (2, TranslationDOF.Y),
                ),
                local_stiffness,
            )
        ],
    )

    assert global_stiffness.shape == (4, 4)
    assert_allclose(global_stiffness, local_stiffness)


def test_assemble_rejects_wrong_shaped_matrix() -> None:
    dof_map = DOFMap(node_ids=[1, 2], dofs_per_node=1)
    bad_matrix = np.zeros((3, 3))

    with pytest.raises(ValidationError):
        assemble_global_stiffness(
            dof_map, [ElementStiffnessContribution(((1, X), (2, X)), bad_matrix)]
        )


def test_assemble_rejects_inactive_dof() -> None:
    """A dof_map with dofs_per_node=1 only activates X; Y is out of range."""
    dof_map = DOFMap(node_ids=[1, 2], dofs_per_node=1)
    local_stiffness = bar_element_stiffness(youngs_modulus=200e9, area=0.01, length=2.0)

    with pytest.raises(ValidationError):
        assemble_global_stiffness(
            dof_map,
            [ElementStiffnessContribution(((1, TranslationDOF.Y), (2, X)), local_stiffness)],
        )


def test_assemble_rejects_unknown_node() -> None:
    dof_map = DOFMap(node_ids=[1, 2], dofs_per_node=1)
    local_stiffness = bar_element_stiffness(youngs_modulus=200e9, area=0.01, length=2.0)

    with pytest.raises(EntityNotFoundError):
        assemble_global_stiffness(
            dof_map, [ElementStiffnessContribution(((1, X), (99, X)), local_stiffness)]
        )
