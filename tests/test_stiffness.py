"""Tests for the 1D bar element stiffness matrix."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from femtoolkit.analysis import bar_element_stiffness
from femtoolkit.exceptions import ValidationError


def test_bar_element_stiffness_shape() -> None:
    stiffness = bar_element_stiffness(youngs_modulus=200e9, area=0.01, length=2.0)

    assert stiffness.shape == (2, 2)


def test_bar_element_stiffness_values() -> None:
    youngs_modulus = 200e9
    area = 0.01
    length = 2.0
    expected_k = youngs_modulus * area / length

    stiffness = bar_element_stiffness(youngs_modulus=youngs_modulus, area=area, length=length)

    assert_allclose(stiffness[0, 0], expected_k)
    assert_allclose(stiffness[0, 1], -expected_k)
    assert_allclose(stiffness[1, 0], -expected_k)
    assert_allclose(stiffness[1, 1], expected_k)


def test_bar_element_stiffness_is_symmetric() -> None:
    stiffness = bar_element_stiffness(youngs_modulus=200e9, area=0.01, length=2.0)

    assert_allclose(stiffness, stiffness.T)


def test_bar_element_stiffness_row_sums_to_zero() -> None:
    """Each row must sum to zero: rigid-body translation produces no force."""
    stiffness = bar_element_stiffness(youngs_modulus=200e9, area=0.01, length=2.0)

    assert_allclose(np.sum(stiffness, axis=1), [0.0, 0.0], atol=1e-6)


@pytest.mark.parametrize("youngs_modulus", [0.0, -200e9, float("nan"), float("inf")])
def test_invalid_youngs_modulus_raises(youngs_modulus: float) -> None:
    with pytest.raises(ValidationError):
        bar_element_stiffness(youngs_modulus=youngs_modulus, area=0.01, length=2.0)


@pytest.mark.parametrize("area", [0.0, -0.01, float("nan"), float("inf")])
def test_invalid_area_raises(area: float) -> None:
    with pytest.raises(ValidationError):
        bar_element_stiffness(youngs_modulus=200e9, area=area, length=2.0)


@pytest.mark.parametrize("length", [0.0, -2.0, float("nan"), float("inf")])
def test_invalid_length_raises(length: float) -> None:
    with pytest.raises(ValidationError):
        bar_element_stiffness(youngs_modulus=200e9, area=0.01, length=length)
