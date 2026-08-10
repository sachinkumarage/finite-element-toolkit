"""Tests for the 1D bar and 2D truss element stiffness matrices."""

import math

import numpy as np
import pytest
from numpy.testing import assert_allclose

from femtoolkit.analysis import bar_element_stiffness
from femtoolkit.analysis.stiffness import truss_element_stiffness_2d
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


def test_truss_stiffness_2d_shape() -> None:
    stiffness = truss_element_stiffness_2d(
        youngs_modulus=200e9, area=0.01, length=2.0, cos_theta=1.0, sin_theta=0.0
    )

    assert stiffness.shape == (4, 4)


def test_truss_stiffness_2d_is_symmetric() -> None:
    stiffness = truss_element_stiffness_2d(
        youngs_modulus=200e9, area=0.01, length=2.0, cos_theta=0.6, sin_theta=0.8
    )

    assert_allclose(stiffness, stiffness.T)


def test_truss_stiffness_2d_horizontal_reduces_to_bar() -> None:
    """A horizontal truss element (c=1, s=0) has no Y coupling, and its
    ux1/ux2 sub-block matches the 1D bar stiffness matrix exactly.
    """
    youngs_modulus, area, length = 200e9, 0.01, 2.0
    truss_stiffness = truss_element_stiffness_2d(
        youngs_modulus=youngs_modulus,
        area=area,
        length=length,
        cos_theta=1.0,
        sin_theta=0.0,
    )
    bar_stiffness = bar_element_stiffness(youngs_modulus=youngs_modulus, area=area, length=length)

    ux_indices = [0, 2]  # ux1, ux2 among [ux1, uy1, ux2, uy2]
    assert_allclose(truss_stiffness[np.ix_(ux_indices, ux_indices)], bar_stiffness)
    assert_allclose(truss_stiffness[[1, 3], :], 0.0, atol=1e-9)  # no Y coupling
    assert_allclose(truss_stiffness[:, [1, 3]], 0.0, atol=1e-9)


def test_truss_stiffness_2d_vertical_couples_only_y() -> None:
    """A vertical truss element (c=0, s=1) has no X coupling."""
    stiffness = truss_element_stiffness_2d(
        youngs_modulus=200e9, area=0.01, length=2.0, cos_theta=0.0, sin_theta=1.0
    )

    assert_allclose(stiffness[[0, 2], :], 0.0, atol=1e-9)  # no X coupling
    assert_allclose(stiffness[:, [0, 2]], 0.0, atol=1e-9)
    uy_indices = [1, 3]
    expected_k = 200e9 * 0.01 / 2.0
    assert_allclose(
        stiffness[np.ix_(uy_indices, uy_indices)],
        [[expected_k, -expected_k], [-expected_k, expected_k]],
    )


def test_truss_stiffness_2d_diagonal_values() -> None:
    """45-degree element: c = s = sqrt(2)/2, so every entry has equal magnitude k/2."""
    youngs_modulus, area, length = 200e9, 0.01, 2.0
    c = s = math.sqrt(2) / 2
    stiffness = truss_element_stiffness_2d(
        youngs_modulus=youngs_modulus, area=area, length=length, cos_theta=c, sin_theta=s
    )

    k = youngs_modulus * area / length
    expected = k * np.array(
        [
            [0.5, 0.5, -0.5, -0.5],
            [0.5, 0.5, -0.5, -0.5],
            [-0.5, -0.5, 0.5, 0.5],
            [-0.5, -0.5, 0.5, 0.5],
        ]
    )
    assert_allclose(stiffness, expected, rtol=1e-10)


@pytest.mark.parametrize("youngs_modulus", [0.0, -200e9, float("nan"), float("inf")])
def test_truss_stiffness_2d_invalid_youngs_modulus_raises(youngs_modulus: float) -> None:
    with pytest.raises(ValidationError):
        truss_element_stiffness_2d(
            youngs_modulus=youngs_modulus, area=0.01, length=2.0, cos_theta=1.0, sin_theta=0.0
        )


@pytest.mark.parametrize("area", [0.0, -0.01, float("nan"), float("inf")])
def test_truss_stiffness_2d_invalid_area_raises(area: float) -> None:
    with pytest.raises(ValidationError):
        truss_element_stiffness_2d(
            youngs_modulus=200e9, area=area, length=2.0, cos_theta=1.0, sin_theta=0.0
        )


@pytest.mark.parametrize("length", [0.0, -2.0, float("nan"), float("inf")])
def test_truss_stiffness_2d_invalid_length_raises(length: float) -> None:
    with pytest.raises(ValidationError):
        truss_element_stiffness_2d(
            youngs_modulus=200e9, area=0.01, length=length, cos_theta=1.0, sin_theta=0.0
        )
