"""Tests for the 1D bar, 2D truss, and 2D frame element stiffness matrices."""

import math

import numpy as np
import pytest
from numpy.testing import assert_allclose

from femtoolkit.analysis import bar_element_stiffness
from femtoolkit.analysis.stiffness import (
    cst_element_stiffness,
    frame_element_stiffness_2d,
    frame_element_stiffness_local,
    truss_element_stiffness_2d,
)
from femtoolkit.continuum import plane_stress_matrix, triangle_strain_displacement_matrix
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


# --- 2D frame element (Version 5) ---

YOUNGS_MODULUS = 200e9
AREA = 0.01
SECOND_MOMENT_OF_AREA = 8.333e-6
LENGTH = 2.0


def test_frame_local_stiffness_shape_and_symmetry() -> None:
    stiffness = frame_element_stiffness_local(
        youngs_modulus=YOUNGS_MODULUS,
        area=AREA,
        second_moment_of_area=SECOND_MOMENT_OF_AREA,
        length=LENGTH,
    )

    assert stiffness.shape == (6, 6)
    assert_allclose(stiffness, stiffness.T)


def test_frame_local_stiffness_axial_terms_match_bar() -> None:
    """Rows/columns 0 and 3 (axial DOFs) must match the 1D bar formula exactly."""
    stiffness = frame_element_stiffness_local(
        youngs_modulus=YOUNGS_MODULUS,
        area=AREA,
        second_moment_of_area=SECOND_MOMENT_OF_AREA,
        length=LENGTH,
    )
    bar_stiffness = bar_element_stiffness(
        youngs_modulus=YOUNGS_MODULUS, area=AREA, length=LENGTH
    )

    axial_indices = [0, 3]
    assert_allclose(stiffness[np.ix_(axial_indices, axial_indices)], bar_stiffness)


def test_frame_local_stiffness_bending_terms() -> None:
    stiffness = frame_element_stiffness_local(
        youngs_modulus=YOUNGS_MODULUS,
        area=AREA,
        second_moment_of_area=SECOND_MOMENT_OF_AREA,
        length=LENGTH,
    )
    e, i, length = YOUNGS_MODULUS, SECOND_MOMENT_OF_AREA, LENGTH

    assert_allclose(stiffness[1, 1], 12 * e * i / length**3)
    assert_allclose(stiffness[1, 2], 6 * e * i / length**2)
    assert_allclose(stiffness[2, 2], 4 * e * i / length)
    assert_allclose(stiffness[2, 5], 2 * e * i / length)
    assert_allclose(stiffness[4, 4], 12 * e * i / length**3)
    assert_allclose(stiffness[1, 4], -12 * e * i / length**3)


def test_frame_local_stiffness_axial_bending_uncoupled() -> None:
    """Axial DOFs (0, 3) and bending DOFs (1, 2, 4, 5) do not interact."""
    stiffness = frame_element_stiffness_local(
        youngs_modulus=YOUNGS_MODULUS,
        area=AREA,
        second_moment_of_area=SECOND_MOMENT_OF_AREA,
        length=LENGTH,
    )
    axial_indices = [0, 3]
    bending_indices = [1, 2, 4, 5]

    assert_allclose(stiffness[np.ix_(axial_indices, bending_indices)], 0.0)
    assert_allclose(stiffness[np.ix_(bending_indices, axial_indices)], 0.0)


@pytest.mark.parametrize("youngs_modulus", [0.0, -200e9, float("nan"), float("inf")])
def test_frame_local_stiffness_invalid_youngs_modulus_raises(youngs_modulus: float) -> None:
    with pytest.raises(ValidationError):
        frame_element_stiffness_local(
            youngs_modulus=youngs_modulus,
            area=AREA,
            second_moment_of_area=SECOND_MOMENT_OF_AREA,
            length=LENGTH,
        )


@pytest.mark.parametrize("second_moment_of_area", [0.0, -8.333e-6, float("nan"), float("inf")])
def test_frame_local_stiffness_invalid_second_moment_raises(second_moment_of_area: float) -> None:
    with pytest.raises(ValidationError):
        frame_element_stiffness_local(
            youngs_modulus=YOUNGS_MODULUS,
            area=AREA,
            second_moment_of_area=second_moment_of_area,
            length=LENGTH,
        )


def test_frame_global_stiffness_shape_and_symmetry() -> None:
    stiffness = frame_element_stiffness_2d(
        youngs_modulus=YOUNGS_MODULUS,
        area=AREA,
        second_moment_of_area=SECOND_MOMENT_OF_AREA,
        length=LENGTH,
        cos_theta=math.sqrt(2) / 2,
        sin_theta=math.sqrt(2) / 2,
    )

    assert stiffness.shape == (6, 6)
    assert_allclose(stiffness, stiffness.T)


def test_frame_global_stiffness_horizontal_matches_local() -> None:
    """A horizontal frame element (c=1, s=0) has T = identity, so Kg == Kl."""
    local_stiffness = frame_element_stiffness_local(
        youngs_modulus=YOUNGS_MODULUS,
        area=AREA,
        second_moment_of_area=SECOND_MOMENT_OF_AREA,
        length=LENGTH,
    )
    global_stiffness = frame_element_stiffness_2d(
        youngs_modulus=YOUNGS_MODULUS,
        area=AREA,
        second_moment_of_area=SECOND_MOMENT_OF_AREA,
        length=LENGTH,
        cos_theta=1.0,
        sin_theta=0.0,
    )

    assert_allclose(global_stiffness, local_stiffness)


def test_frame_global_stiffness_vertical_swaps_axial_and_bending_rows() -> None:
    """A vertical frame element (c=0, s=1) has its axial stiffness acting
    on global uy (not ux), since the local axial axis now points along Y.
    """
    global_stiffness = frame_element_stiffness_2d(
        youngs_modulus=YOUNGS_MODULUS,
        area=AREA,
        second_moment_of_area=SECOND_MOMENT_OF_AREA,
        length=LENGTH,
        cos_theta=0.0,
        sin_theta=1.0,
    )
    expected_axial = YOUNGS_MODULUS * AREA / LENGTH

    assert_allclose(global_stiffness[1, 1], expected_axial)  # uy1-uy1: axial
    assert_allclose(global_stiffness[0, 0], 12 * YOUNGS_MODULUS * SECOND_MOMENT_OF_AREA / LENGTH**3)


def test_frame_global_stiffness_matches_transformation_formula() -> None:
    """Kg must equal T^T * Kl * T for an arbitrary orientation."""
    from femtoolkit.analysis.transformation import frame_transformation_matrix_2d

    cos_theta, sin_theta = 0.6, 0.8
    local_stiffness = frame_element_stiffness_local(
        youngs_modulus=YOUNGS_MODULUS,
        area=AREA,
        second_moment_of_area=SECOND_MOMENT_OF_AREA,
        length=LENGTH,
    )
    transformation = frame_transformation_matrix_2d(cos_theta, sin_theta)
    expected = transformation.T @ local_stiffness @ transformation

    global_stiffness = frame_element_stiffness_2d(
        youngs_modulus=YOUNGS_MODULUS,
        area=AREA,
        second_moment_of_area=SECOND_MOMENT_OF_AREA,
        length=LENGTH,
        cos_theta=cos_theta,
        sin_theta=sin_theta,
    )
    assert_allclose(global_stiffness, expected)


@pytest.mark.parametrize("length", [0.0, -2.0, float("nan"), float("inf")])
def test_frame_global_stiffness_invalid_length_raises(length: float) -> None:
    with pytest.raises(ValidationError):
        frame_element_stiffness_2d(
            youngs_modulus=YOUNGS_MODULUS,
            area=AREA,
            second_moment_of_area=SECOND_MOMENT_OF_AREA,
            length=length,
            cos_theta=1.0,
            sin_theta=0.0,
        )


# --- 3-node constant strain triangle (CST) element (Version 6) ---

CST_TRIANGLE = (0.0, 0.0, 1.0, 0.0, 0.0, 1.0)


def test_cst_stiffness_shape_and_symmetry() -> None:
    b_matrix = triangle_strain_displacement_matrix(*CST_TRIANGLE)
    d_matrix = plane_stress_matrix(youngs_modulus=200e9, poisson_ratio=0.3)

    stiffness = cst_element_stiffness(
        thickness=0.01, area=0.5, b_matrix=b_matrix, d_matrix=d_matrix
    )

    assert stiffness.shape == (6, 6)
    assert_allclose(stiffness, stiffness.T)


def test_cst_stiffness_matches_direct_formula() -> None:
    b_matrix = triangle_strain_displacement_matrix(*CST_TRIANGLE)
    d_matrix = plane_stress_matrix(youngs_modulus=200e9, poisson_ratio=0.3)
    thickness, area = 0.01, 0.5

    stiffness = cst_element_stiffness(
        thickness=thickness, area=area, b_matrix=b_matrix, d_matrix=d_matrix
    )

    expected = thickness * area * b_matrix.T @ d_matrix @ b_matrix
    assert_allclose(stiffness, expected)


def test_cst_stiffness_scales_linearly_with_thickness() -> None:
    b_matrix = triangle_strain_displacement_matrix(*CST_TRIANGLE)
    d_matrix = plane_stress_matrix(youngs_modulus=200e9, poisson_ratio=0.3)

    k_t1 = cst_element_stiffness(thickness=1.0, area=0.5, b_matrix=b_matrix, d_matrix=d_matrix)
    k_t2 = cst_element_stiffness(thickness=2.0, area=0.5, b_matrix=b_matrix, d_matrix=d_matrix)

    assert_allclose(k_t2, 2.0 * k_t1)


def test_cst_stiffness_scales_linearly_with_youngs_modulus() -> None:
    b_matrix = triangle_strain_displacement_matrix(*CST_TRIANGLE)
    d_e1 = plane_stress_matrix(youngs_modulus=1.0, poisson_ratio=0.3)
    d_e2 = plane_stress_matrix(youngs_modulus=2.0, poisson_ratio=0.3)

    k_e1 = cst_element_stiffness(thickness=1.0, area=0.5, b_matrix=b_matrix, d_matrix=d_e1)
    k_e2 = cst_element_stiffness(thickness=1.0, area=0.5, b_matrix=b_matrix, d_matrix=d_e2)

    assert_allclose(k_e2, 2.0 * k_e1)


def test_cst_stiffness_is_positive_semidefinite() -> None:
    """Before boundary conditions, K must have no negative eigenvalues
    (rigid-body modes give exactly-zero eigenvalues, never negative ones).
    """
    b_matrix = triangle_strain_displacement_matrix(*CST_TRIANGLE)
    d_matrix = plane_stress_matrix(youngs_modulus=200e9, poisson_ratio=0.3)

    stiffness = cst_element_stiffness(
        thickness=0.01, area=0.5, b_matrix=b_matrix, d_matrix=d_matrix
    )

    eigenvalues = np.linalg.eigvalsh(stiffness)
    tolerance = 1e-6 * np.max(np.abs(eigenvalues))
    assert np.all(eigenvalues > -tolerance)


@pytest.mark.parametrize("thickness", [0.0, -0.01, float("nan"), float("inf")])
def test_cst_stiffness_invalid_thickness_raises(thickness: float) -> None:
    b_matrix = triangle_strain_displacement_matrix(*CST_TRIANGLE)
    d_matrix = plane_stress_matrix(youngs_modulus=200e9, poisson_ratio=0.3)

    with pytest.raises(ValidationError):
        cst_element_stiffness(thickness=thickness, area=0.5, b_matrix=b_matrix, d_matrix=d_matrix)


@pytest.mark.parametrize("area", [0.0, -0.5, float("nan"), float("inf")])
def test_cst_stiffness_invalid_area_raises(area: float) -> None:
    b_matrix = triangle_strain_displacement_matrix(*CST_TRIANGLE)
    d_matrix = plane_stress_matrix(youngs_modulus=200e9, poisson_ratio=0.3)

    with pytest.raises(ValidationError):
        cst_element_stiffness(thickness=0.01, area=area, b_matrix=b_matrix, d_matrix=d_matrix)
