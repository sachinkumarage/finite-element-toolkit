"""Tests for the isoparametric Jacobian (femtoolkit.continuum.jacobian)."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from femtoolkit.continuum.jacobian import (
    inverse_jacobian,
    jacobian_determinant,
    jacobian_matrix,
    physical_shape_function_derivatives,
)
from femtoolkit.continuum.shape_functions import quad_shape_function_derivatives
from femtoolkit.exceptions import DegenerateElementError

UNIT_SQUARE_X = (0.0, 1.0, 1.0, 0.0)
UNIT_SQUARE_Y = (0.0, 0.0, 1.0, 1.0)


def test_jacobian_matrix_shape() -> None:
    dn_dxi, dn_deta = quad_shape_function_derivatives(0.0, 0.0)
    jacobian = jacobian_matrix(dn_dxi, dn_deta, UNIT_SQUARE_X, UNIT_SQUARE_Y)

    assert jacobian.shape == (2, 2)


def test_jacobian_matrix_for_unit_square_at_center() -> None:
    """Mapping xi,eta in [-1,1] to x,y in [0,1] scales each direction by
    1/2, so J = [[0.5, 0], [0, 0.5]] and det(J) = 0.25 (the ratio of the
    unit square's area, 1, to the natural square's area, 4).
    """
    dn_dxi, dn_deta = quad_shape_function_derivatives(0.0, 0.0)
    jacobian = jacobian_matrix(dn_dxi, dn_deta, UNIT_SQUARE_X, UNIT_SQUARE_Y)

    assert_allclose(jacobian, [[0.5, 0.0], [0.0, 0.5]])
    assert_allclose(jacobian_determinant(jacobian), 0.25)


def test_jacobian_determinant_constant_for_a_parallelogram() -> None:
    """For any element whose edges are straight and parallel (a
    parallelogram, including a rectangle), the isoparametric mapping is
    affine, so det(J) is the same at every point in the element.
    """
    x, y = (0.0, 3.0, 3.0, 0.0), (0.0, 0.0, 2.0, 2.0)
    determinants = []
    for xi, eta in [(-0.9, -0.9), (0.0, 0.0), (0.5, -0.3), (0.99, 0.99)]:
        dn_dxi, dn_deta = quad_shape_function_derivatives(xi, eta)
        jacobian = jacobian_matrix(dn_dxi, dn_deta, x, y)
        determinants.append(jacobian_determinant(jacobian))

    assert_allclose(determinants, determinants[0])
    # Physical area (3*2=6) / natural area (2*2=4) = 1.5.
    assert_allclose(determinants[0], 1.5)


def test_jacobian_determinant_varies_for_a_non_parallelogram() -> None:
    """A general (non-parallelogram) quadrilateral has a det(J) that
    varies linearly over the element -- not constant, unlike a
    parallelogram.
    """
    x, y = (0.0, 2.0, 1.5, 0.0), (0.0, 0.0, 1.0, 1.0)
    determinants = []
    for xi, eta in [(-0.9, -0.9), (0.9, 0.9)]:
        dn_dxi, dn_deta = quad_shape_function_derivatives(xi, eta)
        jacobian = jacobian_matrix(dn_dxi, dn_deta, x, y)
        determinants.append(jacobian_determinant(jacobian))

    assert abs(determinants[0] - determinants[1]) > 1e-9


def test_inverse_jacobian_is_a_true_inverse() -> None:
    dn_dxi, dn_deta = quad_shape_function_derivatives(0.2, -0.3)
    jacobian = jacobian_matrix(dn_dxi, dn_deta, (0.0, 2.0, 2.0, 0.0), (0.0, 0.0, 1.0, 1.0))

    j_inv = inverse_jacobian(jacobian)
    assert_allclose(jacobian @ j_inv, np.eye(2), atol=1e-12)


def test_inverse_jacobian_rejects_degenerate_geometry() -> None:
    """Four collinear nodes give a zero-area mapping: det(J) = 0."""
    dn_dxi, dn_deta = quad_shape_function_derivatives(0.0, 0.0)
    jacobian = jacobian_matrix(dn_dxi, dn_deta, (0.0, 1.0, 2.0, 3.0), (0.0, 0.0, 0.0, 0.0))

    with pytest.raises(DegenerateElementError):
        inverse_jacobian(jacobian)


def test_inverse_jacobian_rejects_negative_determinant() -> None:
    """Clockwise (inverted) node order gives a negative determinant."""
    dn_dxi, dn_deta = quad_shape_function_derivatives(0.0, 0.0)
    # Same unit square, but nodes listed clockwise instead of counter-clockwise.
    jacobian = jacobian_matrix(dn_dxi, dn_deta, (0.0, 0.0, 1.0, 1.0), (0.0, 1.0, 1.0, 0.0))

    assert jacobian_determinant(jacobian) < 0.0
    with pytest.raises(DegenerateElementError):
        inverse_jacobian(jacobian)


def test_physical_shape_function_derivatives_shape_and_values() -> None:
    dn_dxi, dn_deta = quad_shape_function_derivatives(0.0, 0.0)
    dn_dx, dn_dy, det_j = physical_shape_function_derivatives(
        dn_dxi, dn_deta, UNIT_SQUARE_X, UNIT_SQUARE_Y
    )

    assert dn_dx.shape == (4,)
    assert dn_dy.shape == (4,)
    assert_allclose(det_j, 0.25)
    # Inverse Jacobian for the unit square is [[2,0],[0,2]], so physical
    # derivatives are just 2x the natural ones.
    assert_allclose(dn_dx, 2.0 * np.array(dn_dxi))
    assert_allclose(dn_dy, 2.0 * np.array(dn_deta))


def test_physical_shape_function_derivatives_sum_to_zero() -> None:
    """Rigid-body translation must produce zero strain: sum(dNi/dx) = 0
    and sum(dNi/dy) = 0, inherited from the natural-coordinate identity.
    """
    dn_dxi, dn_deta = quad_shape_function_derivatives(0.3, -0.4)
    dn_dx, dn_dy, _ = physical_shape_function_derivatives(
        dn_dxi, dn_deta, (0.0, 2.0, 2.5, 0.0), (0.0, 0.0, 1.5, 1.0)
    )

    assert_allclose(np.sum(dn_dx), 0.0, atol=1e-12)
    assert_allclose(np.sum(dn_dy), 0.0, atol=1e-12)
