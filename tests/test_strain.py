"""Tests for the CST strain-displacement matrix (femtoolkit.continuum.strain)."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from femtoolkit.continuum.strain import (
    strain_from_displacements,
    triangle_strain_displacement_matrix,
)
from femtoolkit.exceptions import DegenerateElementError

TRIANGLE = (0.0, 0.0, 1.0, 0.0, 0.0, 1.0)  # node1=(0,0), node2=(1,0), node3=(0,1)


def test_b_matrix_shape() -> None:
    b_matrix = triangle_strain_displacement_matrix(*TRIANGLE)

    assert b_matrix.shape == (3, 6)


def test_b_matrix_expected_values() -> None:
    """Hand-derived: b1=-1, b2=1, b3=0; c1=-1, c2=0, c3=1; 2A=1."""
    b_matrix = triangle_strain_displacement_matrix(*TRIANGLE)

    expected = np.array(
        [
            [-1.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0, 0.0, 1.0],
            [-1.0, -1.0, 0.0, 1.0, 1.0, 0.0],
        ]
    )
    assert_allclose(b_matrix, expected)


def test_b_matrix_is_constant_regardless_of_evaluation_point() -> None:
    """The defining CST property: B does not depend on (x, y) at all --
    it is purely a function of the node coordinates.
    """
    b_matrix = triangle_strain_displacement_matrix(*TRIANGLE)

    # B has no (x, y) parameters to vary in the first place; this test
    # documents that fact by checking the same call is always identical.
    b_matrix_again = triangle_strain_displacement_matrix(*TRIANGLE)
    assert_allclose(b_matrix, b_matrix_again)


def test_b_matrix_same_for_clockwise_and_counter_clockwise_input() -> None:
    """Reversing node winding must not change the recovered B matrix's
    physical meaning: strain computed with either ordering must agree,
    per the orientation policy documented in femtoolkit.continuum.geometry.
    """
    ccw = (0.0, 0.0, 1.0, 0.0, 0.0, 1.0)
    cw = (0.0, 0.0, 0.0, 1.0, 1.0, 0.0)  # node2 and node3 swapped

    b_ccw = triangle_strain_displacement_matrix(*ccw)
    b_cw = triangle_strain_displacement_matrix(*cw)

    # Displacements for the same physical field, ux=1 at node2 (ccw) /
    # node3 (cw) at position (1,0), and 0 elsewhere, produce the same
    # physical strain regardless of listing order.
    d_ccw = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    d_cw = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    assert_allclose(b_ccw @ d_ccw, b_cw @ d_cw)


def test_strain_from_displacements_matches_direct_matrix_multiplication() -> None:
    b_matrix = triangle_strain_displacement_matrix(*TRIANGLE)
    displacements = [0.0, 0.0, 0.002, 0.0, 0.0, 0.001]

    strain = strain_from_displacements(b_matrix, displacements)

    assert_allclose(strain, b_matrix @ np.array(displacements))


def test_constant_strain_field_is_recovered_exactly() -> None:
    """A prescribed linear displacement field u=ax+by, v=cx+dy must
    produce the exact constant strain [a, d, b+c] -- the fundamental CST
    validation (see tests/validation/test_cst_patch.py for the full case).
    """
    a, b, c, d = 0.01, 0.02, -0.005, 0.015
    x1, y1, x2, y2, x3, y3 = TRIANGLE

    def u(x: float, y: float) -> float:
        return a * x + b * y

    def v(x: float, y: float) -> float:
        return c * x + d * y

    displacements = [
        u(x1, y1), v(x1, y1),
        u(x2, y2), v(x2, y2),
        u(x3, y3), v(x3, y3),
    ]

    b_matrix = triangle_strain_displacement_matrix(*TRIANGLE)
    strain = strain_from_displacements(b_matrix, displacements)

    assert_allclose(strain, [a, d, b + c], atol=1e-12)


@pytest.mark.parametrize(
    "triangle",
    [
        (0.0, 0.0, 1.0, 1.0, 2.0, 2.0),  # collinear
        (0.0, 0.0, 1.0, 0.0, 0.5, 1e-14),  # nearly collinear
        (1.0, 1.0, 1.0, 1.0, 2.0, 2.0),  # duplicate node
    ],
)
def test_degenerate_triangle_raises(triangle: tuple[float, ...]) -> None:
    with pytest.raises(DegenerateElementError):
        triangle_strain_displacement_matrix(*triangle)
