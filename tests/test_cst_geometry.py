"""Tests for triangle geometry (femtoolkit.continuum.geometry)."""

from numpy.testing import assert_allclose

from femtoolkit.continuum.geometry import MIN_TRIANGLE_AREA, triangle_signed_area


def test_counter_clockwise_triangle_has_positive_area() -> None:
    area = triangle_signed_area(0.0, 0.0, 1.0, 0.0, 0.0, 1.0)

    assert area > 0.0
    assert_allclose(area, 0.5)


def test_clockwise_triangle_has_negative_area() -> None:
    area = triangle_signed_area(0.0, 0.0, 0.0, 1.0, 1.0, 0.0)

    assert area < 0.0
    assert_allclose(area, -0.5)


def test_clockwise_is_exact_negative_of_counter_clockwise() -> None:
    """Swapping any two nodes reverses orientation and negates the signed area."""
    ccw_area = triangle_signed_area(0.0, 0.0, 1.0, 0.0, 0.0, 1.0)
    cw_area = triangle_signed_area(0.0, 0.0, 0.0, 1.0, 1.0, 0.0)

    assert_allclose(cw_area, -ccw_area)


def test_right_triangle_area() -> None:
    area = triangle_signed_area(0.0, 0.0, 2.0, 0.0, 0.0, 3.0)

    assert_allclose(area, 3.0)


def test_arbitrary_triangle_area() -> None:
    """A triangle with (2, 1), (5, 2), (3, 6) has a hand-computable area."""
    area = triangle_signed_area(2.0, 1.0, 5.0, 2.0, 3.0, 6.0)

    # 2A = 2*(2-6) + 5*(6-1) + 3*(1-2) = -8 + 25 - 3 = 14 -> A = 7
    assert_allclose(area, 7.0)


def test_collinear_nodes_have_zero_area() -> None:
    area = triangle_signed_area(0.0, 0.0, 1.0, 1.0, 2.0, 2.0)

    assert_allclose(area, 0.0, atol=1e-12)


def test_nearly_collinear_nodes_have_near_zero_area() -> None:
    area = triangle_signed_area(0.0, 0.0, 1.0, 0.0, 0.5, 1e-14)

    assert abs(area) < MIN_TRIANGLE_AREA


def test_duplicate_nodes_have_zero_area() -> None:
    area = triangle_signed_area(0.0, 0.0, 1.0, 1.0, 1.0, 1.0)

    assert_allclose(area, 0.0, atol=1e-12)


def test_negative_coordinates() -> None:
    area = triangle_signed_area(-1.0, -1.0, 1.0, -1.0, -1.0, 1.0)

    assert_allclose(area, 2.0)
