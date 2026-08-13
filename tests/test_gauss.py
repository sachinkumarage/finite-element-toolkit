"""Tests for the 2x2 Gauss-Legendre quadrature rule (femtoolkit.continuum.gauss)."""

import math

from numpy.testing import assert_allclose

from femtoolkit.continuum.gauss import GAUSS_2X2_POINTS


def test_four_points() -> None:
    assert len(GAUSS_2X2_POINTS) == 4


def test_points_match_specified_coordinates() -> None:
    expected_abscissa = 1.0 / math.sqrt(3.0)
    expected_points = {
        (-expected_abscissa, -expected_abscissa),
        (expected_abscissa, -expected_abscissa),
        (expected_abscissa, expected_abscissa),
        (-expected_abscissa, expected_abscissa),
    }

    actual_points = {(round(p.xi, 12), round(p.eta, 12)) for p in GAUSS_2X2_POINTS}
    expected_rounded = {(round(x, 12), round(y, 12)) for x, y in expected_points}
    assert actual_points == expected_rounded


def test_all_weights_are_one() -> None:
    for point in GAUSS_2X2_POINTS:
        assert_allclose(point.weight, 1.0)


def test_points_lie_within_natural_coordinate_bounds() -> None:
    for point in GAUSS_2X2_POINTS:
        assert -1.0 < point.xi < 1.0
        assert -1.0 < point.eta < 1.0


def test_quadrature_exactly_integrates_a_constant() -> None:
    """Integrating f(xi,eta)=1 over [-1,1]x[-1,1] should give the exact
    area, 4 -- the sum of the weights.
    """
    total = sum(point.weight for point in GAUSS_2X2_POINTS)
    assert_allclose(total, 4.0)


def test_quadrature_exactly_integrates_a_cubic_polynomial() -> None:
    """The 2-point 1D Gauss rule is exact for polynomials up to degree 3;
    the tensor-product 2x2 rule is exact for xi^3, eta^3, and their
    products up to that degree. The exact integral of xi^3 over [-1,1] is
    zero (odd function); check the quadrature reproduces that exactly.
    """

    def f(xi: float, eta: float) -> float:
        return xi**3 + eta**3 + xi * eta

    total = sum(point.weight * f(point.xi, point.eta) for point in GAUSS_2X2_POINTS)
    assert_allclose(total, 0.0, atol=1e-12)
