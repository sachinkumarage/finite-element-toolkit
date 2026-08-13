"""2x2 Gauss-Legendre quadrature for isoparametric quadrilateral elements.

A Q4 element's stiffness integral, ``Ke = integral(B^T D B t) dA``, has no
closed form once mapped through the isoparametric Jacobian (see
:mod:`femtoolkit.continuum.jacobian`) -- unlike the constant-strain
triangle, whose constant ``B`` lets the integral collapse to a single
multiplication by area. Gauss-Legendre quadrature evaluates it instead as
a weighted sum of the integrand at a small number of points, exact for
polynomials up to a known degree.

The 2-point rule per natural-coordinate direction (2x2 = 4 points total)
used here is exact for the bilinear geometric mapping and is the standard
choice for a 4-node quadrilateral -- higher-order (3x3+) quadrature would
be needed only for higher-order elements, out of scope for this version.
"""

from __future__ import annotations

from typing import NamedTuple

_ONE_OVER_SQRT_3 = 1.0 / 3.0**0.5


class GaussPoint(NamedTuple):
    """A single 2D Gauss-Legendre integration point and its weight.

    Attributes:
        xi: First natural coordinate.
        eta: Second natural coordinate.
        weight: Integration weight (the product of the 1D weights along
            each natural-coordinate direction).
    """

    xi: float
    eta: float
    weight: float


GAUSS_2X2_POINTS: tuple[GaussPoint, GaussPoint, GaussPoint, GaussPoint] = (
    GaussPoint(-_ONE_OVER_SQRT_3, -_ONE_OVER_SQRT_3, 1.0),
    GaussPoint(_ONE_OVER_SQRT_3, -_ONE_OVER_SQRT_3, 1.0),
    GaussPoint(_ONE_OVER_SQRT_3, _ONE_OVER_SQRT_3, 1.0),
    GaussPoint(-_ONE_OVER_SQRT_3, _ONE_OVER_SQRT_3, 1.0),
)
"""The four points of the 2x2 Gauss-Legendre rule on ``[-1,1] x [-1,1]``.

Each point uses the standard 2-point 1D Gauss-Legendre abscissa
``+/- 1/sqrt(3)``, whose exact 1D weight is ``1.0``; the 2D weight is the
product of the two 1D weights, ``1.0 * 1.0 = 1.0``, for every point.
"""
