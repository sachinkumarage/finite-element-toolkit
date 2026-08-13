"""Tests for the toolkit's shape functions (femtoolkit.continuum.shape_functions)."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from femtoolkit.continuum.shape_functions import (
    quad_shape_function_derivatives,
    quad_shape_functions,
    triangle_shape_functions,
)
from femtoolkit.exceptions import DegenerateElementError

TRIANGLE = (0.0, 0.0, 1.0, 0.0, 0.0, 1.0)  # node1=(0,0), node2=(1,0), node3=(0,1)
QUAD_CORNERS = ((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0))


def test_partition_of_unity_at_arbitrary_point() -> None:
    n1, n2, n3 = triangle_shape_functions(0.2, 0.3, *TRIANGLE)

    assert_allclose(n1 + n2 + n3, 1.0)


def test_partition_of_unity_at_several_points() -> None:
    for x, y in [(0.1, 0.1), (0.5, 0.4), (0.9, 0.05), (-1.0, 2.0)]:
        n1, n2, n3 = triangle_shape_functions(x, y, *TRIANGLE)
        assert_allclose(n1 + n2 + n3, 1.0)


def test_nodal_property_node_1() -> None:
    n1, n2, n3 = triangle_shape_functions(0.0, 0.0, *TRIANGLE)

    assert_allclose(n1, 1.0)
    assert_allclose(n2, 0.0, atol=1e-12)
    assert_allclose(n3, 0.0, atol=1e-12)


def test_nodal_property_node_2() -> None:
    n1, n2, n3 = triangle_shape_functions(1.0, 0.0, *TRIANGLE)

    assert_allclose(n1, 0.0, atol=1e-12)
    assert_allclose(n2, 1.0)
    assert_allclose(n3, 0.0, atol=1e-12)


def test_nodal_property_node_3() -> None:
    n1, n2, n3 = triangle_shape_functions(0.0, 1.0, *TRIANGLE)

    assert_allclose(n1, 0.0, atol=1e-12)
    assert_allclose(n2, 0.0, atol=1e-12)
    assert_allclose(n3, 1.0)


def test_centroid_gives_equal_weights() -> None:
    """The centroid of a triangle is equidistant (in barycentric terms) from all three nodes."""
    x1, y1, x2, y2, x3, y3 = TRIANGLE
    centroid_x = (x1 + x2 + x3) / 3.0
    centroid_y = (y1 + y2 + y3) / 3.0

    n1, n2, n3 = triangle_shape_functions(centroid_x, centroid_y, *TRIANGLE)

    assert_allclose([n1, n2, n3], [1 / 3, 1 / 3, 1 / 3])


def test_linear_field_is_reproduced_exactly() -> None:
    """For any linear field u(x,y) = a + b*x + c*y, interpolating via shape
    functions from nodal values must reproduce the exact field value.
    """
    a, b, c = 2.0, 3.0, -1.5
    x1, y1, x2, y2, x3, y3 = TRIANGLE

    def field(x: float, y: float) -> float:
        return a + b * x + c * y

    u1, u2, u3 = field(x1, y1), field(x2, y2), field(x3, y3)

    x, y = 0.3, 0.4
    n1, n2, n3 = triangle_shape_functions(x, y, *TRIANGLE)
    interpolated = n1 * u1 + n2 * u2 + n3 * u3

    assert_allclose(interpolated, field(x, y))


def test_degenerate_triangle_raises() -> None:
    with pytest.raises(DegenerateElementError):
        triangle_shape_functions(0.1, 0.1, 0.0, 0.0, 1.0, 1.0, 2.0, 2.0)


# --- Q4 bilinear shape functions (Version 7) ---


def test_quad_partition_of_unity_at_center() -> None:
    n1, n2, n3, n4 = quad_shape_functions(0.0, 0.0)

    assert_allclose(n1 + n2 + n3 + n4, 1.0)
    assert_allclose([n1, n2, n3, n4], [0.25, 0.25, 0.25, 0.25])


@pytest.mark.parametrize(
    "xi,eta",
    [(0.3, 0.2), (-0.7, 0.9), (0.999, -0.999), (0.0, 0.5), (-1.0, -1.0), (1.0, 1.0)],
)
def test_quad_partition_of_unity_at_several_points(xi: float, eta: float) -> None:
    n1, n2, n3, n4 = quad_shape_functions(xi, eta)

    assert_allclose(n1 + n2 + n3 + n4, 1.0)


def test_quad_nodal_property() -> None:
    for expected_index, (xi, eta) in enumerate(QUAD_CORNERS):
        values = quad_shape_functions(xi, eta)
        for actual_index, value in enumerate(values):
            expected = 1.0 if actual_index == expected_index else 0.0
            assert_allclose(value, expected, atol=1e-12)


def test_quad_linear_field_is_reproduced_exactly() -> None:
    """A linear field u = a + b*x + c*y is a special case of bilinear, so a
    Q4 element must reproduce it exactly (the foundation of the patch test).
    """
    a, b, c = 1.0, 2.0, -0.5
    node_coords = [(0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (0.0, 1.0)]

    def field(x: float, y: float) -> float:
        return a + b * x + c * y

    nodal_values = [field(x, y) for x, y in node_coords]

    xi, eta = 0.25, -0.6
    shape_values = quad_shape_functions(xi, eta)
    # Physical (x, y) at (xi, eta), via isoparametric mapping.
    x = sum(n * coord[0] for n, coord in zip(shape_values, node_coords, strict=True))
    y = sum(n * coord[1] for n, coord in zip(shape_values, node_coords, strict=True))
    interpolated = sum(n * v for n, v in zip(shape_values, nodal_values, strict=True))

    assert_allclose(interpolated, field(x, y))


def test_quad_shape_function_derivatives_match_finite_difference() -> None:
    xi, eta = 0.2, 0.4
    h = 1e-6

    n0 = np.array(quad_shape_functions(xi, eta))
    n_xi_plus = np.array(quad_shape_functions(xi + h, eta))
    n_eta_plus = np.array(quad_shape_functions(xi, eta + h))

    expected_dxi = (n_xi_plus - n0) / h
    expected_deta = (n_eta_plus - n0) / h

    dn_dxi, dn_deta = quad_shape_function_derivatives(xi, eta)
    assert_allclose(dn_dxi, expected_dxi, atol=1e-6)
    assert_allclose(dn_deta, expected_deta, atol=1e-6)


def test_quad_shape_function_derivatives_sum_to_zero() -> None:
    """Differentiating the partition-of-unity identity N1+N2+N3+N4=1 gives
    sum(dNi/dxi) = 0 and sum(dNi/deta) = 0 at every point.
    """
    dn_dxi, dn_deta = quad_shape_function_derivatives(0.3, -0.6)

    assert_allclose(sum(dn_dxi), 0.0, atol=1e-12)
    assert_allclose(sum(dn_deta), 0.0, atol=1e-12)
