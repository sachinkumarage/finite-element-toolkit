"""Tests for the CST linear shape functions (femtoolkit.continuum.shape_functions)."""

import pytest
from numpy.testing import assert_allclose

from femtoolkit.continuum.shape_functions import triangle_shape_functions
from femtoolkit.exceptions import DegenerateElementError

TRIANGLE = (0.0, 0.0, 1.0, 0.0, 0.0, 1.0)  # node1=(0,0), node2=(1,0), node3=(0,1)


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
