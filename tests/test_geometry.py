"""Tests for the geometry package (femtoolkit.geometry)."""

import math

import pytest
from numpy.testing import assert_allclose

from femtoolkit.exceptions import ValidationError
from femtoolkit.geometry import BoundaryRegion, LineSegment2D, Point2D, Rectangle

# --- Point2D ---


def test_point_creation() -> None:
    point = Point2D(1.0, 2.0)

    assert point.x == 1.0
    assert point.y == 2.0


@pytest.mark.parametrize("x,y", [(math.nan, 0.0), (0.0, math.nan), (math.inf, 0.0)])
def test_point_rejects_non_finite_coordinates(x: float, y: float) -> None:
    with pytest.raises(ValidationError):
        Point2D(x, y)


def test_point_distance_to() -> None:
    a = Point2D(0.0, 0.0)
    b = Point2D(3.0, 4.0)

    assert_allclose(a.distance_to(b), 5.0)


# --- LineSegment2D ---


def test_line_segment_length() -> None:
    segment = LineSegment2D(Point2D(0.0, 0.0), Point2D(3.0, 4.0))

    assert_allclose(segment.length, 5.0)


def test_line_segment_rejects_zero_length() -> None:
    with pytest.raises(ValidationError):
        LineSegment2D(Point2D(1.0, 1.0), Point2D(1.0, 1.0))


def test_line_segment_contains_endpoint() -> None:
    segment = LineSegment2D(Point2D(0.0, 0.0), Point2D(1.0, 0.0))

    assert segment.contains_point(Point2D(0.0, 0.0))
    assert segment.contains_point(Point2D(1.0, 0.0))


def test_line_segment_contains_midpoint() -> None:
    segment = LineSegment2D(Point2D(0.0, 0.0), Point2D(1.0, 0.0))

    assert segment.contains_point(Point2D(0.5, 0.0))


def test_line_segment_rejects_point_off_the_line() -> None:
    segment = LineSegment2D(Point2D(0.0, 0.0), Point2D(1.0, 0.0))

    assert not segment.contains_point(Point2D(0.5, 0.1))


def test_line_segment_rejects_point_beyond_the_endpoint() -> None:
    """A point collinear with the segment but past an endpoint is not contained."""
    segment = LineSegment2D(Point2D(0.0, 0.0), Point2D(1.0, 0.0))

    assert not segment.contains_point(Point2D(1.5, 0.0))
    assert not segment.contains_point(Point2D(-0.5, 0.0))


def test_line_segment_diagonal_orientation() -> None:
    """contains_point must work for non-axis-aligned segments too."""
    segment = LineSegment2D(Point2D(0.0, 0.0), Point2D(1.0, 1.0))

    assert segment.contains_point(Point2D(0.5, 0.5))
    assert not segment.contains_point(Point2D(0.5, 0.6))


def test_line_segment_tolerance() -> None:
    segment = LineSegment2D(Point2D(0.0, 0.0), Point2D(1.0, 0.0))

    assert segment.contains_point(Point2D(0.5, 1e-10), tolerance=1e-9)
    assert not segment.contains_point(Point2D(0.5, 1e-3), tolerance=1e-9)


# --- Rectangle ---


def test_rectangle_dimensions() -> None:
    domain = Rectangle(width=2.0, height=1.0)

    assert domain.width == 2.0
    assert domain.height == 1.0
    assert_allclose(domain.area, 2.0)


@pytest.mark.parametrize("width,height", [(0.0, 1.0), (-1.0, 1.0), (1.0, 0.0), (1.0, -1.0)])
def test_rectangle_rejects_non_positive_dimensions(width: float, height: float) -> None:
    with pytest.raises(ValidationError):
        Rectangle(width=width, height=height)


def test_rectangle_corners_counter_clockwise_from_bottom_left() -> None:
    domain = Rectangle(width=2.0, height=1.0)

    bottom_left, bottom_right, top_right, top_left = domain.corners
    assert (bottom_left.x, bottom_left.y) == (0.0, 0.0)
    assert (bottom_right.x, bottom_right.y) == (2.0, 0.0)
    assert (top_right.x, top_right.y) == (2.0, 1.0)
    assert (top_left.x, top_left.y) == (0.0, 1.0)


def test_rectangle_boundaries_dict_has_four_named_regions() -> None:
    domain = Rectangle(width=2.0, height=1.0)

    boundaries = domain.boundaries
    assert set(boundaries.keys()) == {"left", "right", "top", "bottom"}
    assert all(isinstance(region, BoundaryRegion) for region in boundaries.values())


def test_rectangle_left_boundary() -> None:
    domain = Rectangle(width=2.0, height=1.0)

    left = domain.boundary("left")
    assert left.name == "left"
    assert left.outward_normal == (-1.0, 0.0)
    assert left.contains_point(0.0, 0.5)
    assert not left.contains_point(2.0, 0.5)


def test_rectangle_right_boundary() -> None:
    domain = Rectangle(width=2.0, height=1.0)

    right = domain.boundary("right")
    assert right.outward_normal == (1.0, 0.0)
    assert right.contains_point(2.0, 0.5)
    assert not right.contains_point(0.0, 0.5)


def test_rectangle_top_boundary() -> None:
    domain = Rectangle(width=2.0, height=1.0)

    top = domain.boundary("top")
    assert top.outward_normal == (0.0, 1.0)
    assert top.contains_point(1.0, 1.0)
    assert not top.contains_point(1.0, 0.0)


def test_rectangle_bottom_boundary() -> None:
    domain = Rectangle(width=2.0, height=1.0)

    bottom = domain.boundary("bottom")
    assert bottom.outward_normal == (0.0, -1.0)
    assert bottom.contains_point(1.0, 0.0)
    assert not bottom.contains_point(1.0, 1.0)


def test_rectangle_corner_point_is_on_two_boundaries() -> None:
    domain = Rectangle(width=2.0, height=1.0)

    assert domain.boundary("left").contains_point(0.0, 0.0)
    assert domain.boundary("bottom").contains_point(0.0, 0.0)


def test_rectangle_unknown_boundary_name_raises() -> None:
    domain = Rectangle(width=2.0, height=1.0)

    with pytest.raises(ValidationError):
        domain.boundary("diagonal")


# --- BoundaryRegion ---


def test_boundary_region_rejects_non_unit_normal() -> None:
    segment = LineSegment2D(Point2D(0.0, 0.0), Point2D(0.0, 1.0))

    with pytest.raises(ValidationError):
        BoundaryRegion("left", segment, outward_normal=(-2.0, 0.0))
