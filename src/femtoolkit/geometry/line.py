"""2D line segment.

A straight segment between two points, used both as a standalone
geometric primitive and as the shape of a rectangular domain's boundary
regions (:mod:`femtoolkit.geometry.boundary`). :meth:`LineSegment2D.contains_point`
implements a general point-to-segment distance test -- not a check
hardcoded to horizontal or vertical lines -- so it works correctly for
any orientation, keeping the door open for non-axis-aligned boundaries in
future geometry types.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from femtoolkit.exceptions import ValidationError
from femtoolkit.geometry.point import Point2D


@dataclass(frozen=True)
class LineSegment2D:
    """A straight line segment between two points.

    Attributes:
        start: Segment start point.
        end: Segment end point.

    Raises:
        ValidationError: If ``start`` and ``end`` coincide (a zero-length
            segment).

    Example:
        >>> segment = LineSegment2D(Point2D(0.0, 0.0), Point2D(1.0, 0.0))
        >>> segment.length
        1.0
    """

    start: Point2D
    end: Point2D

    def __post_init__(self) -> None:
        """Validate the segment immediately after construction.

        Raises:
            ValidationError: If ``start`` and ``end`` coincide.
        """
        if self.length <= 0.0:
            raise ValidationError(
                f"LineSegment2D requires distinct start and end points, got {self.start} "
                f"and {self.end}."
            )

    @property
    def length(self) -> float:
        """Segment length, in meters."""
        return self.start.distance_to(self.end)

    def contains_point(self, point: Point2D, tolerance: float = 1e-9) -> bool:
        """Check whether a point lies on this segment, within a distance tolerance.

        Computes the perpendicular distance from ``point`` to the segment
        (projecting onto the infinite line and clamping the projection to
        the segment's extent, so points beyond either endpoint are
        correctly treated as off the segment, not just off the line).

        Args:
            point: The point to test.
            tolerance: Maximum perpendicular distance, in meters, for the
                point to be considered on the segment. Accounts for
                floating-point round-off in generated node coordinates.

        Returns:
            ``True`` if ``point`` lies within ``tolerance`` of the segment.
        """
        dx = self.end.x - self.start.x
        dy = self.end.y - self.start.y
        length_squared = dx * dx + dy * dy

        t = ((point.x - self.start.x) * dx + (point.y - self.start.y) * dy) / length_squared
        t_clamped = max(0.0, min(1.0, t))

        closest_x = self.start.x + t_clamped * dx
        closest_y = self.start.y + t_clamped * dy
        distance = math.hypot(point.x - closest_x, point.y - closest_y)
        return distance <= tolerance
