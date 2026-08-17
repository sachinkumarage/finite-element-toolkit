"""2D point.

The atomic building block of the toolkit's geometry model. Deliberately
minimal -- a bare ``(x, y)`` pair, in meters (SI units) -- since Version 9
only needs points as the endpoints of line segments
(:mod:`femtoolkit.geometry.line`) and the corners of rectangular domains
(:mod:`femtoolkit.geometry.rectangle`), not as a general-purpose CAD
primitive.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from femtoolkit.exceptions import ValidationError


@dataclass(frozen=True)
class Point2D:
    """A point in the 2D plane.

    Attributes:
        x: X coordinate, in meters.
        y: Y coordinate, in meters.

    Raises:
        ValidationError: If ``x`` or ``y`` is not a finite number.

    Example:
        >>> Point2D(1.0, 2.0)
        Point2D(x=1.0, y=2.0)
    """

    x: float
    y: float

    def __post_init__(self) -> None:
        """Validate the coordinates immediately after construction.

        Raises:
            ValidationError: If ``x`` or ``y`` is not a finite number.
        """
        if not math.isfinite(self.x) or not math.isfinite(self.y):
            raise ValidationError(f"Point2D coordinates must be finite, got ({self.x}, {self.y}).")

    def distance_to(self, other: Point2D) -> float:
        """Euclidean distance to another point.

        Args:
            other: The point to measure distance to.

        Returns:
            Distance, in meters.
        """
        return math.hypot(self.x - other.x, self.y - other.y)
