"""Named boundary regions.

A :class:`BoundaryRegion` is a named piece of a domain's boundary --
``"left"``, ``"right"``, ``"top"``, ``"bottom"`` for a
:class:`~femtoolkit.geometry.rectangle.Rectangle`, but the type itself
has no idea what a rectangle is. It just pairs a name with a geometric
shape (currently always a :class:`~femtoolkit.geometry.line.LineSegment2D`,
since Version 9 only has straight boundaries) and an outward-pointing
unit normal vector. This is deliberate: any future geometry type
(a non-rectangular polygon, for example) constructs its own
:class:`BoundaryRegion` instances with whatever names and normals make
sense for its own shape, without this module -- or anything downstream
that consumes a :class:`BoundaryRegion` (mesh boundary-node selection,
boundary-condition assignment, distributed loads) -- needing to know
anything rectangle-specific.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from femtoolkit.exceptions import ValidationError
from femtoolkit.geometry.line import LineSegment2D
from femtoolkit.geometry.point import Point2D


@dataclass(frozen=True)
class BoundaryRegion:
    """A named region of a domain's boundary.

    Attributes:
        name: Human-readable boundary name (e.g. ``"left"``).
        segment: The boundary's geometry.
        outward_normal: Unit vector ``(nx, ny)`` pointing away from the
            domain interior, used to resolve normal/tangential
            distributed tractions (see
            :mod:`femtoolkit.analysis.distributed_load`) into global
            components.

    Raises:
        ValidationError: If ``outward_normal`` is not (approximately) a
            unit vector.

    Example:
        >>> segment = LineSegment2D(Point2D(0.0, 0.0), Point2D(0.0, 1.0))
        >>> left = BoundaryRegion("left", segment, outward_normal=(-1.0, 0.0))
        >>> left.contains_point(0.0, 0.5)
        True
    """

    name: str
    segment: LineSegment2D
    outward_normal: tuple[float, float]

    def __post_init__(self) -> None:
        """Validate the outward normal immediately after construction.

        Raises:
            ValidationError: If ``outward_normal`` is not a finite,
                (approximately) unit-length vector.
        """
        nx, ny = self.outward_normal
        if not math.isfinite(nx) or not math.isfinite(ny):
            raise ValidationError(f"outward_normal must be finite, got {self.outward_normal}.")
        magnitude = math.hypot(nx, ny)
        if not math.isclose(magnitude, 1.0, abs_tol=1e-6):
            raise ValidationError(
                f"outward_normal must be a unit vector, got {self.outward_normal} "
                f"(magnitude {magnitude})."
            )

    def contains_point(self, x: float, y: float, tolerance: float = 1e-9) -> bool:
        """Check whether a physical point lies on this boundary.

        Args:
            x: X coordinate, in meters.
            y: Y coordinate, in meters.
            tolerance: Maximum distance, in meters, for the point to be
                considered on the boundary.

        Returns:
            ``True`` if ``(x, y)`` lies within ``tolerance`` of this
            boundary's geometry.
        """
        return self.segment.contains_point(Point2D(x, y), tolerance)
