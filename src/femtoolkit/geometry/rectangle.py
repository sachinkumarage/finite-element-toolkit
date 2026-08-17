"""Rectangular 2D domain.

The one concrete geometry type Version 9 provides -- deliberately not a
CAD system, just enough shape to describe the domains
:mod:`femtoolkit.mesh.generator` already produces and to name their four
straight edges. A :class:`Rectangle`'s corners and node-generation origin
match :func:`~femtoolkit.mesh.generator.create_quad_mesh`/
:func:`~femtoolkit.mesh.generator.create_triangular_mesh` exactly (both
place node 1 at ``(0, 0)``), so ``Rectangle(width=W, height=H)`` and
``create_quad_mesh(width=W, height=H, ...)`` describe the same domain.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from femtoolkit.exceptions import ValidationError
from femtoolkit.geometry.boundary import BoundaryRegion
from femtoolkit.geometry.line import LineSegment2D
from femtoolkit.geometry.point import Point2D

BOUNDARY_NAMES = ("left", "right", "top", "bottom")


@dataclass(frozen=True)
class Rectangle:
    """An axis-aligned rectangular domain.

    Attributes:
        width: Extent along X, in meters. Must be positive.
        height: Extent along Y, in meters. Must be positive.
        origin: Bottom-left corner. Defaults to the global origin,
            matching the mesh generator's node-placement convention.

    Raises:
        ValidationError: If ``width`` or ``height`` is not positive and finite.

    Example:
        >>> domain = Rectangle(width=2.0, height=1.0)
        >>> domain.boundary("left").outward_normal
        (-1.0, 0.0)
    """

    width: float
    height: float
    origin: Point2D = Point2D(0.0, 0.0)

    def __post_init__(self) -> None:
        """Validate the rectangle's dimensions immediately after construction.

        Raises:
            ValidationError: If ``width`` or ``height`` is not positive and finite.
        """
        if not math.isfinite(self.width) or self.width <= 0:
            raise ValidationError(f"Rectangle width must be positive, got {self.width}.")
        if not math.isfinite(self.height) or self.height <= 0:
            raise ValidationError(f"Rectangle height must be positive, got {self.height}.")

    @property
    def corners(self) -> tuple[Point2D, Point2D, Point2D, Point2D]:
        """The four corners, counter-clockwise from the bottom-left.

        Returns:
            ``(bottom_left, bottom_right, top_right, top_left)``, matching
            the node-ordering convention used throughout
            :class:`~femtoolkit.mesh.quad_element.QuadElement2D` and
            :mod:`femtoolkit.mesh.generator`.
        """
        x0, y0 = self.origin.x, self.origin.y
        return (
            Point2D(x0, y0),
            Point2D(x0 + self.width, y0),
            Point2D(x0 + self.width, y0 + self.height),
            Point2D(x0, y0 + self.height),
        )

    def boundary(self, name: str) -> BoundaryRegion:
        """Return the named boundary region.

        Args:
            name: One of ``"left"``, ``"right"``, ``"top"``, ``"bottom"``.

        Returns:
            The corresponding :class:`~femtoolkit.geometry.boundary.BoundaryRegion`.

        Raises:
            ValidationError: If ``name`` is not a recognized boundary name.
        """
        bottom_left, bottom_right, top_right, top_left = self.corners
        if name == "bottom":
            return BoundaryRegion(
                "bottom", LineSegment2D(bottom_left, bottom_right), outward_normal=(0.0, -1.0)
            )
        if name == "right":
            return BoundaryRegion(
                "right", LineSegment2D(bottom_right, top_right), outward_normal=(1.0, 0.0)
            )
        if name == "top":
            return BoundaryRegion(
                "top", LineSegment2D(top_right, top_left), outward_normal=(0.0, 1.0)
            )
        if name == "left":
            return BoundaryRegion(
                "left", LineSegment2D(top_left, bottom_left), outward_normal=(-1.0, 0.0)
            )
        raise ValidationError(
            f"Unknown boundary {name!r}; Rectangle defines {list(BOUNDARY_NAMES)}."
        )

    @property
    def boundaries(self) -> dict[str, BoundaryRegion]:
        """All four named boundary regions, keyed by name."""
        return {name: self.boundary(name) for name in BOUNDARY_NAMES}

    @property
    def area(self) -> float:
        """Rectangle area, in square meters."""
        return self.width * self.height
