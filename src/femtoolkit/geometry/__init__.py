"""Lightweight 2D geometry: points, line segments, rectangles, and named boundary regions.

This is not a CAD system -- no NURBS, splines, curves, or boolean
operations. It exists to answer one question cleanly: *which physical
location does a named boundary (e.g. "left") correspond to*, so that
:mod:`femtoolkit.mesh` and :mod:`femtoolkit.analysis` can select mesh
nodes and edges on that boundary without the caller manually enumerating
node IDs.
"""

from femtoolkit.geometry.boundary import BoundaryRegion
from femtoolkit.geometry.line import LineSegment2D
from femtoolkit.geometry.point import Point2D
from femtoolkit.geometry.rectangle import BOUNDARY_NAMES, Rectangle

__all__ = [
    "BOUNDARY_NAMES",
    "BoundaryRegion",
    "LineSegment2D",
    "Point2D",
    "Rectangle",
]
