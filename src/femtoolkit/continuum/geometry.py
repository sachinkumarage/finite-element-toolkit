"""Triangle geometry for 2D continuum elements.

This module computes the one geometric quantity every other piece of
constant-strain-triangle (CST) math is built on: the element's area.

**Orientation policy.** The signed-area formula below is positive for
nodes listed counter-clockwise and negative for nodes listed clockwise.
Rather than rejecting clockwise input, this module -- and every function
built on top of it (see :mod:`femtoolkit.continuum.strain`) -- accepts
either orientation and normalizes automatically: the *signed* area is
used consistently inside the strain-displacement matrix (matching the
literal shape-function-derivative formulas), while the *absolute* area
is used everywhere a physical area is needed (e.g. the element stiffness
formula ``Ke = t * A * B^T * D * B``). This combination makes every
downstream quantity -- strain, stress, stiffness -- identical regardless
of whether the caller happened to list nodes clockwise or
counter-clockwise, without silently reordering the caller's nodes.
"""

from __future__ import annotations

MIN_TRIANGLE_AREA: float = 1e-12
"""Minimum acceptable absolute triangle area, in square meters.

Guards against degenerate triangles (collinear or duplicate-node
geometry), for which the strain-displacement matrix is undefined (it
divides by twice the signed area). This is not a general
dimensional-analysis tolerance -- it is small enough that no physically
reasonable SI-scale element area (from micro-scale to continental-scale
models) could legitimately fall below it.
"""


def triangle_signed_area(
    x1: float, y1: float, x2: float, y2: float, x3: float, y3: float
) -> float:
    """Compute the signed area of a triangle from its three vertices.

    .. code-block:: text

        2A = x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2)
        A = 2A / 2

    Positive when the nodes ``(1, 2, 3)`` are listed counter-clockwise;
    negative when listed clockwise. See the module docstring for how
    this sign is used elsewhere.

    Args:
        x1: X coordinate of node 1, in meters.
        y1: Y coordinate of node 1, in meters.
        x2: X coordinate of node 2, in meters.
        y2: Y coordinate of node 2, in meters.
        x3: X coordinate of node 3, in meters.
        y3: Y coordinate of node 3, in meters.

    Returns:
        The signed area, in square meters.

    Example:
        >>> triangle_signed_area(0.0, 0.0, 1.0, 0.0, 0.0, 1.0)
        0.5
    """
    twice_signed_area = x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)
    return twice_signed_area / 2.0
