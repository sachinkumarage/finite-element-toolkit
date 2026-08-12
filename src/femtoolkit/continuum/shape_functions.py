"""Linear shape functions for the constant strain triangle (CST).

For a 2D continuum element, the displacement field is approximated from
the nodal displacements:

.. code-block:: text

    u(x, y) = N1(x,y)*u1 + N2(x,y)*u2 + N3(x,y)*u3
    v(x, y) = N1(x,y)*v1 + N2(x,y)*v2 + N3(x,y)*v3

For the 3-node triangle, each ``Ni`` is linear in ``x`` and ``y``:

.. code-block:: text

    Ni(x, y) = (ai + bi*x + ci*y) / (2A)

Two properties define a valid finite element shape function set, both
verified directly in ``tests/test_shape_functions.py``:

* **Partition of unity**: ``N1(x,y) + N2(x,y) + N3(x,y) = 1`` everywhere,
  so a rigid-body translation of all three nodes produces the same
  translation at every interior point.
* **Nodal (Kronecker delta) property**: ``Ni(node_j) = 1`` if ``i == j``,
  else ``0`` -- each shape function equals 1 at its own node and 0 at the
  other two.

Because each ``Ni`` is linear in ``x`` and ``y``, its gradient is
constant over the triangle -- this is the origin of the "constant
strain" property documented in :mod:`femtoolkit.continuum.strain`.
"""

from __future__ import annotations

from femtoolkit.continuum.geometry import MIN_TRIANGLE_AREA, triangle_signed_area
from femtoolkit.exceptions import DegenerateElementError


def triangle_shape_functions(
    x: float,
    y: float,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    x3: float,
    y3: float,
) -> tuple[float, float, float]:
    """Evaluate the three linear shape functions at a point ``(x, y)``.

    Args:
        x: X coordinate of the evaluation point, in meters.
        y: Y coordinate of the evaluation point, in meters.
        x1: X coordinate of node 1, in meters.
        y1: Y coordinate of node 1, in meters.
        x2: X coordinate of node 2, in meters.
        y2: Y coordinate of node 2, in meters.
        x3: X coordinate of node 3, in meters.
        y3: Y coordinate of node 3, in meters.

    Returns:
        ``(N1, N2, N3)`` evaluated at ``(x, y)``.

    Raises:
        DegenerateElementError: If the triangle's area is (near) zero.
    """
    signed_area = triangle_signed_area(x1, y1, x2, y2, x3, y3)
    if abs(signed_area) < MIN_TRIANGLE_AREA:
        raise DegenerateElementError(
            f"Triangle with nodes ({x1},{y1}), ({x2},{y2}), ({x3},{y3}) has "
            f"area {signed_area}, which is degenerate (collinear or duplicate nodes)."
        )

    a1 = x2 * y3 - x3 * y2
    a2 = x3 * y1 - x1 * y3
    a3 = x1 * y2 - x2 * y1
    b1, b2, b3 = y2 - y3, y3 - y1, y1 - y2
    c1, c2, c3 = x3 - x2, x1 - x3, x2 - x1

    two_signed_area = 2.0 * signed_area
    n1 = (a1 + b1 * x + c1 * y) / two_signed_area
    n2 = (a2 + b2 * x + c2 * y) / two_signed_area
    n3 = (a3 + b3 * x + c3 * y) / two_signed_area
    return n1, n2, n3
