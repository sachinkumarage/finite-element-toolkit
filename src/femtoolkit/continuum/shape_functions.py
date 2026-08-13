"""Shape functions for the toolkit's 2D continuum elements.

For a 2D continuum element, the displacement field is approximated from
the nodal displacements:

.. code-block:: text

    u = N1*u1 + N2*u2 + ... + Nn*un
    v = N1*v1 + N2*v2 + ... + Nn*vn

Two properties define a valid finite element shape function set:

* **Partition of unity**: the shape functions sum to 1 everywhere, so a
  rigid-body translation of every node produces the same translation at
  every interior point.
* **Nodal (Kronecker delta) property**: ``Ni(node_j) = 1`` if ``i == j``,
  else ``0`` -- each shape function equals 1 at its own node and 0 at
  every other node.

This module provides two independent shape function families:

* :func:`triangle_shape_functions` -- the 3-node constant strain
  triangle (CST, Version 6). Each ``Ni`` is linear in the *physical*
  coordinates ``(x, y)`` directly: ``Ni(x, y) = (ai + bi*x + ci*y) / (2A)``.
  Because each is linear, its gradient is constant over the triangle --
  the origin of the "constant strain" property documented in
  :mod:`femtoolkit.continuum.strain`.
* :func:`quad_shape_functions` -- the 4-node bilinear quadrilateral (Q4,
  Version 7). Each ``Ni`` is defined in *natural* coordinates
  ``(xi, eta) in [-1, 1] x [-1, 1]``, not physical coordinates directly;
  see :mod:`femtoolkit.continuum.jacobian` for how natural-coordinate
  derivatives are converted to physical ones via isoparametric mapping.
  Because each ``Ni`` is *bilinear* (linear in ``xi`` and ``eta``
  separately, but containing an ``xi*eta`` cross term), its gradient is
  **not** constant over the element -- a Q4 element's strain varies
  within the element, unlike a CST element's.
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


def quad_shape_functions(xi: float, eta: float) -> tuple[float, float, float, float]:
    """Evaluate the four bilinear shape functions at natural coordinates ``(xi, eta)``.

    Node ordering follows the standard counter-clockwise isoparametric
    convention, matching the corners of the natural-coordinate square:

    .. code-block:: text

        Node 1: (xi, eta) = (-1, -1)      Node 4 ------- Node 3
        Node 2: (xi, eta) = ( 1, -1)        |               |
        Node 3: (xi, eta) = ( 1,  1)        |               |
        Node 4: (xi, eta) = (-1,  1)      Node 1 ------- Node 2

        N1(xi,eta) = (1-xi)*(1-eta) / 4
        N2(xi,eta) = (1+xi)*(1-eta) / 4
        N3(xi,eta) = (1+xi)*(1+eta) / 4
        N4(xi,eta) = (1-xi)*(1+eta) / 4

    Args:
        xi: First natural coordinate, expected in ``[-1, 1]``.
        eta: Second natural coordinate, expected in ``[-1, 1]``.

    Returns:
        ``(N1, N2, N3, N4)`` evaluated at ``(xi, eta)``.

    Example:
        >>> quad_shape_functions(0.0, 0.0)
        (0.25, 0.25, 0.25, 0.25)
    """
    n1 = (1.0 - xi) * (1.0 - eta) / 4.0
    n2 = (1.0 + xi) * (1.0 - eta) / 4.0
    n3 = (1.0 + xi) * (1.0 + eta) / 4.0
    n4 = (1.0 - xi) * (1.0 + eta) / 4.0
    return n1, n2, n3, n4


def quad_shape_function_derivatives(
    xi: float, eta: float
) -> tuple[tuple[float, float, float, float], tuple[float, float, float, float]]:
    """Evaluate the natural-coordinate derivatives of the four bilinear shape functions.

    .. code-block:: text

        dN1/dxi = -(1-eta)/4     dN1/deta = -(1-xi)/4
        dN2/dxi =  (1-eta)/4     dN2/deta = -(1+xi)/4
        dN3/dxi =  (1+eta)/4     dN3/deta =  (1+xi)/4
        dN4/dxi = -(1+eta)/4     dN4/deta =  (1-xi)/4

    These are the inputs to the isoparametric Jacobian (see
    :func:`~femtoolkit.continuum.jacobian.jacobian_matrix`); they are
    *not* yet the physical-coordinate derivatives ``dNi/dx``, ``dNi/dy``
    needed by the strain-displacement matrix -- computing those requires
    dividing through by the Jacobian, which depends on the element's
    actual node coordinates, not just ``(xi, eta)``.

    Args:
        xi: First natural coordinate, expected in ``[-1, 1]``.
        eta: Second natural coordinate, expected in ``[-1, 1]``.

    Returns:
        ``((dN1/dxi, dN2/dxi, dN3/dxi, dN4/dxi), (dN1/deta, dN2/deta, dN3/deta, dN4/deta))``.
    """
    dn_dxi = (
        -(1.0 - eta) / 4.0,
        (1.0 - eta) / 4.0,
        (1.0 + eta) / 4.0,
        -(1.0 + eta) / 4.0,
    )
    dn_deta = (
        -(1.0 - xi) / 4.0,
        -(1.0 + xi) / 4.0,
        (1.0 + xi) / 4.0,
        (1.0 - xi) / 4.0,
    )
    return dn_dxi, dn_deta
