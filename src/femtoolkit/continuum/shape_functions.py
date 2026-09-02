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


def tet4_shape_functions(
    xi: float, eta: float, zeta: float
) -> tuple[float, float, float, float]:
    """Evaluate the four linear shape functions of a 4-node tetrahedron (TET4, Version 15).

    .. code-block:: text

        N1(xi, eta, zeta) = 1 - xi - eta - zeta
        N2(xi, eta, zeta) = xi
        N3(xi, eta, zeta) = eta
        N4(xi, eta, zeta) = zeta

    Node 1 sits at the natural-coordinate origin ``(0, 0, 0)``; nodes 2,
    3, 4 sit at ``(1, 0, 0)``, ``(0, 1, 0)``, ``(0, 0, 1)`` respectively.
    Each ``Ni`` is linear in the natural coordinates, so (like
    :func:`triangle_shape_functions`, its 2D analogue) its gradient is
    constant over the element -- the origin of TET4's constant-strain
    property (see :mod:`femtoolkit.continuum.strain`).

    Args:
        xi: First natural coordinate.
        eta: Second natural coordinate.
        zeta: Third natural coordinate.

    Returns:
        ``(N1, N2, N3, N4)`` evaluated at ``(xi, eta, zeta)``.

    Example:
        >>> tet4_shape_functions(0.0, 0.0, 0.0)
        (1.0, 0.0, 0.0, 0.0)
    """
    return 1.0 - xi - eta - zeta, xi, eta, zeta


_Tet4DerivativeRow = tuple[float, float, float, float]
_Tet4DerivativeRows = tuple[_Tet4DerivativeRow, _Tet4DerivativeRow, _Tet4DerivativeRow]


def tet4_shape_function_derivatives() -> _Tet4DerivativeRows:
    """Return the four TET4 shape functions' constant natural-coordinate derivatives.

    Because each shape function is linear (see :func:`tet4_shape_functions`),
    every derivative is a constant, independent of ``(xi, eta, zeta)`` --
    unlike :func:`quad_shape_function_derivatives`, this function takes no
    arguments.

    Returns:
        ``((dN1/dxi, dN2/dxi, dN3/dxi, dN4/dxi),
        (dN1/deta, dN2/deta, dN3/deta, dN4/deta),
        (dN1/dzeta, dN2/dzeta, dN3/dzeta, dN4/dzeta))``.
    """
    dn_dxi = (-1.0, 1.0, 0.0, 0.0)
    dn_deta = (-1.0, 0.0, 1.0, 0.0)
    dn_dzeta = (-1.0, 0.0, 0.0, 1.0)
    return dn_dxi, dn_deta, dn_dzeta


_HEX8_NATURAL_COORDS: tuple[tuple[float, float, float], ...] = (
    (-1.0, -1.0, -1.0),
    (1.0, -1.0, -1.0),
    (1.0, 1.0, -1.0),
    (-1.0, 1.0, -1.0),
    (-1.0, -1.0, 1.0),
    (1.0, -1.0, 1.0),
    (1.0, 1.0, 1.0),
    (-1.0, 1.0, 1.0),
)
"""Natural-coordinate corners of the 8-node hexahedron (HEX8, Version 15).

.. code-block:: text

    Bottom face (zeta = -1), counter-clockwise from (x, y) = (-1, -1):
        Node 1: (-1, -1, -1)      Node 4 ------- Node 3
        Node 2: ( 1, -1, -1)        |               |
        Node 3: ( 1,  1, -1)        |               |
        Node 4: (-1,  1, -1)      Node 1 ------- Node 2

    Top face (zeta = +1), directly above the bottom face, same winding:
        Node 5: (-1, -1,  1)      Node 8 ------- Node 7
        Node 6: ( 1, -1,  1)        |               |
        Node 7: ( 1,  1,  1)        |               |
        Node 8: (-1,  1,  1)      Node 5 ------- Node 6

This is the standard isoparametric hexahedron node ordering used by most
commercial FEA codes (e.g. Abaqus C3D8, ANSYS SOLID185).
"""


def hex8_shape_functions(
    xi: float, eta: float, zeta: float
) -> tuple[float, float, float, float, float, float, float, float]:
    """Evaluate the eight trilinear shape functions of an 8-node hexahedron (HEX8, Version 15).

    .. code-block:: text

        Ni(xi, eta, zeta) = (1 + xi*xi_i)(1 + eta*eta_i)(1 + zeta*zeta_i) / 8

    where ``(xi_i, eta_i, zeta_i)`` is node ``i``'s natural-coordinate
    corner (see :data:`_HEX8_NATURAL_COORDS`). Each ``Ni`` is *trilinear*
    (linear in each natural coordinate separately, with ``xi*eta``,
    ``eta*zeta``, ``xi*zeta``, and ``xi*eta*zeta`` cross terms), so its
    gradient is **not** constant over the element -- a HEX8 element's
    strain varies within the element, unlike TET4's (see
    :func:`tet4_shape_functions`).

    Args:
        xi: First natural coordinate, expected in ``[-1, 1]``.
        eta: Second natural coordinate, expected in ``[-1, 1]``.
        zeta: Third natural coordinate, expected in ``[-1, 1]``.

    Returns:
        ``(N1, N2, ..., N8)`` evaluated at ``(xi, eta, zeta)``.

    Example:
        >>> hex8_shape_functions(0.0, 0.0, 0.0)
        (0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125)
    """
    return tuple(
        (1.0 + xi * xi_i) * (1.0 + eta * eta_i) * (1.0 + zeta * zeta_i) / 8.0
        for xi_i, eta_i, zeta_i in _HEX8_NATURAL_COORDS
    )


def hex8_shape_function_derivatives(
    xi: float, eta: float, zeta: float
) -> tuple[
    tuple[float, ...],
    tuple[float, ...],
    tuple[float, ...],
]:
    """Evaluate the natural-coordinate derivatives of the eight HEX8 shape functions.

    .. code-block:: text

        dNi/dxi   = xi_i   * (1 + eta*eta_i)  * (1 + zeta*zeta_i) / 8
        dNi/deta  = eta_i  * (1 + xi*xi_i)    * (1 + zeta*zeta_i) / 8
        dNi/dzeta = zeta_i * (1 + xi*xi_i)    * (1 + eta*eta_i)   / 8

    These are the inputs to the 3D isoparametric Jacobian (see
    :func:`~femtoolkit.continuum.jacobian.jacobian_matrix_3d`); like the
    Q4 case, they are *not* yet the physical-coordinate derivatives
    ``dNi/dx``, ``dNi/dy``, ``dNi/dz`` the strain-displacement matrix
    needs.

    Args:
        xi: First natural coordinate, expected in ``[-1, 1]``.
        eta: Second natural coordinate, expected in ``[-1, 1]``.
        zeta: Third natural coordinate, expected in ``[-1, 1]``.

    Returns:
        ``((dN1/dxi, ..., dN8/dxi), (dN1/deta, ..., dN8/deta),
        (dN1/dzeta, ..., dN8/dzeta))``.
    """
    dn_dxi = tuple(
        xi_i * (1.0 + eta * eta_i) * (1.0 + zeta * zeta_i) / 8.0
        for xi_i, eta_i, zeta_i in _HEX8_NATURAL_COORDS
    )
    dn_deta = tuple(
        eta_i * (1.0 + xi * xi_i) * (1.0 + zeta * zeta_i) / 8.0
        for xi_i, eta_i, zeta_i in _HEX8_NATURAL_COORDS
    )
    dn_dzeta = tuple(
        zeta_i * (1.0 + xi * xi_i) * (1.0 + eta * eta_i) / 8.0
        for xi_i, eta_i, zeta_i in _HEX8_NATURAL_COORDS
    )
    return dn_dxi, dn_deta, dn_dzeta
