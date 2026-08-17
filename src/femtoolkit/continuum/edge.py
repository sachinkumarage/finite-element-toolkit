"""Equivalent nodal forces for a distributed traction along a straight element edge.

A **distributed load** (a *traction*, in pascals -- force per unit area)
applied along a 2D continuum element's edge cannot be handed straight to
the solver, which only understands discrete nodal forces (see
:class:`~femtoolkit.analysis.loads.NodalLoad`). It must first be
converted into a statically **equivalent** set of nodal forces -- forces
at the edge's two nodes that reproduce the same total force and the same
virtual work as the real distributed traction.

**Traction vs. pressure vs. force vs. force per unit length.** These are
easy to conflate, so this module is explicit about each:

* **Traction** ``t = [tx, ty]``, in Pa (N/m^2) -- the physical
  distributed load, force per unit *area* of the edge. "Pressure" is the
  special case of a traction acting purely along the inward normal
  direction; this module treats any traction the same way regardless of
  its direction.
* **Force per unit length**, in N/m -- traction multiplied by element
  thickness (``t * thickness``); the quantity actually integrated along
  the edge, since a 2D continuum element's "area" for load purposes is
  ``edge_length * thickness``.
* **Nodal force**, in N -- the equivalent concentrated force at one edge
  node after integration; this is what gets fed to the solver as a
  :class:`~femtoolkit.analysis.loads.NodalLoad`.

Both :class:`~femtoolkit.mesh.cst_element.CSTElement2D` and
:class:`~femtoolkit.mesh.quad_element.QuadElement2D` have straight edges
between two corner nodes (neither has midside nodes), so a single
2-node linear edge formulation serves both element types.

**Equivalent nodal force formula:**

.. code-block:: text

    fe = integral( N^T @ t ) * thickness ds

where ``N`` is the 2-node edge shape-function matrix, ``t = [tx, ty]``
is the traction vector, and ``ds`` is the boundary-length differential.
Mapped to the natural edge coordinate ``xi in [-1, 1]`` via the edge
Jacobian ``ds = (L/2) dxi`` (``L`` = edge length), this becomes a
1D integral evaluated by 2-point Gauss quadrature -- exact for the
linear (in ``xi``) integrand a straight edge with linear shape functions
and constant traction produces.

For a **constant** traction, the total equivalent force
(``F = traction * L * thickness``) splits evenly between the two edge
nodes -- verified directly in ``tests/test_continuum_edge.py``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from femtoolkit.exceptions import DegenerateElementError, ValidationError

GAUSS_1D_2POINT: tuple[tuple[float, float], tuple[float, float]] = (
    (-1.0 / 3.0**0.5, 1.0),
    (1.0 / 3.0**0.5, 1.0),
)
"""The two points of the 2-point 1D Gauss-Legendre rule on ``[-1, 1]``.

Each ``(xi, weight)`` pair uses the standard abscissa ``+/- 1/sqrt(3)``,
exact for polynomials up to degree 3 -- more than enough for the linear
edge shape functions used here.
"""

MIN_EDGE_LENGTH: float = 1e-12
"""Minimum acceptable edge length, in meters.

Analogous to :data:`femtoolkit.continuum.geometry.MIN_TRIANGLE_AREA`.
"""


def edge_shape_functions(xi: float) -> tuple[float, float]:
    """Evaluate the two linear edge shape functions at natural coordinate ``xi``.

    .. code-block:: text

        N1(xi) = (1 - xi) / 2      (1 at xi=-1, 0 at xi=1)
        N2(xi) = (1 + xi) / 2      (0 at xi=-1, 1 at xi=1)

    Args:
        xi: Natural edge coordinate, expected in ``[-1, 1]``.

    Returns:
        ``(N1, N2)`` evaluated at ``xi``.
    """
    return (1.0 - xi) / 2.0, (1.0 + xi) / 2.0


def edge_equivalent_nodal_force(
    node_a: Sequence[float],
    node_b: Sequence[float],
    traction: Sequence[float],
    thickness: float,
) -> np.ndarray:
    """Compute the equivalent nodal force vector for a traction along a straight edge.

    Args:
        node_a: ``(x, y)`` coordinates of the edge's first node, in meters.
        node_b: ``(x, y)`` coordinates of the edge's second node, in meters.
        traction: ``(tx, ty)`` traction vector, in pascals (assumed
            constant along the edge).
        thickness: Element thickness, in meters. Must be positive.

    Returns:
        A length-4 NumPy array ``[fx1, fy1, fx2, fy2]``, in newtons -- the
        equivalent nodal forces at ``node_a`` and ``node_b`` respectively.

    Raises:
        ValidationError: If ``thickness`` is not a positive, finite number.
        DegenerateElementError: If the edge has (near) zero length.

    Example:
        >>> edge_equivalent_nodal_force((0.0, 0.0), (1.0, 0.0), (0.0, 1000.0), 0.1)
        array([ 0.,  50.,  0.,  50.])
    """
    if not math.isfinite(thickness) or thickness <= 0:
        raise ValidationError(f"thickness must be positive, got {thickness}.")

    xa, ya = node_a
    xb, yb = node_b
    length = math.hypot(xb - xa, yb - ya)
    if not math.isfinite(length) or length < MIN_EDGE_LENGTH:
        raise DegenerateElementError(
            f"Edge from ({xa},{ya}) to ({xb},{yb}) has length {length}, which is degenerate."
        )

    tx, ty = traction
    jacobian = length / 2.0  # ds/dxi for a straight edge

    force = np.zeros(4)
    for xi, weight in GAUSS_1D_2POINT:
        n1, n2 = edge_shape_functions(xi)
        force += weight * thickness * jacobian * np.array([n1 * tx, n1 * ty, n2 * tx, n2 * ty])
    return force
