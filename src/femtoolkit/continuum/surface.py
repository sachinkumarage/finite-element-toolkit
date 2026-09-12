"""Reusable geometry and integration utilities for element *surfaces* (Version 21).

Convection and radiation boundary conditions act on a **surface** --
a 2D element's edge, or a 3D element's face -- not on the element's
whole volume/area. Computing their contribution to the finite element
equations therefore needs, for each supported surface type, an area (or
length) and a way to integrate a shape-function product over it. This
module gathers exactly the *new* geometric math that requires (surface
face tables and the one genuinely new integration routine, for HEX8's
bilinear quad face embedded in 3D space); everywhere a straight-line
edge or a flat triangular face is involved, this module deliberately
reuses existing, already-tested machinery instead of re-deriving it:

* A straight 2D edge (a CST/Q4 element boundary) is mathematically
  identical to a 2-node bar, so its convection matrix/load vector reuse
  :mod:`femtoolkit.continuum.edge`'s existing closed-form line
  integration (see :func:`~femtoolkit.continuum.edge.edge_consistent_conductance_matrix`
  and :func:`~femtoolkit.continuum.edge.edge_equivalent_nodal_force`).
* A flat triangular face (a TET4 face) has the same linear shape
  functions as a CST element, so its area and ``integral(N^T N)``
  matrix reuse :func:`~femtoolkit.continuum.mass.triangle_consistent_mass_matrix`
  directly (passing ``thickness=1.0``, since a face is a bare area, not
  a volume).

Only a HEX8 face -- a **bilinear quadrilateral surface embedded in 3D
space**, generally non-planar and not axis-aligned -- has no existing
analogue in this toolkit: :func:`~femtoolkit.continuum.mass.quad_consistent_mass_matrix`
assumes its quad lies flat in the ``xy``-plane (a 2D in-plane Jacobian),
which does not apply to an arbitrarily oriented 3D face. This module
adds exactly that one new piece of math, reusing the *shape functions*
(:func:`~femtoolkit.continuum.shape_functions.quad_shape_functions`) and
*derivatives* already defined for the mechanical Q4 element, only
introducing a new **surface Jacobian**:

.. code-block:: text

    dA = | (dX/dxi) x (dX/deta) | dxi deta

the standard area element of a parametric surface ``X(xi, eta)`` in 3D:
the cross product of the two tangent vectors (each a weighted sum of the
face's node coordinates, weighted by the shape-function derivatives)
gives a vector normal to the surface whose magnitude is exactly the
local area-scaling factor -- the 3D-surface generalization of the
familiar 2D Jacobian determinant.
"""

from __future__ import annotations

import numpy as np

from femtoolkit.continuum.gauss import GAUSS_2X2_POINTS
from femtoolkit.continuum.mass import triangle_consistent_mass_matrix
from femtoolkit.continuum.shape_functions import (
    quad_shape_function_derivatives,
    quad_shape_functions,
)
from femtoolkit.exceptions import DegenerateElementError

MIN_FACE_AREA: float = 1e-12
"""Minimum acceptable face area, in square meters (the 3D-surface analogue of
:data:`~femtoolkit.continuum.geometry.MIN_TRIANGLE_AREA`)."""

TET4_FACES: tuple[tuple[int, int, int], ...] = ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))
"""The four triangular faces of a TET4 element, as local node indices.

Since only unsigned face *area* is needed for convection/radiation
(neither depends on an outward-normal direction, only on the surface's
magnitude), the winding of each triple does not need to be outward-
consistent -- unlike a mechanical traction, which would need one."""

HEX8_FACES: tuple[tuple[int, int, int, int], ...] = (
    (0, 1, 2, 3),  # bottom, zeta = -1
    (4, 5, 6, 7),  # top, zeta = +1
    (0, 1, 5, 4),  # front, eta = -1
    (2, 3, 7, 6),  # back, eta = +1
    (1, 2, 6, 5),  # right, xi = +1
    (0, 3, 7, 4),  # left, xi = -1
)
"""The six quadrilateral faces of a HEX8 element, as local node indices --
each a non-self-intersecting ring around the face, matching
:data:`~femtoolkit.continuum.shape_functions._HEX8_NATURAL_COORDS`'s node
ordering convention. As with :data:`TET4_FACES`, the winding is not
required to be outward-consistent for this module's purposes."""


def triangle_face_area(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> float:
    """Return a triangular face's area from its three 3D vertex coordinates.

    Args:
        p0: First vertex, ``(x, y, z)``.
        p1: Second vertex, ``(x, y, z)``.
        p2: Third vertex, ``(x, y, z)``.

    Returns:
        The (always positive) triangle area, in square meters.

    Raises:
        DegenerateElementError: If the three points are collinear or
            coincide (zero or near-zero area).
    """
    cross = np.cross(np.asarray(p1) - np.asarray(p0), np.asarray(p2) - np.asarray(p0))
    area = 0.5 * float(np.linalg.norm(cross))
    if area < MIN_FACE_AREA:
        raise DegenerateElementError(f"Triangular face has area {area}, which is degenerate.")
    return area


def triangle_face_conductance_matrix(
    p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, coefficient: float
) -> np.ndarray:
    """Return ``coefficient * integral(N^T N) dA`` over a flat triangular face.

    Reuses :func:`~femtoolkit.continuum.mass.triangle_consistent_mass_matrix`
    with ``thickness=1.0`` (a face is a bare area, not a volume) and
    ``density=coefficient``, then extracts the un-replicated 3x3 scalar
    block -- exactly the same slicing technique
    :mod:`femtoolkit.thermal.thermal_elements` uses for the CST capacity
    matrix.

    Args:
        p0: First vertex, ``(x, y, z)``.
        p1: Second vertex, ``(x, y, z)``.
        p2: Third vertex, ``(x, y, z)``.
        coefficient: The (positive) scalar multiplying the integral --
            a convection coefficient ``h``, for example.

    Returns:
        A symmetric 3x3 NumPy array.
    """
    area = triangle_face_area(p0, p1, p2)
    mechanical_mass = triangle_consistent_mass_matrix(density=coefficient, area=area, thickness=1.0)
    return mechanical_mass[0::2, 0::2]


def triangle_face_load_vector(
    p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, coefficient: float
) -> np.ndarray:
    """Return ``coefficient * integral(N^T) dA`` over a flat triangular face.

    For a linear triangle, ``integral(Ni) dA = A/3`` for every node
    (the same closed-form identity used throughout this project's CST
    heat-generation and mass-matrix formulas), so the total
    ``coefficient * area`` splits evenly across the three nodes.

    Args:
        p0: First vertex, ``(x, y, z)``.
        p1: Second vertex, ``(x, y, z)``.
        p2: Third vertex, ``(x, y, z)``.
        coefficient: The scalar multiplying the integral -- typically
            ``h * T_infinity`` for a convective load, or an
            emissivity/Stefan-Boltzmann term for radiation.

    Returns:
        A length-3 NumPy array, one contribution per vertex.
    """
    area = triangle_face_area(p0, p1, p2)
    return np.full(3, coefficient * area / 3.0)


def _quad_face_tangents(
    coords: np.ndarray, xi: float, eta: float
) -> tuple[np.ndarray, np.ndarray]:
    dn_dxi, dn_deta = quad_shape_function_derivatives(xi, eta)
    tangent_xi = sum(d * coords[i] for i, d in enumerate(dn_dxi))
    tangent_eta = sum(d * coords[i] for i, d in enumerate(dn_deta))
    return np.asarray(tangent_xi, dtype=float), np.asarray(tangent_eta, dtype=float)


def quad_face_surface_jacobian(coords: np.ndarray, xi: float, eta: float) -> float:
    """Return the surface Jacobian ``|dX/dxi x dX/deta|`` at one natural-coordinate point.

    See the module docstring for the derivation. ``coords`` must be
    ordered per the isoparametric convention
    :func:`~femtoolkit.continuum.shape_functions.quad_shape_functions`
    uses (matching one entry of :data:`HEX8_FACES`).

    Args:
        coords: The face's four node coordinates, shape ``(4, 3)``.
        xi: First natural coordinate, in ``[-1, 1]``.
        eta: Second natural coordinate, in ``[-1, 1]``.

    Returns:
        The (always non-negative) surface Jacobian at ``(xi, eta)``.
    """
    tangent_xi, tangent_eta = _quad_face_tangents(np.asarray(coords, dtype=float), xi, eta)
    return float(np.linalg.norm(np.cross(tangent_xi, tangent_eta)))


def quad_face_area(coords: np.ndarray) -> float:
    """Return a HEX8 quad face's area, by 2x2 Gauss quadrature of the surface Jacobian.

    Args:
        coords: The face's four node coordinates, shape ``(4, 3)``.

    Returns:
        The (always positive) face area, in square meters.

    Raises:
        DegenerateElementError: If the resulting area is zero or near-zero.
    """
    area = sum(
        point.weight * quad_face_surface_jacobian(coords, point.xi, point.eta)
        for point in GAUSS_2X2_POINTS
    )
    if area < MIN_FACE_AREA:
        raise DegenerateElementError(f"Quadrilateral face has area {area}, which is degenerate.")
    return area


def quad_face_consistent_matrix(coords: np.ndarray, coefficient: float) -> np.ndarray:
    """Return ``coefficient * integral(N^T N) dA`` over a HEX8 quad face, by 2x2 Gauss quadrature.

    The direct surface analogue of
    :func:`~femtoolkit.continuum.mass.quad_consistent_mass_matrix`, with
    the in-plane 2D Jacobian determinant replaced by the 3D
    :func:`quad_face_surface_jacobian`.

    Args:
        coords: The face's four node coordinates, shape ``(4, 3)``.
        coefficient: The scalar multiplying the integral.

    Returns:
        A symmetric 4x4 NumPy array.
    """
    coords = np.asarray(coords, dtype=float)
    matrix = np.zeros((4, 4))
    for point in GAUSS_2X2_POINTS:
        n_values = np.array(quad_shape_functions(point.xi, point.eta))
        jacobian = quad_face_surface_jacobian(coords, point.xi, point.eta)
        matrix += point.weight * coefficient * np.outer(n_values, n_values) * jacobian
    return matrix


def quad_face_load_vector(coords: np.ndarray, coefficient: float) -> np.ndarray:
    """Return ``coefficient * integral(N^T) dA`` over a HEX8 quad face, by 2x2 Gauss quadrature.

    Args:
        coords: The face's four node coordinates, shape ``(4, 3)``.
        coefficient: The scalar multiplying the integral.

    Returns:
        A length-4 NumPy array.
    """
    coords = np.asarray(coords, dtype=float)
    vector = np.zeros(4)
    for point in GAUSS_2X2_POINTS:
        n_values = np.array(quad_shape_functions(point.xi, point.eta))
        jacobian = quad_face_surface_jacobian(coords, point.xi, point.eta)
        vector += point.weight * coefficient * n_values * jacobian
    return vector
