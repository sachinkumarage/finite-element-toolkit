"""Element mass matrices for 2D continuum elements: consistent and lumped.

Dynamic analysis needs a way to relate acceleration to inertial force,
just as the stiffness matrix relates displacement to elastic force. The
**consistent mass matrix** is derived from the same virtual-work
principle as the stiffness matrix, but through kinetic rather than
elastic energy -- using the *same* shape functions that interpolate
displacement to interpolate (and weight) the distributed inertia:

.. code-block:: text

    Me = integral( rho * N^T * N * t ) dA

where ``rho`` is the material density, ``N`` is the element's shape
function matrix (values, not derivatives -- contrast with the
strain-displacement matrix ``B`` used for stiffness), and ``t`` is the
thickness. Because ``N`` is exactly the same interpolation used for
displacement, this integral is *consistent* with the element's assumed
displacement field -- an accurate statement of the element's kinetic
energy, but a *full* (not diagonal) matrix: every DOF is inertially
coupled to every other DOF in the element.

**Lumped mass** instead distributes the element's total mass directly
onto its DOFs as if concentrated there, producing a *diagonal* matrix.
This module uses the standard **row-sum lumping** technique (see
:func:`lumped_mass_matrix`): each diagonal entry is the sum of its row in
the consistent mass matrix. Row-sum lumping is mass-conserving by
construction (see the function's docstring) and needs no judgment calls
about how to redistribute mass -- unlike, say, HRZ lumping, which scales
diagonal terms to match the total mass and is out of scope here. Lumped
mass is a computational convenience (trivially invertible, and required
by some explicit time-integration schemes not implemented in this
version) at the cost of physical accuracy relative to the consistent
form -- it does not, in general, reproduce the same natural frequencies.

For the **CST element** (Version 6), the shape functions are the
triangle's area coordinates, whose products integrate in closed form via
the standard triangle area-coordinate integral identity:

.. code-block:: text

    integral( Li^a * Lj^b * Lk^c ) dA = 2A * a! b! c! / (a + b + c + 2)!

giving the well-known consistent mass matrix (for nodal displacements
ordered ``[u1, v1, u2, v2, u3, v3]``):

.. code-block:: text

    Me = (rho * A * t / 12) * [[2, 1, 1], [1, 2, 1], [1, 1, 2]] (kron) I2

For the **Q4 element** (Version 7), the bilinear shape functions have no
such closed form on a general quadrilateral, so the same 2x2 Gauss
quadrature used for the stiffness matrix (see
:mod:`femtoolkit.continuum.gauss`) is reused here.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from femtoolkit.continuum.gauss import GAUSS_2X2_POINTS, GAUSS_2X2X2_POINTS
from femtoolkit.continuum.jacobian import (
    physical_shape_function_derivatives,
    physical_shape_function_derivatives_3d,
)
from femtoolkit.continuum.shape_functions import (
    hex8_shape_function_derivatives,
    hex8_shape_functions,
    quad_shape_function_derivatives,
    quad_shape_functions,
)
from femtoolkit.exceptions import ValidationError

_TRIANGLE_CONSISTENT_MASS_SHAPE = np.array(
    [
        [2.0, 1.0, 1.0],
        [1.0, 2.0, 1.0],
        [1.0, 1.0, 2.0],
    ]
)

_TETRAHEDRON_CONSISTENT_MASS_SHAPE = np.array(
    [
        [2.0, 1.0, 1.0, 1.0],
        [1.0, 2.0, 1.0, 1.0],
        [1.0, 1.0, 2.0, 1.0],
        [1.0, 1.0, 1.0, 2.0],
    ]
)
"""The TET4 analogue of :data:`_TRIANGLE_CONSISTENT_MASS_SHAPE`.

Derived from the closed-form volume-coordinate integral identity
``integral(Li^a Lj^b Lk^c Ll^d) dV = 6V * a!b!c!d! / (a+b+c+d+3)!``: the
diagonal terms (``i == j``) integrate to ``V/10`` and the off-diagonal
terms (``i != j``) to ``V/20``, giving this ``2``/``1`` coefficient
pattern once both are expressed as a multiple of ``V/20``.
"""


def _validate_positive_finite(**values: float) -> None:
    for name, value in values.items():
        if not math.isfinite(value) or value <= 0:
            raise ValidationError(f"{name} must be positive, got {value}.")


def triangle_consistent_mass_matrix(density: float, area: float, thickness: float) -> np.ndarray:
    """Compute the closed-form consistent mass matrix of a CST element.

    .. code-block:: text

        Me = (rho * A * t / 12) * [[2,1,1],[1,2,1],[1,1,2]] (kron) I2

    Args:
        density: Material density, in kg/m^3. Must be positive.
        area: Element's physical (always positive) area, in square meters.
            Must be positive.
        thickness: Element thickness, in meters. Must be positive.

    Returns:
        A 6x6 NumPy array, the symmetric consistent mass matrix for
        nodal DOFs ordered ``[u1, v1, u2, v2, u3, v3]``.

    Raises:
        ValidationError: If ``density``, ``area``, or ``thickness`` is
            not a positive, finite number.

    Example:
        >>> m = triangle_consistent_mass_matrix(density=1000.0, area=0.5, thickness=0.01)
        >>> m.shape
        (6, 6)
        >>> float(m.sum()) / 2  # total element mass, summed over one direction
        5.0
    """
    _validate_positive_finite(density=density, area=area, thickness=thickness)

    factor = density * area * thickness / 12.0
    return factor * np.kron(_TRIANGLE_CONSISTENT_MASS_SHAPE, np.eye(2))


def quad_shape_function_matrix(n_values: Sequence[float]) -> np.ndarray:
    """Build the 2x8 shape function matrix ``N`` for a Q4 element at one point.

    Analogous to :func:`~femtoolkit.continuum.strain.quad_strain_displacement_matrix`,
    but placing shape function *values* (not derivatives) into the
    block-diagonal X/Y layout:

    .. code-block:: text

        N =
        [ N1   0    N2   0    N3   0    N4   0  ]
        [ 0    N1   0    N2   0    N3   0    N4 ]

    Args:
        n_values: The four shape function values ``(N1, N2, N3, N4)`` at
            one natural-coordinate point (see
            :func:`~femtoolkit.continuum.shape_functions.quad_shape_functions`).

    Returns:
        A 2x8 NumPy array.
    """
    n_matrix = np.zeros((2, 8))
    for i, n_value in enumerate(n_values):
        n_matrix[0, 2 * i] = n_value
        n_matrix[1, 2 * i + 1] = n_value
    return n_matrix


def quad_consistent_mass_matrix(
    x_coords: Sequence[float],
    y_coords: Sequence[float],
    density: float,
    thickness: float,
) -> np.ndarray:
    """Compute the consistent mass matrix of a Q4 element by 2x2 Gauss quadrature.

    .. code-block:: text

        Me = sum over the 4 Gauss points of:
             weight * rho * t * N(xi,eta)^T * N(xi,eta) * det(J(xi,eta))

    Args:
        x_coords: The element's four node X coordinates, in meters,
            ordered per the isoparametric convention (see
            :func:`~femtoolkit.continuum.shape_functions.quad_shape_functions`).
        y_coords: The element's four node Y coordinates, in meters.
        density: Material density, in kg/m^3. Must be positive.
        thickness: Element thickness, in meters. Must be positive.

    Returns:
        An 8x8 NumPy array, the symmetric consistent mass matrix for
        nodal DOFs ordered ``[u1, v1, u2, v2, u3, v3, u4, v4]``.

    Raises:
        ValidationError: If ``density`` or ``thickness`` is not a
            positive, finite number.
        DegenerateElementError: If the Jacobian determinant is not
            positive at any of the four Gauss points.
    """
    _validate_positive_finite(density=density, thickness=thickness)

    mass = np.zeros((8, 8))
    for point in GAUSS_2X2_POINTS:
        n_values = quad_shape_functions(point.xi, point.eta)
        dn_dxi, dn_deta = quad_shape_function_derivatives(point.xi, point.eta)
        _, _, det_j = physical_shape_function_derivatives(dn_dxi, dn_deta, x_coords, y_coords)
        n_matrix = quad_shape_function_matrix(n_values)
        mass += point.weight * density * thickness * (n_matrix.T @ n_matrix) * det_j

    return mass


def lumped_mass_matrix(consistent_mass: np.ndarray) -> np.ndarray:
    """Diagonalize a consistent mass matrix by row-sum ("lumped") lumping.

    Each diagonal entry is the sum of its row in ``consistent_mass``, and
    every off-diagonal entry is zero. This conserves total mass exactly:
    because the shape functions of every element in this toolkit satisfy
    the partition-of-unity property (``sum(Ni) = 1`` everywhere -- see
    :mod:`femtoolkit.continuum.shape_functions`), the X-direction (or
    Y-direction) block of the consistent mass matrix sums, in total, to
    exactly the element's physical mass, ``rho * A * t``; row-sum lumping
    only redistributes that same total onto the diagonal, never changing
    it.

    Args:
        consistent_mass: A symmetric consistent mass matrix (e.g. from
            :func:`triangle_consistent_mass_matrix` or
            :func:`quad_consistent_mass_matrix`).

    Returns:
        A diagonal NumPy array of the same shape as ``consistent_mass``.

    Example:
        >>> m = triangle_consistent_mass_matrix(density=1000.0, area=0.5, thickness=0.01)
        >>> lumped = lumped_mass_matrix(m)
        >>> bool((lumped.sum(axis=1) == np.diag(lumped)).all())
        True
    """
    return np.diag(consistent_mass.sum(axis=1))


def tetrahedron_consistent_mass_matrix(density: float, volume: float) -> np.ndarray:
    """Compute the closed-form consistent mass matrix of a TET4 element.

    .. code-block:: text

        Me = (rho * V / 20) * [[2,1,1,1],[1,2,1,1],[1,1,2,1],[1,1,1,2]] (kron) I3

    See :data:`_TETRAHEDRON_CONSISTENT_MASS_SHAPE` for the derivation.
    Unlike the 2D triangle case, there is no ``thickness`` factor -- a
    TET4 element already represents a genuine 3D volume.

    Args:
        density: Material density, in kg/m^3. Must be positive.
        volume: Element's physical (always positive) volume, in cubic
            meters. Must be positive.

    Returns:
        A 12x12 NumPy array, the symmetric consistent mass matrix for
        nodal DOFs ordered ``[u1, v1, w1, u2, v2, w2, u3, v3, w3, u4, v4, w4]``.

    Raises:
        ValidationError: If ``density`` or ``volume`` is not a positive,
            finite number.

    Example:
        >>> m = tetrahedron_consistent_mass_matrix(density=1000.0, volume=1.0)
        >>> m.shape
        (12, 12)
        >>> float(m.sum()) / 3  # total element mass, summed over one direction
        1000.0
    """
    _validate_positive_finite(density=density, volume=volume)

    factor = density * volume / 20.0
    return factor * np.kron(_TETRAHEDRON_CONSISTENT_MASS_SHAPE, np.eye(3))


def hex8_shape_function_matrix(n_values: Sequence[float]) -> np.ndarray:
    """Build the 3x24 shape function matrix ``N`` for a HEX8 element at one point.

    The 3D analogue of :func:`quad_shape_function_matrix`:

    .. code-block:: text

        N =
        [ N1  0   0   N2  0   0   ...  N8  0   0  ]
        [ 0   N1  0   0   N2  0   ...  0   N8  0  ]
        [ 0   0   N1  0   0   N2  ...  0   0   N8 ]

    Args:
        n_values: The eight shape function values ``(N1, ..., N8)`` at one
            natural-coordinate point (see
            :func:`~femtoolkit.continuum.shape_functions.hex8_shape_functions`).

    Returns:
        A 3x24 NumPy array.
    """
    n_matrix = np.zeros((3, 24))
    for i, n_value in enumerate(n_values):
        n_matrix[0, 3 * i] = n_value
        n_matrix[1, 3 * i + 1] = n_value
        n_matrix[2, 3 * i + 2] = n_value
    return n_matrix


def hex8_consistent_mass_matrix(
    x_coords: Sequence[float],
    y_coords: Sequence[float],
    z_coords: Sequence[float],
    density: float,
) -> np.ndarray:
    """Compute the consistent mass matrix of a HEX8 element by 2x2x2 Gauss quadrature.

    .. code-block:: text

        Me = sum over the 8 Gauss points of:
             weight * rho * N(xi,eta,zeta)^T * N(xi,eta,zeta) * det(J(xi,eta,zeta))

    The 3D analogue of :func:`quad_consistent_mass_matrix`; unlike the 2D
    Q4 case, there is no ``thickness`` factor.

    Args:
        x_coords: The element's eight node X coordinates, in meters,
            ordered per the isoparametric convention (see
            :func:`~femtoolkit.continuum.shape_functions.hex8_shape_functions`).
        y_coords: The element's eight node Y coordinates, in meters.
        z_coords: The element's eight node Z coordinates, in meters.
        density: Material density, in kg/m^3. Must be positive.

    Returns:
        A 24x24 NumPy array, the symmetric consistent mass matrix for
        nodal DOFs ordered ``[u1, v1, w1, ..., u8, v8, w8]``.

    Raises:
        ValidationError: If ``density`` is not a positive, finite number.
        DegenerateElementError: If the Jacobian determinant is not
            positive at any of the eight Gauss points.
    """
    _validate_positive_finite(density=density)

    mass = np.zeros((24, 24))
    for point in GAUSS_2X2X2_POINTS:
        n_values = hex8_shape_functions(point.xi, point.eta, point.zeta)
        dn_dxi, dn_deta, dn_dzeta = hex8_shape_function_derivatives(point.xi, point.eta, point.zeta)
        _, _, _, det_j = physical_shape_function_derivatives_3d(
            dn_dxi, dn_deta, dn_dzeta, x_coords, y_coords, z_coords
        )
        n_matrix = hex8_shape_function_matrix(n_values)
        mass += point.weight * density * (n_matrix.T @ n_matrix) * det_j

    return mass
