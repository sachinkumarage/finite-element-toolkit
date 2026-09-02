"""Isoparametric Jacobian for the 4-node quadrilateral (Q4) element.

An isoparametric element uses the *same* shape functions to interpolate
both geometry and displacement:

.. code-block:: text

    x(xi, eta) = N1*x1 + N2*x2 + N3*x3 + N4*x4
    y(xi, eta) = N1*y1 + N2*y2 + N3*y3 + N4*y4

The **Jacobian matrix** relates derivatives in the natural coordinate
system ``(xi, eta)`` (where the shape functions are simple, see
:mod:`femtoolkit.continuum.shape_functions`) to derivatives in the
physical coordinate system ``(x, y)`` (where strain is actually defined):

.. code-block:: text

    J =
    [ dx/dxi   dy/dxi  ]
    [ dx/deta  dy/deta ]

    dx/dxi  = sum(dNi/dxi  * xi_coord)   (and similarly for the other three entries)

Its determinant, ``det(J)``, is the area-scaling factor between the
natural-coordinate square and the physical element -- used both to reject
degenerate geometry and to convert the area integral for the stiffness
matrix into a natural-coordinate integral (see
:func:`~femtoolkit.analysis.stiffness.quad_element_stiffness`).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from femtoolkit.exceptions import DegenerateElementError

MIN_JACOBIAN_DETERMINANT: float = 1e-12
"""Minimum acceptable Jacobian determinant, in square meters.

Guards against degenerate quadrilaterals (collinear, self-intersecting,
or inverted node ordering), for which the isoparametric mapping is not
well-defined. Analogous to
:data:`femtoolkit.continuum.geometry.MIN_TRIANGLE_AREA`.
"""


def jacobian_matrix(
    dn_dxi: Sequence[float],
    dn_deta: Sequence[float],
    x_coords: Sequence[float],
    y_coords: Sequence[float],
) -> np.ndarray:
    """Build the 2x2 isoparametric Jacobian matrix at one natural-coordinate point.

    Args:
        dn_dxi: The four shape functions' ``dNi/dxi`` derivatives at this
            point (see
            :func:`~femtoolkit.continuum.shape_functions.quad_shape_function_derivatives`).
        dn_deta: The four shape functions' ``dNi/deta`` derivatives at
            this point.
        x_coords: The element's four node X coordinates, in meters,
            ordered to match ``dn_dxi``/``dn_deta``.
        y_coords: The element's four node Y coordinates, in meters.

    Returns:
        A 2x2 NumPy array, the Jacobian matrix.
    """
    dn_dxi_arr = np.asarray(dn_dxi, dtype=float)
    dn_deta_arr = np.asarray(dn_deta, dtype=float)
    x = np.asarray(x_coords, dtype=float)
    y = np.asarray(y_coords, dtype=float)

    return np.array(
        [
            [np.dot(dn_dxi_arr, x), np.dot(dn_dxi_arr, y)],
            [np.dot(dn_deta_arr, x), np.dot(dn_deta_arr, y)],
        ]
    )


def jacobian_determinant(jacobian: np.ndarray) -> float:
    """Compute the determinant of a 2x2 Jacobian matrix.

    Args:
        jacobian: A 2x2 Jacobian matrix (see :func:`jacobian_matrix`).

    Returns:
        ``det(J) = J[0,0]*J[1,1] - J[0,1]*J[1,0]``.
    """
    return float(jacobian[0, 0] * jacobian[1, 1] - jacobian[0, 1] * jacobian[1, 0])


def inverse_jacobian(jacobian: np.ndarray) -> np.ndarray:
    """Compute the inverse of a 2x2 Jacobian matrix directly (no general solver).

    .. code-block:: text

        J^-1 = 1/det(J) * [ J[1,1]  -J[0,1] ]
                          [ -J[1,0]  J[0,0] ]

    Args:
        jacobian: A 2x2 Jacobian matrix.

    Returns:
        A 2x2 NumPy array, the inverse Jacobian.

    Raises:
        DegenerateElementError: If ``det(J)`` is not positive and finite
            (see :data:`MIN_JACOBIAN_DETERMINANT`).
    """
    determinant = jacobian_determinant(jacobian)
    if not math.isfinite(determinant) or determinant < MIN_JACOBIAN_DETERMINANT:
        raise DegenerateElementError(
            f"Jacobian determinant {determinant} is not positive; the element geometry "
            "is degenerate, self-intersecting, or has inverted (clockwise) node order."
        )

    return (1.0 / determinant) * np.array(
        [
            [jacobian[1, 1], -jacobian[0, 1]],
            [-jacobian[1, 0], jacobian[0, 0]],
        ]
    )


def physical_shape_function_derivatives(
    dn_dxi: Sequence[float],
    dn_deta: Sequence[float],
    x_coords: Sequence[float],
    y_coords: Sequence[float],
) -> tuple[np.ndarray, np.ndarray, float]:
    """Convert natural-coordinate shape function derivatives to physical ones.

    Via the chain rule, ``[dNi/dxi; dNi/deta] = J @ [dNi/dx; dNi/dy]``, so
    the physical derivatives needed by the strain-displacement matrix
    (:func:`~femtoolkit.continuum.strain.quad_strain_displacement_matrix`)
    are recovered as ``[dNi/dx; dNi/dy] = J^-1 @ [dNi/dxi; dNi/deta]``.

    Args:
        dn_dxi: The four shape functions' ``dNi/dxi`` derivatives at this
            natural-coordinate point.
        dn_deta: The four shape functions' ``dNi/deta`` derivatives at
            this point.
        x_coords: The element's four node X coordinates, in meters.
        y_coords: The element's four node Y coordinates, in meters.

    Returns:
        ``(dN_dx, dN_dy, det_J)``: two length-4 NumPy arrays of physical
        derivatives, and the Jacobian determinant at this point (needed
        by the caller to weight the numerical integration).

    Raises:
        DegenerateElementError: If the Jacobian determinant is not
            positive at this point.
    """
    jacobian = jacobian_matrix(dn_dxi, dn_deta, x_coords, y_coords)
    det_j = jacobian_determinant(jacobian)
    j_inv = inverse_jacobian(jacobian)

    natural_derivatives = np.array([dn_dxi, dn_deta], dtype=float)
    physical_derivatives = j_inv @ natural_derivatives
    return physical_derivatives[0], physical_derivatives[1], det_j


MIN_JACOBIAN_DETERMINANT_3D: float = 1e-12
"""Minimum acceptable 3D Jacobian determinant, in cubic meters.

The 3D analogue of :data:`MIN_JACOBIAN_DETERMINANT`, guarding TET4 and
HEX8 against degenerate, self-intersecting, or inverted-node-order
geometry, for which the isoparametric mapping is not well-defined.
"""


def jacobian_matrix_3d(
    dn_dxi: Sequence[float],
    dn_deta: Sequence[float],
    dn_dzeta: Sequence[float],
    x_coords: Sequence[float],
    y_coords: Sequence[float],
    z_coords: Sequence[float],
) -> np.ndarray:
    """Build the 3x3 isoparametric Jacobian matrix at one natural-coordinate point.

    The 3D analogue of :func:`jacobian_matrix`:

    .. code-block:: text

        J =
        [ dx/dxi    dy/dxi    dz/dxi   ]
        [ dx/deta   dy/deta   dz/deta  ]
        [ dx/dzeta  dy/dzeta  dz/dzeta ]

    Used by both TET4 (where ``dn_dxi``/``dn_deta``/``dn_dzeta`` are
    constant, see :func:`~femtoolkit.continuum.shape_functions.tet4_shape_function_derivatives`,
    giving a single, element-constant Jacobian) and HEX8 (evaluated fresh
    at each of the 8 Gauss points, see
    :func:`~femtoolkit.continuum.shape_functions.hex8_shape_function_derivatives`).

    Args:
        dn_dxi: Each node's ``dNi/dxi`` derivative at this point.
        dn_deta: Each node's ``dNi/deta`` derivative at this point.
        dn_dzeta: Each node's ``dNi/dzeta`` derivative at this point.
        x_coords: The element's node X coordinates, in meters, ordered to
            match ``dn_dxi``/``dn_deta``/``dn_dzeta``.
        y_coords: The element's node Y coordinates, in meters.
        z_coords: The element's node Z coordinates, in meters.

    Returns:
        A 3x3 NumPy array, the Jacobian matrix.
    """
    dn_dxi_arr = np.asarray(dn_dxi, dtype=float)
    dn_deta_arr = np.asarray(dn_deta, dtype=float)
    dn_dzeta_arr = np.asarray(dn_dzeta, dtype=float)
    x = np.asarray(x_coords, dtype=float)
    y = np.asarray(y_coords, dtype=float)
    z = np.asarray(z_coords, dtype=float)

    return np.array(
        [
            [np.dot(dn_dxi_arr, x), np.dot(dn_dxi_arr, y), np.dot(dn_dxi_arr, z)],
            [np.dot(dn_deta_arr, x), np.dot(dn_deta_arr, y), np.dot(dn_deta_arr, z)],
            [np.dot(dn_dzeta_arr, x), np.dot(dn_dzeta_arr, y), np.dot(dn_dzeta_arr, z)],
        ]
    )


def jacobian_determinant_3d(jacobian: np.ndarray) -> float:
    """Compute the determinant of a 3x3 Jacobian matrix via :func:`numpy.linalg.det`.

    A general (not hand-expanded cofactor) determinant routine is used
    deliberately for numerical robustness on the 3x3 case -- unlike the
    2x2 :func:`jacobian_determinant`, where the closed-form expansion is
    both simpler and just as accurate.

    Args:
        jacobian: A 3x3 Jacobian matrix (see :func:`jacobian_matrix_3d`).

    Returns:
        ``det(J)``.
    """
    return float(np.linalg.det(jacobian))


def inverse_jacobian_3d(jacobian: np.ndarray) -> np.ndarray:
    """Compute the inverse of a 3x3 Jacobian matrix via :func:`numpy.linalg.inv`.

    Args:
        jacobian: A 3x3 Jacobian matrix.

    Returns:
        A 3x3 NumPy array, the inverse Jacobian.

    Raises:
        DegenerateElementError: If ``det(J)`` is not positive and finite
            (see :data:`MIN_JACOBIAN_DETERMINANT_3D`).
    """
    determinant = jacobian_determinant_3d(jacobian)
    if not math.isfinite(determinant) or determinant < MIN_JACOBIAN_DETERMINANT_3D:
        raise DegenerateElementError(
            f"3D Jacobian determinant {determinant} is not positive; the element "
            "geometry is degenerate, self-intersecting, or has inverted node order."
        )

    return np.linalg.inv(jacobian)


def physical_shape_function_derivatives_3d(
    dn_dxi: Sequence[float],
    dn_deta: Sequence[float],
    dn_dzeta: Sequence[float],
    x_coords: Sequence[float],
    y_coords: Sequence[float],
    z_coords: Sequence[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Convert natural-coordinate shape function derivatives to physical ones, in 3D.

    The 3D analogue of :func:`physical_shape_function_derivatives`: via
    the chain rule, ``[dNi/dxi; dNi/deta; dNi/dzeta] = J @ [dNi/dx;
    dNi/dy; dNi/dz]``, so the physical derivatives needed by the
    strain-displacement matrix are recovered as ``[dNi/dx; dNi/dy;
    dNi/dz] = J^-1 @ [dNi/dxi; dNi/deta; dNi/dzeta]``.

    Args:
        dn_dxi: Each node's ``dNi/dxi`` derivative at this natural-coordinate point.
        dn_deta: Each node's ``dNi/deta`` derivative at this point.
        dn_dzeta: Each node's ``dNi/dzeta`` derivative at this point.
        x_coords: The element's node X coordinates, in meters.
        y_coords: The element's node Y coordinates, in meters.
        z_coords: The element's node Z coordinates, in meters.

    Returns:
        ``(dN_dx, dN_dy, dN_dz, det_J)``: three NumPy arrays (one entry
        per node) of physical derivatives, and the Jacobian determinant
        at this point (needed to weight numerical integration).

    Raises:
        DegenerateElementError: If the Jacobian determinant is not
            positive at this point.
    """
    jacobian = jacobian_matrix_3d(dn_dxi, dn_deta, dn_dzeta, x_coords, y_coords, z_coords)
    det_j = jacobian_determinant_3d(jacobian)
    j_inv = inverse_jacobian_3d(jacobian)

    natural_derivatives = np.array([dn_dxi, dn_deta, dn_dzeta], dtype=float)
    physical_derivatives = j_inv @ natural_derivatives
    return physical_derivatives[0], physical_derivatives[1], physical_derivatives[2], det_j
