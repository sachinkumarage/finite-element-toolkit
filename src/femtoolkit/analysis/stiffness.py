"""Element stiffness matrix foundation.

The **element stiffness matrix** relates a finite element's nodal
displacements to the nodal forces required to produce them:
``{f} = [k]{u}``. This module implements the stiffness matrix for the two
simplest finite elements in the toolkit: a two-node 1D axial bar, and a
two-node 2D truss element (an axial bar transformed into global X/Y
coordinates via its direction cosines).
"""

from __future__ import annotations

import math

import numpy as np

from femtoolkit.exceptions import ValidationError


def _validate_positive_finite(**values: float) -> None:
    """Raise ValidationError if any named value is not positive and finite."""
    for name, value in values.items():
        if not math.isfinite(value) or value <= 0:
            raise ValidationError(f"{name} must be positive, got {value}.")


def bar_element_stiffness(youngs_modulus: float, area: float, length: float) -> np.ndarray:
    """Compute the local stiffness matrix of a two-node 1D axial bar element.

    For a bar with Young's modulus ``E``, cross-sectional area ``A``, and
    length ``L``, the axial stiffness ``k = EA/L`` relates the two nodal
    axial displacements ``u1, u2`` to the two nodal axial forces
    ``f1, f2``:

    .. code-block:: text

        [ f1 ]   EA/L [  1  -1 ] [ u1 ]
        [ f2 ] =      [ -1   1 ] [ u2 ]

    Args:
        youngs_modulus: Young's modulus of the bar material, in pascals.
            Must be positive.
        area: Cross-sectional area of the bar, in square meters. Must be
            positive.
        length: Length of the bar element, in meters. Must be positive.

    Returns:
        A 2x2 NumPy array, the symmetric local stiffness matrix
        ``[[k, -k], [-k, k]]`` with ``k = youngs_modulus * area / length``.

    Raises:
        ValidationError: If any of ``youngs_modulus``, ``area``, or
            ``length`` is not a positive, finite number.

    Example:
        >>> bar_element_stiffness(youngs_modulus=200e9, area=0.01, length=2.0)
        array([[ 1.e+09, -1.e+09],
               [-1.e+09,  1.e+09]])
    """
    _validate_positive_finite(youngs_modulus=youngs_modulus, area=area, length=length)

    axial_stiffness = youngs_modulus * area / length
    return axial_stiffness * np.array([[1.0, -1.0], [-1.0, 1.0]])


def truss_element_stiffness_2d(
    youngs_modulus: float,
    area: float,
    length: float,
    cos_theta: float,
    sin_theta: float,
) -> np.ndarray:
    """Compute the local stiffness matrix of a two-node 2D truss element.

    A 2D truss element is a 1D axial bar transformed from its local axial
    coordinate into global X/Y coordinates using its direction cosines
    ``c = cos_theta`` and ``s = sin_theta`` (the cosine and sine of the
    angle from the global X axis to the element's axis, from node 1 to
    node 2). For nodal DOFs ordered ``[ux1, uy1, ux2, uy2]``:

    .. code-block:: text

        k = EA / L

        [ c*c   c*s  -c*c  -c*s ]
        [ c*s   s*s  -c*s  -s*s ]
    k * [-c*c  -c*s   c*c   c*s ]
        [-c*s  -s*s   c*s   s*s ]

    Args:
        youngs_modulus: Young's modulus of the truss material, in pascals.
            Must be positive.
        area: Cross-sectional area of the truss member, in square meters.
            Must be positive.
        length: Length of the truss element, in meters. Must be positive.
        cos_theta: Cosine of the element's orientation angle, ``c = (x2-x1)/L``.
        sin_theta: Sine of the element's orientation angle, ``s = (y2-y1)/L``.

    Returns:
        A 4x4 NumPy array, the symmetric local stiffness matrix in global
        X/Y coordinates.

    Raises:
        ValidationError: If any of ``youngs_modulus``, ``area``, or
            ``length`` is not a positive, finite number.

    Example:
        >>> truss_element_stiffness_2d(
        ...     youngs_modulus=200e9, area=0.01, length=2.0, cos_theta=1.0, sin_theta=0.0
        ... )
    """
    _validate_positive_finite(youngs_modulus=youngs_modulus, area=area, length=length)

    axial_stiffness = youngs_modulus * area / length
    c, s = cos_theta, sin_theta
    transformation = np.array(
        [
            [c * c, c * s, -c * c, -c * s],
            [c * s, s * s, -c * s, -s * s],
            [-c * c, -c * s, c * c, c * s],
            [-c * s, -s * s, c * s, s * s],
        ]
    )
    return axial_stiffness * transformation
