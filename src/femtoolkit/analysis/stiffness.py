"""Element stiffness matrix foundation.

The **element stiffness matrix** relates a finite element's nodal
displacements to the nodal forces required to produce them:
``{f} = [k]{u}``. This module implements the stiffness matrix for the
simplest possible finite element, a two-node 1D axial bar, as the
starting point for the toolkit's numerical foundation.
"""

from __future__ import annotations

import math

import numpy as np

from femtoolkit.exceptions import ValidationError


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
    for name, value in (
        ("youngs_modulus", youngs_modulus),
        ("area", area),
        ("length", length),
    ):
        if not math.isfinite(value) or value <= 0:
            raise ValidationError(f"bar_element_stiffness {name} must be positive, got {value}.")

    axial_stiffness = youngs_modulus * area / length
    return axial_stiffness * np.array([[1.0, -1.0], [-1.0, 1.0]])
