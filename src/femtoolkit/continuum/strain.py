"""Strain-displacement relation for the constant strain triangle (CST).

**Engineering shear strain convention.** This module (and the entire
Version 6 continuum formulation) uses the *engineering* shear strain
``gamma_xy = du/dy + dv/dx``, not the tensorial shear strain
``epsilon_xy = (du/dy + dv/dx) / 2``. Mixing the two conventions produces
a strain-displacement matrix that is off by a factor of two on its shear
row and an incorrect stiffness matrix -- so every formula in this module,
:mod:`femtoolkit.continuum.constitutive`, and :mod:`femtoolkit.continuum.stress`
consistently uses the engineering convention.

For a 2D continuum, the small-strain field is:

.. code-block:: text

    epsilon_x  = du/dx
    epsilon_y  = dv/dy
    gamma_xy   = du/dy + dv/dx

    epsilon = [ epsilon_x, epsilon_y, gamma_xy ]^T

**Constant strain property.** Because the CST shape functions are linear
in ``x`` and ``y`` (see :mod:`femtoolkit.continuum.shape_functions`),
their gradients -- and therefore the strain field ``epsilon = B @ d`` --
are *constant* over the entire element for any fixed nodal displacement
vector ``d``. This is the origin of the element's name and its main
limitation: a single CST element cannot represent a strain gradient
(e.g. bending), only a uniform strain state.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from femtoolkit.continuum.geometry import MIN_TRIANGLE_AREA, triangle_signed_area
from femtoolkit.exceptions import DegenerateElementError


def triangle_strain_displacement_matrix(
    x1: float, y1: float, x2: float, y2: float, x3: float, y3: float
) -> np.ndarray:
    """Build the 3x6 strain-displacement (``B``) matrix for a CST element.

    For nodal displacements ordered ``[u1, v1, u2, v2, u3, v3]``:

    .. code-block:: text

        B =
        1/(2A) *
        [ b1   0    b2   0    b3   0  ]
        [ 0    c1   0    c2   0    c3 ]
        [ c1   b1   c2   b2   c3   b3 ]

        b1 = y2 - y3     c1 = x3 - x2
        b2 = y3 - y1     c2 = x1 - x3
        b3 = y1 - y2     c3 = x2 - x1

    ``A`` is the triangle's *signed* area (see the orientation policy
    documented in :mod:`femtoolkit.continuum.geometry`): using it
    consistently here (rather than its absolute value) makes ``B``, and
    therefore the recovered strain and stress, identical regardless of
    whether the three nodes are listed clockwise or counter-clockwise.
    The physical (always positive) area is used separately, only in the
    element stiffness formula (see
    :func:`~femtoolkit.analysis.stiffness.cst_element_stiffness`).

    ``B`` does not depend on ``x`` or ``y``: it is constant over the
    element (see the module docstring).

    Args:
        x1: X coordinate of node 1, in meters.
        y1: Y coordinate of node 1, in meters.
        x2: X coordinate of node 2, in meters.
        y2: Y coordinate of node 2, in meters.
        x3: X coordinate of node 3, in meters.
        y3: Y coordinate of node 3, in meters.

    Returns:
        A 3x6 NumPy array, the strain-displacement matrix ``B``.

    Raises:
        DegenerateElementError: If the triangle's area is (near) zero.
    """
    signed_area = triangle_signed_area(x1, y1, x2, y2, x3, y3)
    if abs(signed_area) < MIN_TRIANGLE_AREA:
        raise DegenerateElementError(
            f"Triangle with nodes ({x1},{y1}), ({x2},{y2}), ({x3},{y3}) has "
            f"area {signed_area}, which is degenerate (collinear or duplicate nodes)."
        )

    b1, b2, b3 = y2 - y3, y3 - y1, y1 - y2
    c1, c2, c3 = x3 - x2, x1 - x3, x2 - x1

    factor = 1.0 / (2.0 * signed_area)
    return factor * np.array(
        [
            [b1, 0.0, b2, 0.0, b3, 0.0],
            [0.0, c1, 0.0, c2, 0.0, c3],
            [c1, b1, c2, b2, c3, b3],
        ]
    )


def strain_from_displacements(b_matrix: np.ndarray, displacements: Sequence[float]) -> np.ndarray:
    """Compute the constant strain field, ``epsilon = B @ d``.

    Args:
        b_matrix: The element's 3x6 strain-displacement matrix.
        displacements: Nodal displacements ``[u1, v1, u2, v2, u3, v3]``.

    Returns:
        A length-3 NumPy array, ``[epsilon_x, epsilon_y, gamma_xy]``.
    """
    return b_matrix @ np.asarray(displacements, dtype=float)
