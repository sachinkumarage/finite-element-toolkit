"""Coordinate transformation for 2D frame elements.

A 2D frame element's local stiffness matrix
(:func:`~femtoolkit.analysis.stiffness.frame_element_stiffness_local`) is
expressed in the element's own local axes: axial (``u``), transverse
(``v``), and rotation (``theta``). The global structural system, however,
works in global X/Y/RZ coordinates shared by every element regardless of
orientation. The **transformation matrix** ``T`` relates the two::

    {u_local} = [T]{u_global}

and is used to transform the local stiffness matrix into global
coordinates via ``Kg = T^T * Kl * T`` (see
:func:`~femtoolkit.analysis.stiffness.frame_element_stiffness_2d`).
"""

from __future__ import annotations

import numpy as np


def frame_transformation_matrix_2d(cos_theta: float, sin_theta: float) -> np.ndarray:
    """Build the 6x6 transformation matrix for a 2D frame element.

    Relates local DOFs ``[u1, v1, theta1, u2, v2, theta2]`` to global DOFs
    ``[ux1, uy1, rz1, ux2, uy2, rz2]`` using the element's direction
    cosines ``c = cos_theta`` and ``s = sin_theta``:

    .. code-block:: text

        [ c   s   0   0   0   0 ]
        [-s   c   0   0   0   0 ]
        [ 0   0   1   0   0   0 ]
        [ 0   0   0   c   s   0 ]
        [ 0   0   0  -s   c   0 ]
        [ 0   0   0   0   0   1 ]

    Each node contributes an independent 2x2 in-plane rotation block for
    its translational DOFs, plus an identity entry for its rotational DOF
    (rotation about the out-of-plane Z axis is the same in local and
    global coordinates for a planar frame).

    Args:
        cos_theta: Cosine of the element's orientation angle, the angle
            from the global X axis to the vector from node 1 to node 2.
        sin_theta: Sine of the element's orientation angle.

    Returns:
        A 6x6 NumPy array, the orthogonal transformation matrix ``T``.

    Example:
        >>> frame_transformation_matrix_2d(cos_theta=1.0, sin_theta=0.0)
    """
    c, s = cos_theta, sin_theta
    return np.array(
        [
            [c, s, 0.0, 0.0, 0.0, 0.0],
            [-s, c, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, c, s, 0.0],
            [0.0, 0.0, 0.0, -s, c, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ]
    )
