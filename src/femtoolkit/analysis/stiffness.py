"""Element stiffness matrix foundation.

The **element stiffness matrix** relates a finite element's nodal
displacements to the nodal forces required to produce them:
``{f} = [k]{u}``. This module implements the stiffness matrix for the
finite elements in the toolkit: a two-node 1D axial bar, a two-node 2D
truss element (an axial bar transformed into global X/Y coordinates via
its direction cosines), a two-node 2D Euler-Bernoulli frame element
(axial + bending stiffness, transformed into global X/Y/RZ coordinates),
a three-node 2D continuum triangle (CST) element with a closed-form
stiffness integral, and a four-node 2D continuum quadrilateral (Q4)
element whose stiffness integral has no closed form and is instead
evaluated with 2x2 Gauss quadrature -- all built directly from the
reusable continuum math in :mod:`femtoolkit.continuum`.
"""

from __future__ import annotations

import math

import numpy as np

from femtoolkit.continuum.gauss import GAUSS_2X2_POINTS
from femtoolkit.continuum.jacobian import physical_shape_function_derivatives
from femtoolkit.continuum.shape_functions import quad_shape_function_derivatives
from femtoolkit.continuum.strain import quad_strain_displacement_matrix
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


def frame_element_stiffness_local(
    youngs_modulus: float, area: float, second_moment_of_area: float, length: float
) -> np.ndarray:
    """Compute the local stiffness matrix of a two-node 2D Euler-Bernoulli frame element.

    A frame element combines the axial stiffness of a bar (``EA/L``) with
    the flexural stiffness of an Euler-Bernoulli beam (``EI``). Its local
    DOFs are ordered ``[u1, v1, theta1, u2, v2, theta2]``, where ``u`` is
    axial displacement, ``v`` is transverse displacement, and ``theta`` is
    rotation about the local out-of-plane axis:

    .. code-block:: text

        [ EA/L        0             0        -EA/L        0             0      ]
        [ 0       12EI/L^3      6EI/L^2        0      -12EI/L^3      6EI/L^2   ]
        [ 0        6EI/L^2       4EI/L         0       -6EI/L^2       2EI/L    ]
        [-EA/L       0             0         EA/L         0             0      ]
        [ 0      -12EI/L^3      -6EI/L^2       0       12EI/L^3      -6EI/L^2  ]
        [ 0        6EI/L^2       2EI/L         0       -6EI/L^2       4EI/L    ]

    The axial terms (rows/columns 0 and 3) are uncoupled from the bending
    terms (rows/columns 1, 2, 4, 5): a frame element is a bar element and
    a beam element sharing the same two nodes, not a new formulation.

    Args:
        youngs_modulus: Young's modulus of the frame material, in
            pascals. Must be positive.
        area: Cross-sectional area, in square meters. Must be positive.
        second_moment_of_area: Second moment of area about the bending
            axis, in meters to the fourth power. Must be positive.
        length: Length of the frame element, in meters. Must be positive.

    Returns:
        A 6x6 NumPy array, the symmetric local stiffness matrix.

    Raises:
        ValidationError: If any of ``youngs_modulus``, ``area``,
            ``second_moment_of_area``, or ``length`` is not a positive,
            finite number.
    """
    _validate_positive_finite(
        youngs_modulus=youngs_modulus,
        area=area,
        second_moment_of_area=second_moment_of_area,
        length=length,
    )

    e, a, i, length_ = youngs_modulus, area, second_moment_of_area, length
    axial = e * a / length_
    bend_v = 12.0 * e * i / length_**3
    bend_vt = 6.0 * e * i / length_**2
    bend_t_major = 4.0 * e * i / length_
    bend_t_minor = 2.0 * e * i / length_

    return np.array(
        [
            [axial, 0.0, 0.0, -axial, 0.0, 0.0],
            [0.0, bend_v, bend_vt, 0.0, -bend_v, bend_vt],
            [0.0, bend_vt, bend_t_major, 0.0, -bend_vt, bend_t_minor],
            [-axial, 0.0, 0.0, axial, 0.0, 0.0],
            [0.0, -bend_v, -bend_vt, 0.0, bend_v, -bend_vt],
            [0.0, bend_vt, bend_t_minor, 0.0, -bend_vt, bend_t_major],
        ]
    )


def frame_element_stiffness_2d(
    youngs_modulus: float,
    area: float,
    second_moment_of_area: float,
    length: float,
    cos_theta: float,
    sin_theta: float,
) -> np.ndarray:
    """Compute the global stiffness matrix of a two-node 2D frame element.

    The local stiffness matrix from :func:`frame_element_stiffness_local`
    is transformed into global ``[ux1, uy1, rz1, ux2, uy2, rz2]``
    coordinates using ``Kg = T^T * Kl * T``, where ``T`` is the frame
    transformation matrix (see
    :func:`~femtoolkit.analysis.transformation.frame_transformation_matrix_2d`)
    built from the element's direction cosines ``cos_theta = (x2-x1)/L``
    and ``sin_theta = (y2-y1)/L``.

    Args:
        youngs_modulus: Young's modulus of the frame material, in
            pascals. Must be positive.
        area: Cross-sectional area, in square meters. Must be positive.
        second_moment_of_area: Second moment of area about the bending
            axis, in meters to the fourth power. Must be positive.
        length: Length of the frame element, in meters. Must be positive.
        cos_theta: Cosine of the element's orientation angle.
        sin_theta: Sine of the element's orientation angle.

    Returns:
        A 6x6 NumPy array, the symmetric local stiffness matrix expressed
        in global X/Y/RZ coordinates.

    Raises:
        ValidationError: If any of ``youngs_modulus``, ``area``,
            ``second_moment_of_area``, or ``length`` is not a positive,
            finite number.
    """
    # Imported locally to avoid a module-level circular import between
    # `stiffness` and `transformation` (both are leaf modules within
    # `femtoolkit.analysis`, so this keeps import order irrelevant).
    from femtoolkit.analysis.transformation import frame_transformation_matrix_2d

    local_stiffness = frame_element_stiffness_local(
        youngs_modulus=youngs_modulus,
        area=area,
        second_moment_of_area=second_moment_of_area,
        length=length,
    )
    transformation = frame_transformation_matrix_2d(cos_theta, sin_theta)
    return transformation.T @ local_stiffness @ transformation


def cst_element_stiffness(
    thickness: float, area: float, b_matrix: np.ndarray, d_matrix: np.ndarray
) -> np.ndarray:
    """Compute the stiffness matrix of a 3-node constant strain triangle (CST).

    .. code-block:: text

        Ke = t * A * B^T * D * B

    where ``t`` is the element thickness, ``A`` is its (physical, always
    positive) area, ``B`` is its strain-displacement matrix (see
    :func:`~femtoolkit.continuum.strain.triangle_strain_displacement_matrix`),
    and ``D`` is the material's constitutive matrix (see
    :mod:`femtoolkit.continuum.constitutive`). Unlike a truss or frame
    element, no coordinate transformation is needed: ``ux``/``uy`` are
    already expressed in global coordinates for a continuum element, so
    this formula produces the element's *global* stiffness matrix
    directly.

    Args:
        thickness: Element thickness, in meters. Must be positive.
        area: Element area, in square meters. Must be positive (the
            caller is responsible for passing the physical, unsigned
            area -- see :mod:`femtoolkit.continuum.geometry`).
        b_matrix: The element's 3x6 strain-displacement matrix.
        d_matrix: The material's 3x3 constitutive matrix.

    Returns:
        A 6x6 NumPy array, the symmetric CST element stiffness matrix.

    Raises:
        ValidationError: If ``thickness`` or ``area`` is not a positive,
            finite number.

    Example:
        >>> from femtoolkit.continuum import plane_stress_matrix
        >>> from femtoolkit.continuum import triangle_strain_displacement_matrix
        >>> b = triangle_strain_displacement_matrix(0.0, 0.0, 1.0, 0.0, 0.0, 1.0)
        >>> d = plane_stress_matrix(youngs_modulus=1.0, poisson_ratio=0.3)
        >>> cst_element_stiffness(thickness=1.0, area=0.5, b_matrix=b, d_matrix=d).shape
        (6, 6)
    """
    _validate_positive_finite(thickness=thickness, area=area)

    return thickness * area * b_matrix.T @ d_matrix @ b_matrix


def quad_element_stiffness(
    x_coords: np.ndarray | list[float],
    y_coords: np.ndarray | list[float],
    thickness: float,
    d_matrix: np.ndarray,
) -> np.ndarray:
    """Compute the stiffness matrix of a 4-node bilinear quadrilateral (Q4) element.

    Unlike the CST element, a Q4 element's strain-displacement matrix
    ``B`` varies within the element (its shape functions are bilinear,
    not linear -- see :mod:`femtoolkit.continuum.shape_functions`), so
    the stiffness integral has no closed form:

    .. code-block:: text

        Ke = integral( B(x,y)^T D B(x,y) t ) dA

    Mapping the area integral through the isoparametric Jacobian (see
    :mod:`femtoolkit.continuum.jacobian`) turns it into an integral over
    the natural-coordinate square, ``dA = det(J) dxi deta``, which is
    then evaluated with 2x2 Gauss quadrature (see
    :mod:`femtoolkit.continuum.gauss`):

    .. code-block:: text

        Ke = sum over the 4 Gauss points of:
             weight * t * B(xi,eta)^T * D * B(xi,eta) * det(J(xi,eta))

    Args:
        x_coords: The element's four node X coordinates, in meters,
            ordered per the isoparametric convention (see
            :func:`~femtoolkit.continuum.shape_functions.quad_shape_functions`).
        y_coords: The element's four node Y coordinates, in meters.
        thickness: Element thickness, in meters. Must be positive.
        d_matrix: The material's 3x3 constitutive matrix.

    Returns:
        An 8x8 NumPy array, the symmetric Q4 element stiffness matrix.

    Raises:
        ValidationError: If ``thickness`` is not a positive, finite number.
        DegenerateElementError: If the Jacobian determinant is not
            positive at any of the four Gauss points (degenerate,
            self-intersecting, or clockwise-ordered geometry).

    Example:
        >>> from femtoolkit.continuum import plane_stress_matrix
        >>> d = plane_stress_matrix(youngs_modulus=200e9, poisson_ratio=0.3)
        >>> x = [0.0, 1.0, 1.0, 0.0]
        >>> y = [0.0, 0.0, 1.0, 1.0]
        >>> quad_element_stiffness(x, y, thickness=0.01, d_matrix=d).shape
        (8, 8)
    """
    _validate_positive_finite(thickness=thickness)

    stiffness = np.zeros((8, 8))
    for point in GAUSS_2X2_POINTS:
        dn_dxi, dn_deta = quad_shape_function_derivatives(point.xi, point.eta)
        dn_dx, dn_dy, det_j = physical_shape_function_derivatives(
            dn_dxi, dn_deta, x_coords, y_coords
        )
        b_matrix = quad_strain_displacement_matrix(dn_dx, dn_dy)
        stiffness += point.weight * thickness * (b_matrix.T @ d_matrix @ b_matrix) * det_j

    return stiffness
