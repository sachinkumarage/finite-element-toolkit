"""Isotropic linear elastic constitutive matrices for 2D continuum analysis.

A 2D continuum problem must reduce the full 3D stress state to two
dimensions with one of two simplifying assumptions:

* **Plane stress** (``sigma_z = 0``): appropriate for thin, flat bodies
  loaded in their own plane (e.g. a thin plate), where the out-of-plane
  faces are free and the through-thickness stress cannot develop.
* **Plane strain** (``epsilon_z = 0``): appropriate for bodies that are
  very long in the out-of-plane direction relative to their cross-section
  and prevented from extending along it (e.g. a long dam cross-section or
  tunnel lining), where through-thickness strain cannot develop.

Both give a 3x3 constitutive matrix relating the same engineering stress
and strain vectors, ``sigma = D @ epsilon``, but with different values --
using the wrong one for a given problem's geometry produces physically
incorrect stiffness. See :mod:`femtoolkit.continuum.strain` for the
engineering-shear-strain convention these matrices assume.

:func:`isotropic_3d_matrix` (Version 15) builds the full 6x6 constitutive
matrix for genuine 3D isotropic linear elasticity -- no plane-stress or
plane-strain reduction needed, since a 3D solid element already carries
all six independent stress/strain components.
"""

from __future__ import annotations

import math

import numpy as np

from femtoolkit.exceptions import ValidationError

_MIN_POISSONS_RATIO = -1.0
_MAX_POISSONS_RATIO = 0.5


def _validate_elastic_constants(youngs_modulus: float, poisson_ratio: float) -> None:
    if not math.isfinite(youngs_modulus) or youngs_modulus <= 0:
        raise ValidationError(f"youngs_modulus must be positive, got {youngs_modulus}.")
    if not math.isfinite(poisson_ratio) or not (
        _MIN_POISSONS_RATIO < poisson_ratio < _MAX_POISSONS_RATIO
    ):
        raise ValidationError(
            f"poisson_ratio must be within ({_MIN_POISSONS_RATIO}, {_MAX_POISSONS_RATIO}), "
            f"got {poisson_ratio}."
        )


def plane_stress_matrix(youngs_modulus: float, poisson_ratio: float) -> np.ndarray:
    """Build the plane-stress constitutive matrix ``D``.

    .. code-block:: text

        D =
        E/(1-v^2) *
        [ 1    v       0    ]
        [ v    1       0    ]
        [ 0    0   (1-v)/2  ]

    Args:
        youngs_modulus: Young's modulus ``E``, in pascals. Must be positive.
        poisson_ratio: Poisson's ratio ``v`` (dimensionless). Must lie in
            ``(-1.0, 0.5)``, the physically valid range for an isotropic
            elastic material.

    Returns:
        A 3x3 NumPy array, the plane-stress constitutive matrix.

    Raises:
        ValidationError: If ``youngs_modulus`` or ``poisson_ratio`` is
            not a physically valid, finite value.

    Example:
        >>> plane_stress_matrix(youngs_modulus=1.0, poisson_ratio=0.3)
    """
    _validate_elastic_constants(youngs_modulus, poisson_ratio)

    e, v = youngs_modulus, poisson_ratio
    factor = e / (1.0 - v**2)
    return factor * np.array(
        [
            [1.0, v, 0.0],
            [v, 1.0, 0.0],
            [0.0, 0.0, (1.0 - v) / 2.0],
        ]
    )


def plane_strain_matrix(youngs_modulus: float, poisson_ratio: float) -> np.ndarray:
    """Build the plane-strain constitutive matrix ``D``.

    .. code-block:: text

        D =
        E/((1+v)(1-2v)) *
        [ 1-v    v        0    ]
        [ v     1-v       0    ]
        [ 0      0    (1-2v)/2 ]

    Args:
        youngs_modulus: Young's modulus ``E``, in pascals. Must be positive.
        poisson_ratio: Poisson's ratio ``v`` (dimensionless). Must lie in
            ``(-1.0, 0.5)``, the physically valid range for an isotropic
            elastic material.

    Returns:
        A 3x3 NumPy array, the plane-strain constitutive matrix.

    Raises:
        ValidationError: If ``youngs_modulus`` or ``poisson_ratio`` is
            not a physically valid, finite value.

    Example:
        >>> plane_strain_matrix(youngs_modulus=1.0, poisson_ratio=0.3)
    """
    _validate_elastic_constants(youngs_modulus, poisson_ratio)

    e, v = youngs_modulus, poisson_ratio
    factor = e / ((1.0 + v) * (1.0 - 2.0 * v))
    return factor * np.array(
        [
            [1.0 - v, v, 0.0],
            [v, 1.0 - v, 0.0],
            [0.0, 0.0, (1.0 - 2.0 * v) / 2.0],
        ]
    )


def isotropic_3d_matrix(youngs_modulus: float, poisson_ratio: float) -> np.ndarray:
    """Build the 6x6 isotropic linear elastic constitutive matrix ``D`` for 3D solids.

    Expressed through the Lame parameters:

    .. code-block:: text

        lambda = E*v / ((1+v)(1-2v))
        mu     = E / (2(1+v))

        D =
        [ lambda+2mu   lambda       lambda       0    0    0  ]
        [ lambda       lambda+2mu   lambda       0    0    0  ]
        [ lambda       lambda       lambda+2mu   0    0    0  ]
        [ 0            0            0            mu   0    0  ]
        [ 0            0            0            0    mu   0  ]
        [ 0            0            0            0    0    mu ]

    for the Voigt ordering ``[xx, yy, zz, xy, yz, xz]`` (see
    :mod:`femtoolkit.continuum.tensor`). ``mu`` (the shear modulus)
    relates engineering shear stress and strain directly,
    ``tau_xy = mu * gamma_xy``, consistent with the engineering-shear
    convention used throughout this project.

    Unlike :func:`plane_stress_matrix`/:func:`plane_strain_matrix`, no
    dimensional reduction is applied: this is the full, unconstrained
    isotropic elasticity tensor in Voigt form, symmetric and positive
    definite for any physically valid ``(youngs_modulus, poisson_ratio)``
    pair (verified in the test suite).

    Args:
        youngs_modulus: Young's modulus ``E``, in pascals. Must be positive.
        poisson_ratio: Poisson's ratio ``v`` (dimensionless). Must lie in
            ``(-1.0, 0.5)``, the physically valid range for an isotropic
            elastic material.

    Returns:
        A 6x6 NumPy array, the symmetric 3D isotropic constitutive matrix.

    Raises:
        ValidationError: If ``youngs_modulus`` or ``poisson_ratio`` is
            not a physically valid, finite value.

    Example:
        >>> isotropic_3d_matrix(youngs_modulus=200e9, poisson_ratio=0.3).shape
        (6, 6)
    """
    _validate_elastic_constants(youngs_modulus, poisson_ratio)

    e, v = youngs_modulus, poisson_ratio
    lame_lambda = e * v / ((1.0 + v) * (1.0 - 2.0 * v))
    shear_modulus = e / (2.0 * (1.0 + v))
    lambda_plus_2mu = lame_lambda + 2.0 * shear_modulus

    return np.array(
        [
            [lambda_plus_2mu, lame_lambda, lame_lambda, 0.0, 0.0, 0.0],
            [lame_lambda, lambda_plus_2mu, lame_lambda, 0.0, 0.0, 0.0],
            [lame_lambda, lame_lambda, lambda_plus_2mu, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, shear_modulus, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, shear_modulus, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, shear_modulus],
        ]
    )
