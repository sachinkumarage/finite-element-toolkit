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
