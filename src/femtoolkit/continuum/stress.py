"""Stress recovery for 2D continuum elements.

Given the constant strain field of a CST element (see
:mod:`femtoolkit.continuum.strain`) and its constitutive matrix (see
:mod:`femtoolkit.continuum.constitutive`), this module computes stress,
von Mises equivalent stress, and in-plane principal stresses.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np


def stress_from_strain(d_matrix: np.ndarray, strain: Sequence[float]) -> np.ndarray:
    """Compute stress from strain via Hooke's law, ``sigma = D @ epsilon``.

    Args:
        d_matrix: The material's 3x3 constitutive matrix (plane stress or
            plane strain).
        strain: Strain vector ``[epsilon_x, epsilon_y, gamma_xy]``.

    Returns:
        A length-3 NumPy array, ``[sigma_x, sigma_y, tau_xy]``.
    """
    return d_matrix @ np.asarray(strain, dtype=float)


def von_mises_3d(
    sigma_x: float,
    sigma_y: float,
    sigma_z: float,
    tau_xy: float,
    tau_yz: float = 0.0,
    tau_zx: float = 0.0,
) -> float:
    """Compute the general 3D von Mises equivalent stress.

    .. code-block:: text

        sigma_vm = sqrt(
            0.5 * (
                (sigma_x - sigma_y)^2
                + (sigma_y - sigma_z)^2
                + (sigma_z - sigma_x)^2
                + 6 * (tau_xy^2 + tau_yz^2 + tau_zx^2)
            )
        )

    This is the minimal, reusable core that both
    :func:`von_mises_plane_stress` and :func:`von_mises_plane_strain`
    reduce to once their respective out-of-plane stress is substituted
    for ``sigma_z`` (with ``tau_yz = tau_zx = 0`` for a 2D stress state).

    Args:
        sigma_x: Normal stress in X, in pascals.
        sigma_y: Normal stress in Y, in pascals.
        sigma_z: Normal stress in Z, in pascals.
        tau_xy: Shear stress in the XY plane, in pascals.
        tau_yz: Shear stress in the YZ plane, in pascals. Zero for a 2D
            stress state.
        tau_zx: Shear stress in the ZX plane, in pascals. Zero for a 2D
            stress state.

    Returns:
        The von Mises equivalent stress, in pascals (always non-negative).
    """
    return math.sqrt(
        0.5
        * (
            (sigma_x - sigma_y) ** 2
            + (sigma_y - sigma_z) ** 2
            + (sigma_z - sigma_x) ** 2
            + 6.0 * (tau_xy**2 + tau_yz**2 + tau_zx**2)
        )
    )


def von_mises_plane_stress(sigma_x: float, sigma_y: float, tau_xy: float) -> float:
    """Compute the von Mises equivalent stress for a plane-stress state.

    .. code-block:: text

        sigma_vm = sqrt(sigma_x^2 - sigma_x*sigma_y + sigma_y^2 + 3*tau_xy^2)

    This is :func:`von_mises_3d` with ``sigma_z = 0`` (the defining
    assumption of plane stress) -- algebraically identical to the formula
    above.

    Args:
        sigma_x: Normal stress in X, in pascals.
        sigma_y: Normal stress in Y, in pascals.
        tau_xy: Shear stress in the XY plane, in pascals.

    Returns:
        The von Mises equivalent stress, in pascals.
    """
    return von_mises_3d(sigma_x, sigma_y, 0.0, tau_xy)


def von_mises_plane_strain(
    sigma_x: float, sigma_y: float, tau_xy: float, poisson_ratio: float
) -> float:
    """Compute the von Mises equivalent stress for a plane-strain state.

    Unlike plane stress, the out-of-plane stress ``sigma_z`` is generally
    *nonzero* under plane strain: since ``epsilon_z = 0`` is enforced by
    the plane-strain assumption, isotropic Hooke's law requires
    ``sigma_z = poisson_ratio * (sigma_x + sigma_y)``. Ignoring this term
    (i.e. reusing :func:`von_mises_plane_stress`) understates the true
    equivalent stress.

    Args:
        sigma_x: Normal stress in X, in pascals.
        sigma_y: Normal stress in Y, in pascals.
        tau_xy: Shear stress in the XY plane, in pascals.
        poisson_ratio: Poisson's ratio of the material (dimensionless).

    Returns:
        The von Mises equivalent stress, in pascals.
    """
    sigma_z = poisson_ratio * (sigma_x + sigma_y)
    return von_mises_3d(sigma_x, sigma_y, sigma_z, tau_xy)


def principal_stresses_2d(sigma_x: float, sigma_y: float, tau_xy: float) -> tuple[float, float]:
    """Compute the two in-plane principal stresses.

    .. code-block:: text

        sigma_1, sigma_2 = (sigma_x + sigma_y)/2 +/- sqrt(((sigma_x-sigma_y)/2)^2 + tau_xy^2)

    The in-plane principal stresses do not depend on the out-of-plane
    stress, so this formula is the same for plane stress and plane
    strain.

    Args:
        sigma_x: Normal stress in X, in pascals.
        sigma_y: Normal stress in Y, in pascals.
        tau_xy: Shear stress in the XY plane, in pascals.

    Returns:
        ``(sigma_1, sigma_2)``, the maximum and minimum in-plane
        principal stresses, in pascals (``sigma_1 >= sigma_2``).
    """
    average = (sigma_x + sigma_y) / 2.0
    radius = math.sqrt(((sigma_x - sigma_y) / 2.0) ** 2 + tau_xy**2)
    return average + radius, average - radius
