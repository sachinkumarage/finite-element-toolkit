"""Stress recovery for 2D continuum elements, and finite-strain stress-measure conversions.

Given the constant strain field of a CST element (see
:mod:`femtoolkit.continuum.strain`) and its constitutive matrix (see
:mod:`femtoolkit.continuum.constitutive`), this module computes stress,
von Mises equivalent stress, and in-plane principal stresses.

**Stress measures (Version 16).** A finite-strain (Total Lagrangian)
analysis (:mod:`femtoolkit.analysis.geometric_nonlinear`) works entirely in
the reference configuration with the **second Piola-Kirchhoff stress**
``S`` -- the energy-conjugate partner of Green-Lagrange strain ``E`` (see
:mod:`femtoolkit.continuum.deformation`), symmetric, and expressed purely
in reference-configuration quantities, which is exactly what makes it
convenient for a reference-configuration formulation. It is not, however,
a physically direct force-per-area like the stress an engineer usually
means:

* **Cauchy stress** ``sigma`` -- the "true" stress: current force per unit
  *current* area. Symmetric. The physically meaningful stress for
  reporting, but awkward to use as the primary unknown in a
  reference-configuration formulation since the current area it is defined
  over is itself part of the unknown solution.
* **First Piola-Kirchhoff stress** ``P`` -- current force per unit
  *reference* area. A "two-point" tensor (relates a reference-area normal
  to a current-configuration force) and, in general, **not symmetric**.
  Sits between the other two: ``P = F @ S``.
* **Second Piola-Kirchhoff stress** ``S`` -- a fictitious force, *pulled
  back* through ``F`` to act on a reference-area normal and produce a
  reference-configuration force. Symmetric, purely reference-configuration,
  and the natural stress measure for a Total Lagrangian formulation.

:func:`first_piola_kirchhoff_from_second` and
:func:`cauchy_stress_from_second_piola_kirchhoff` convert ``S`` to the other
two -- used only for *reporting* (an example or test recovering a physically
intuitive stress), never by the Total Lagrangian formulation itself, which
needs only ``S``.
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


def first_piola_kirchhoff_from_second(
    deformation_gradient_tensor: np.ndarray, second_piola_kirchhoff_stress: np.ndarray
) -> np.ndarray:
    """Convert second Piola-Kirchhoff stress to first Piola-Kirchhoff stress, ``P = F @ S``.

    Args:
        deformation_gradient_tensor: The 3x3 deformation gradient ``F``.
        second_piola_kirchhoff_stress: The symmetric 3x3 second
            Piola-Kirchhoff stress tensor ``S``.

    Returns:
        A (generally non-symmetric) 3x3 NumPy array, the first
        Piola-Kirchhoff stress ``P``.
    """
    return deformation_gradient_tensor @ second_piola_kirchhoff_stress


def cauchy_stress_from_second_piola_kirchhoff(
    deformation_gradient_tensor: np.ndarray, second_piola_kirchhoff_stress: np.ndarray
) -> np.ndarray:
    """Convert second Piola-Kirchhoff stress to Cauchy (true) stress.

    .. code-block:: text

        sigma = (1 / det(F)) * F @ S @ F^T

    Args:
        deformation_gradient_tensor: The 3x3 deformation gradient ``F``.
        second_piola_kirchhoff_stress: The symmetric 3x3 second
            Piola-Kirchhoff stress tensor ``S``.

    Returns:
        A symmetric 3x3 NumPy array, the Cauchy stress ``sigma``.
    """
    jacobian = float(np.linalg.det(deformation_gradient_tensor))
    return (
        deformation_gradient_tensor
        @ second_piola_kirchhoff_stress
        @ deformation_gradient_tensor.T
    ) / jacobian
