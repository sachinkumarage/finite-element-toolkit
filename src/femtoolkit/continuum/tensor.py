"""3D stress/strain tensor utilities and Voigt notation conversions.

Every 2D continuum module in this package (:mod:`femtoolkit.continuum.strain`,
:mod:`femtoolkit.continuum.stress`) works directly with 3-component Voigt
vectors and never needs the full 2x2 tensor. A 3D solid element
(:mod:`femtoolkit.mesh.tet4_element`, :mod:`femtoolkit.mesh.hex8_element`)
and J2 plasticity (:mod:`femtoolkit.materials.j2_plasticity`) both need to
move between the 6-component Voigt representation used for stiffness/mass
integration and the full symmetric 3x3 tensor representation needed for
tensor operations (trace, deviatoric split, eigenvalues) -- this module is
the single, shared place that conversion happens.

**Voigt ordering**, used consistently throughout the 3D part of this
toolkit:

.. code-block:: text

    stress:  sigma = [sigma_xx, sigma_yy, sigma_zz, tau_xy, tau_yz, tau_xz]
    strain:  epsilon = [epsilon_xx, epsilon_yy, epsilon_zz, gamma_xy, gamma_yz, gamma_xz]

**Tensor vs. engineering shear strain -- the critical distinction.** The
stress tensor's off-diagonal entries are exactly the Voigt shear stresses
(``sigma_tensor[0,1] = tau_xy``, no factor). Strain is different: the
*tensor* shear strain is ``epsilon_xy = (du/dy + dv/dx) / 2``, but the
*engineering* shear strain used in Voigt strain vectors throughout this
project (see :mod:`femtoolkit.continuum.strain`) is
``gamma_xy = du/dy + dv/dx = 2 * epsilon_xy``. Converting a strain Voigt
vector to its tensor form must therefore *halve* the shear entries
(:func:`voigt_strain_to_tensor`), and converting back must *double* them
(:func:`tensor_to_voigt_strain`) -- using the stress conversion functions
(:func:`voigt_stress_to_tensor`/:func:`tensor_to_voigt_stress`, no factor)
on a strain vector is a classic, silent factor-of-two bug. Keeping the
stress and strain conversions as separate, distinctly named functions
(rather than one generic "Voigt to tensor" pair) makes it impossible to
accidentally apply the wrong one.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np


def voigt_stress_to_tensor(stress_voigt: Sequence[float]) -> np.ndarray:
    """Expand a 6-component Voigt stress vector into a symmetric 3x3 tensor.

    Args:
        stress_voigt: ``[sigma_xx, sigma_yy, sigma_zz, tau_xy, tau_yz, tau_xz]``.

    Returns:
        The symmetric 3x3 Cauchy stress tensor, with off-diagonal entries
        equal to the Voigt shear stresses directly (no factor -- see the
        module docstring).
    """
    sigma_xx, sigma_yy, sigma_zz, tau_xy, tau_yz, tau_xz = stress_voigt
    return np.array(
        [
            [sigma_xx, tau_xy, tau_xz],
            [tau_xy, sigma_yy, tau_yz],
            [tau_xz, tau_yz, sigma_zz],
        ]
    )


def tensor_to_voigt_stress(stress_tensor: np.ndarray) -> np.ndarray:
    """Collapse a symmetric 3x3 stress tensor into a 6-component Voigt vector.

    Args:
        stress_tensor: A symmetric 3x3 Cauchy stress tensor.

    Returns:
        ``[sigma_xx, sigma_yy, sigma_zz, tau_xy, tau_yz, tau_xz]``, reading
        the off-diagonal entries directly (no factor).
    """
    return np.array(
        [
            stress_tensor[0, 0],
            stress_tensor[1, 1],
            stress_tensor[2, 2],
            stress_tensor[0, 1],
            stress_tensor[1, 2],
            stress_tensor[0, 2],
        ]
    )


def voigt_strain_to_tensor(strain_voigt: Sequence[float]) -> np.ndarray:
    """Expand a 6-component engineering-shear Voigt strain vector into a tensor.

    Halves each engineering shear strain to recover the tensor shear
    strain (``epsilon_xy = gamma_xy / 2``) -- see the module docstring for
    why this differs from :func:`voigt_stress_to_tensor`.

    Args:
        strain_voigt: ``[epsilon_xx, epsilon_yy, epsilon_zz, gamma_xy, gamma_yz, gamma_xz]``.

    Returns:
        The symmetric 3x3 (tensor) strain tensor.
    """
    epsilon_xx, epsilon_yy, epsilon_zz, gamma_xy, gamma_yz, gamma_xz = strain_voigt
    half_xy, half_yz, half_xz = gamma_xy / 2.0, gamma_yz / 2.0, gamma_xz / 2.0
    return np.array(
        [
            [epsilon_xx, half_xy, half_xz],
            [half_xy, epsilon_yy, half_yz],
            [half_xz, half_yz, epsilon_zz],
        ]
    )


def tensor_to_voigt_strain(strain_tensor: np.ndarray) -> np.ndarray:
    """Collapse a symmetric 3x3 (tensor) strain tensor into an engineering-shear Voigt vector.

    Doubles each tensor shear strain to recover the engineering shear
    strain (``gamma_xy = 2 * epsilon_xy``) -- the inverse of
    :func:`voigt_strain_to_tensor`.

    Args:
        strain_tensor: A symmetric 3x3 (tensor) strain tensor.

    Returns:
        ``[epsilon_xx, epsilon_yy, epsilon_zz, gamma_xy, gamma_yz, gamma_xz]``.
    """
    return np.array(
        [
            strain_tensor[0, 0],
            strain_tensor[1, 1],
            strain_tensor[2, 2],
            2.0 * strain_tensor[0, 1],
            2.0 * strain_tensor[1, 2],
            2.0 * strain_tensor[0, 2],
        ]
    )


def trace(tensor: np.ndarray) -> float:
    """Return the trace of a 3x3 tensor, ``tr(T) = T_xx + T_yy + T_zz``."""
    return float(np.trace(tensor))


def mean_stress(stress_tensor: np.ndarray) -> float:
    """Return the mean (hydrostatic) normal stress, ``sigma_m = tr(sigma) / 3``."""
    return trace(stress_tensor) / 3.0


def hydrostatic_stress(stress_tensor: np.ndarray) -> np.ndarray:
    """Return the hydrostatic stress tensor, ``sigma_hydro = sigma_m * I``.

    The hydrostatic (volumetric) part of stress: a pure, direction-
    independent pressure/tension with no shear component, responsible for
    volume change but not shape change.

    Args:
        stress_tensor: A symmetric 3x3 Cauchy stress tensor.

    Returns:
        ``mean_stress(stress_tensor) * numpy.eye(3)``.
    """
    return mean_stress(stress_tensor) * np.eye(3)


def deviatoric_stress(stress_tensor: np.ndarray) -> np.ndarray:
    """Return the deviatoric stress tensor, ``s = sigma - sigma_m * I``.

    The deviatoric (shape-distorting) part of stress, with the hydrostatic
    part removed -- ``trace(s) == 0`` by construction. See the module
    docstring's note on why J2 plasticity's yield criterion is built from
    this tensor rather than the full stress.

    Args:
        stress_tensor: A symmetric 3x3 Cauchy stress tensor.

    Returns:
        The symmetric, traceless 3x3 deviatoric stress tensor.
    """
    return stress_tensor - hydrostatic_stress(stress_tensor)


def j2_invariant(deviatoric_tensor: np.ndarray) -> float:
    """Return the second deviatoric stress invariant, ``J2 = 0.5 * s:s``.

    ``s:s`` is the full (Frobenius) double contraction
    ``sum_ij(s_ij * s_ij)``, not a matrix product. ``J2`` is always
    non-negative and is the scalar invariant the von Mises yield
    criterion is built from (see :func:`von_mises_stress_from_tensor`).

    Args:
        deviatoric_tensor: A symmetric, traceless 3x3 deviatoric stress
            tensor (see :func:`deviatoric_stress`).

    Returns:
        The second deviatoric invariant ``J2``, in pascals squared.
    """
    return 0.5 * float(np.tensordot(deviatoric_tensor, deviatoric_tensor))


def von_mises_stress_from_tensor(stress_tensor: np.ndarray) -> float:
    """Compute the von Mises equivalent stress directly from the stress tensor.

    .. code-block:: text

        sigma_vm = sqrt(3/2 * s:s) = sqrt(3 * J2)

    Physically equivalent to (and cross-validated against, in the test
    suite, and by :func:`~femtoolkit.continuum.stress.von_mises_3d`) the
    standard Voigt closed-form expression -- this tensor form exists
    because it is the natural quantity a return-mapping algorithm works
    with directly (see :mod:`femtoolkit.materials.j2_plasticity`).

    Args:
        stress_tensor: A symmetric 3x3 Cauchy stress tensor.

    Returns:
        The von Mises equivalent stress, in pascals (always non-negative).
    """
    j2 = j2_invariant(deviatoric_stress(stress_tensor))
    # J2 is mathematically non-negative; the max(..., 0.0) guards only
    # against floating-point round-off driving it fractionally below zero
    # for a near-hydrostatic stress state, which would otherwise raise a
    # domain error in sqrt.
    return math.sqrt(3.0 * max(j2, 0.0))


def principal_stresses_3d(stress_tensor: np.ndarray) -> tuple[float, float, float]:
    """Compute the three principal stresses of a symmetric 3x3 stress tensor.

    Uses :func:`numpy.linalg.eigvalsh`, the robust eigenvalue routine for
    symmetric (Hermitian) matrices -- guaranteed to return real
    eigenvalues for a real symmetric input, unlike the general-purpose
    :func:`numpy.linalg.eigvals`.

    Args:
        stress_tensor: A symmetric 3x3 Cauchy stress tensor.

    Returns:
        ``(sigma_1, sigma_2, sigma_3)``, in pascals, sorted descending
        (``sigma_1 >= sigma_2 >= sigma_3``).
    """
    eigenvalues = np.linalg.eigvalsh(stress_tensor)
    sigma_1, sigma_2, sigma_3 = sorted(eigenvalues, reverse=True)
    return float(sigma_1), float(sigma_2), float(sigma_3)
