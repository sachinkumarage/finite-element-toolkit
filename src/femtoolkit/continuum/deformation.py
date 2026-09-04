"""Finite-deformation kinematics: the deformation gradient and Green-Lagrange strain.

Every element and material through Version 15 assumes **small (infinitesimal)
strain**: the strain-displacement relationship ``epsilon = B @ u`` is linear,
with ``B`` computed once from an element's fixed node coordinates. This is an
excellent approximation while displacements stay small relative to the
structure's own dimensions (``u << characteristic length``), but it breaks
down for genuinely large displacements or rotations -- a cable, a flexible
beam, or a shallow ("snap-through-prone") truss all have equilibrium
configurations a linear analysis cannot represent, because the *geometry
itself*, not just the material's stress-strain law, is changing enough to
matter. This is **geometric nonlinearity**, and it is a fundamentally
different phenomenon from **material nonlinearity** (e.g.
:mod:`femtoolkit.materials.j2_plasticity`, where the stress-strain
relationship itself changes): a perfectly linear-elastic material can still
require a geometrically nonlinear analysis if its displacements are large
enough, and conversely a materially nonlinear (plastic) analysis can stay
geometrically linear if displacements stay small (exactly what every
nonlinear analysis through Version 15 does).

**Reference vs. current configuration.** This module works in the
**reference (undeformed) configuration**: a material point's fixed location
``X`` before any load is applied. The **current (deformed) configuration**
is that same point's location after displacement, ``x = X + u(X)``. The
**deformation gradient**

.. code-block:: text

    F = I + Grad(u) = dx/dX

maps a reference-configuration line element ``dX`` to its current-
configuration image ``dx = F @ dX``, capturing everything about the local
deformation (stretch *and* rotation) at that point. ``Grad(u)`` is the
displacement gradient with respect to the *reference* coordinates -- the
same quantity :func:`displacement_gradient` computes from nodal
displacements and shape-function gradients evaluated at the (fixed)
reference geometry, exactly the Total Lagrangian convention used throughout
this package (see :mod:`femtoolkit.analysis.geometric_nonlinear`).

**Why Green-Lagrange strain.** The linear (small-strain) strain measure is
not *objective*: under a large rigid-body rotation, ``Grad(u)`` alone is
nonzero even though no physical straining occurred, so a linear strain
measure would report spurious "strain" from rotation alone. The
**Green-Lagrange strain tensor**

.. code-block:: text

    C = F^T F                  (right Cauchy-Green deformation tensor)
    E = 1/2 (C - I) = 1/2 (F^T F - I)

fixes this: for a pure rotation ``F = R`` (``R^T R = I``), ``C = I`` exactly,
so ``E = 0`` exactly, regardless of how large the rotation is (verified
numerically in the test suite -- this "objectivity" property is the single
most important correctness check for any finite-strain implementation, see
:mod:`femtoolkit.analysis.geometric_nonlinear`'s module docstring). This is
what makes Green-Lagrange strain suitable for large-displacement analysis
where the small-strain tensor is not: it correctly separates *stretch* (the
physically meaningful part) from *rotation* (physically meaningless for
strain) by construction, via ``C``.

**This does not replace the small-strain formulation.** Every small-strain
element, material, and analysis from Versions 1-15 is untouched; this module
is a **separate, additive pathway** used only by the new geometrically
nonlinear dispatch functions in
:mod:`femtoolkit.analysis.geometric_nonlinear`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from femtoolkit.exceptions import InvalidDeformationGradientError

MIN_DEFORMATION_GRADIENT_DETERMINANT: float = 1e-9
"""Minimum acceptable ``det(F)``.

A physically valid deformation cannot invert or collapse a material element
to zero volume, so ``det(F)`` (the local volume ratio, current/reference)
must stay strictly positive. A determinant at or below this threshold
signals **element inversion** -- excessive deformation the Total Lagrangian
formulation is no longer valid for -- and is rejected outright rather than
silently producing a nonsensical (or NaN-propagating) strain/stress.
"""


def displacement_gradient(
    nodal_displacements: Sequence[float], reference_gradients: np.ndarray
) -> np.ndarray:
    """Compute the displacement gradient ``H = Grad(u)`` w.r.t. the reference configuration.

    .. code-block:: text

        H = sum_a  outer(u_a, Grad0(Na))     (H_iJ = sum_a u_a[i] * dNa/dX_J)

    where ``u_a`` is node ``a``'s displacement vector and ``Grad0(Na))`` is
    that node's shape-function gradient with respect to the *reference*
    coordinates (constant for TET4, evaluated per Gauss point for HEX8 --
    see :mod:`femtoolkit.continuum.jacobian`).

    Args:
        nodal_displacements: Flat nodal displacement vector, ``[u1, v1,
            w1, u2, v2, w2, ...]`` (length ``3 * n_nodes``).
        reference_gradients: An ``(n_nodes, 3)`` array, row ``a`` being
            node ``a``'s reference-configuration shape-function gradient
            ``[dNa/dX, dNa/dY, dNa/dZ]``.

    Returns:
        A 3x3 NumPy array, the displacement gradient ``H``.
    """
    n_nodes = reference_gradients.shape[0]
    nodal_displacements_matrix = np.asarray(nodal_displacements, dtype=float).reshape(n_nodes, 3)
    return nodal_displacements_matrix.T @ reference_gradients


def deformation_gradient(displacement_gradient_tensor: np.ndarray) -> np.ndarray:
    """Compute the deformation gradient, ``F = I + H``.

    Args:
        displacement_gradient_tensor: The 3x3 displacement gradient ``H``
            (see :func:`displacement_gradient`).

    Returns:
        A 3x3 NumPy array, the deformation gradient ``F``.
    """
    return np.eye(3) + displacement_gradient_tensor


def validate_deformation_gradient(deformation_gradient_tensor: np.ndarray) -> float:
    """Validate a deformation gradient and return its determinant.

    Args:
        deformation_gradient_tensor: The 3x3 deformation gradient ``F``.

    Returns:
        ``det(F)``, the local volume ratio (current volume / reference
        volume).

    Raises:
        InvalidDeformationGradientError: If ``F`` contains a non-finite
            entry, or ``det(F)`` is not positive and above
            :data:`MIN_DEFORMATION_GRADIENT_DETERMINANT` (element
            inversion or collapse).
    """
    if not np.all(np.isfinite(deformation_gradient_tensor)):
        raise InvalidDeformationGradientError(
            f"Deformation gradient contains non-finite entries: "
            f"{deformation_gradient_tensor!r}."
        )
    determinant = float(np.linalg.det(deformation_gradient_tensor))
    if not math.isfinite(determinant) or determinant < MIN_DEFORMATION_GRADIENT_DETERMINANT:
        raise InvalidDeformationGradientError(
            f"Deformation gradient determinant {determinant} is not positive; the "
            "element has inverted or collapsed (excessive deformation)."
        )
    return determinant


def right_cauchy_green(deformation_gradient_tensor: np.ndarray) -> np.ndarray:
    """Compute the right Cauchy-Green deformation tensor, ``C = F^T F``.

    ``C`` depends only on the *stretch* part of ``F`` (via the polar
    decomposition ``F = R U``, ``C = U^T R^T R U = U^T U``), never on the
    rotation ``R`` -- the source of Green-Lagrange strain's objectivity
    (see the module docstring).

    Args:
        deformation_gradient_tensor: The 3x3 deformation gradient ``F``.

    Returns:
        A symmetric 3x3 NumPy array, ``C``.
    """
    return deformation_gradient_tensor.T @ deformation_gradient_tensor


def isochoric_deformation_gradient(deformation_gradient_tensor: np.ndarray) -> np.ndarray:
    """Split off the volumetric part of ``F``, returning the isochoric part ``F_bar``.

    .. code-block:: text

        F = J^(1/3) * F_bar          (F_bar has det(F_bar) = 1 exactly)

    This is the **volumetric/isochoric decomposition** (Version 17):
    separating a *volume-changing* part (the scalar ``J^(1/3)`` times the
    identity direction) from a *shape-changing* part (``F_bar``, which by
    construction preserves volume). It underlies compressible hyperelastic
    models that penalize volume change and shape change independently
    (see :mod:`femtoolkit.materials.mooney_rivlin`), and is especially
    useful for nearly incompressible materials, where the volumetric part
    should stay close to its reference value (``J approx 1``) while the
    isochoric part carries essentially all of the deformation.

    Args:
        deformation_gradient_tensor: The 3x3 deformation gradient ``F``.

    Returns:
        A 3x3 NumPy array, ``F_bar = J^(-1/3) * F``, with ``det(F_bar)
        approx 1``.

    Raises:
        InvalidDeformationGradientError: If ``F`` is invalid (see
            :func:`validate_deformation_gradient`).
    """
    jacobian = validate_deformation_gradient(deformation_gradient_tensor)
    return jacobian ** (-1.0 / 3.0) * deformation_gradient_tensor


def green_lagrange_strain_tensor(deformation_gradient_tensor: np.ndarray) -> np.ndarray:
    """Compute the Green-Lagrange strain tensor, ``E = 1/2 (C - I) = 1/2 (F^T F - I)``.

    Args:
        deformation_gradient_tensor: The 3x3 deformation gradient ``F``.

    Returns:
        A symmetric 3x3 NumPy array, ``E``.
    """
    return 0.5 * (right_cauchy_green(deformation_gradient_tensor) - np.eye(3))


def green_lagrange_strain_voigt(deformation_gradient_tensor: np.ndarray) -> np.ndarray:
    """Compute the Green-Lagrange strain directly in 6-component Voigt form.

    Uses :func:`~femtoolkit.continuum.tensor.tensor_to_voigt_strain` for the
    tensor-to-Voigt conversion, so ``E``'s Voigt shear components use the
    same *engineering* shear convention (``gamma = 2*E_shear``) as the
    small-strain formulation throughout this project -- letting the exact
    same 6x6 isotropic constitutive matrix
    (:func:`~femtoolkit.continuum.constitutive.isotropic_3d_matrix`) relate
    ``E`` to the second Piola-Kirchhoff stress ``S`` (see
    :mod:`femtoolkit.materials.finite_strain`).

    Args:
        deformation_gradient_tensor: The 3x3 deformation gradient ``F``.

    Returns:
        A length-6 NumPy array, Voigt
        ``[E_xx, E_yy, E_zz, 2*E_xy, 2*E_yz, 2*E_xz]``.
    """
    from femtoolkit.continuum.tensor import tensor_to_voigt_strain

    return tensor_to_voigt_strain(green_lagrange_strain_tensor(deformation_gradient_tensor))
