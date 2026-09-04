"""Deformation invariants of the right Cauchy-Green tensor (Version 17).

A hyperelastic material's strain-energy function must be **objective**
(frame-indifferent): a rigid-body rotation superposed on any deformation
must not change the stored energy. As established in
:mod:`femtoolkit.continuum.deformation`, this means ``W`` can only depend
on the deformation gradient ``F`` through the right Cauchy-Green tensor
``C = F^T F`` (which is exactly invariant under ``F -> Q @ F`` for any
rotation ``Q``, since ``(QF)^T(QF) = F^T Q^T Q F = F^T F``).

For an **isotropic** material (no preferred material direction -- true of
rubber, unlike e.g. fiber-reinforced composites), ``W`` may be reduced
further: it can only depend on ``C`` through its three principal
invariants, since any two right Cauchy-Green tensors with the same
invariants are related by an orthogonal change of basis, which isotropy
makes physically indistinguishable. This module provides those three
invariants, plus the **isochoric** (volume-normalized) invariants used by
compressible hyperelastic models that decouple deviatoric (shape-
changing) and volumetric (size-changing) response -- see the module
docstring for :mod:`femtoolkit.materials.mooney_rivlin` for why this
decoupling matters in practice.
"""

from __future__ import annotations

import math

import numpy as np

from femtoolkit.exceptions import InvalidDeformationGradientError

MIN_RIGHT_CAUCHY_GREEN_DETERMINANT: float = 1e-18
"""Minimum acceptable ``det(C)``.

``C = F^T F``, so ``det(C) = det(F)^2``; this is the square of
:data:`~femtoolkit.continuum.deformation.MIN_DEFORMATION_GRADIENT_DETERMINANT`,
kept consistent with that tolerance rather than an independently chosen
value. Guards every ``log(J)``/``C^-1`` computation in this package
against ``J <= 0`` (element inversion or collapse) before it can happen.
"""


def first_invariant(right_cauchy_green: np.ndarray) -> float:
    """Compute the first principal invariant, ``I1 = tr(C)``.

    Args:
        right_cauchy_green: The symmetric 3x3 right Cauchy-Green tensor ``C``.

    Returns:
        ``I1``, dimensionless.
    """
    return float(np.trace(right_cauchy_green))


def second_invariant(right_cauchy_green: np.ndarray) -> float:
    """Compute the second principal invariant, ``I2 = 1/2[(tr C)^2 - tr(C^2)]``.

    Args:
        right_cauchy_green: The symmetric 3x3 right Cauchy-Green tensor ``C``.

    Returns:
        ``I2``, dimensionless.
    """
    trace_c = np.trace(right_cauchy_green)
    trace_c_squared = np.trace(right_cauchy_green @ right_cauchy_green)
    return float(0.5 * (trace_c**2 - trace_c_squared))


def third_invariant(right_cauchy_green: np.ndarray) -> float:
    """Compute the third principal invariant, ``I3 = det(C)``.

    Args:
        right_cauchy_green: The symmetric 3x3 right Cauchy-Green tensor ``C``.

    Returns:
        ``I3``, dimensionless. Equal to ``J^2`` (see :func:`jacobian_from_right_cauchy_green`).
    """
    return float(np.linalg.det(right_cauchy_green))


def validate_right_cauchy_green(right_cauchy_green: np.ndarray) -> float:
    """Validate ``C`` and return ``det(C)``.

    Args:
        right_cauchy_green: The symmetric 3x3 right Cauchy-Green tensor ``C``.

    Returns:
        ``det(C)``, guaranteed finite and positive.

    Raises:
        InvalidDeformationGradientError: If ``C`` contains a non-finite
            entry, or ``det(C)`` is not positive and above
            :data:`MIN_RIGHT_CAUCHY_GREEN_DETERMINANT` -- the ``C``-space
            equivalent of an inverted or collapsed element.
    """
    if not np.all(np.isfinite(right_cauchy_green)):
        raise InvalidDeformationGradientError(
            f"Right Cauchy-Green tensor contains non-finite entries: {right_cauchy_green!r}."
        )
    determinant = third_invariant(right_cauchy_green)
    if not math.isfinite(determinant) or determinant < MIN_RIGHT_CAUCHY_GREEN_DETERMINANT:
        raise InvalidDeformationGradientError(
            f"Right Cauchy-Green tensor determinant {determinant} is not positive; the "
            "element has inverted or collapsed (excessive deformation)."
        )
    return determinant


def jacobian_from_right_cauchy_green(right_cauchy_green: np.ndarray) -> float:
    """Compute the volume ratio ``J = sqrt(det(C))``, validating ``C`` first.

    .. code-block:: text

        J = det(F)              (volume ratio, current/reference)
        J^2 = det(C) = I3       (since C = F^T F, det(C) = det(F)^2)

    Args:
        right_cauchy_green: The symmetric 3x3 right Cauchy-Green tensor ``C``.

    Returns:
        ``J``, guaranteed positive.

    Raises:
        InvalidDeformationGradientError: If ``C`` is invalid (see
            :func:`validate_right_cauchy_green`).
    """
    determinant = validate_right_cauchy_green(right_cauchy_green)
    return math.sqrt(determinant)


def isochoric_first_invariant(right_cauchy_green: np.ndarray) -> float:
    """Compute the isochoric (volume-normalized) first invariant, ``I1_bar = J^(-2/3) * I1``.

    Equal to ``first_invariant`` of the isochoric right Cauchy-Green tensor
    ``C_bar = J^(-2/3) * C`` (itself ``F_bar^T F_bar`` for the isochoric
    deformation gradient
    :func:`~femtoolkit.continuum.deformation.isochoric_deformation_gradient`,
    ``F_bar = J^(-1/3) F``) -- computed here directly via the invariant-
    scaling identity rather than by explicitly forming ``F_bar``, which is
    equivalent but avoids an unnecessary matrix construction.

    ``I1_bar`` is exactly ``3`` whenever ``C`` is a pure volumetric
    scaling of the identity (``C = alpha^2 I`` for any ``alpha > 0``), and
    otherwise depends only on the *shape*-changing part of the
    deformation -- the property that makes it useful for decoupled
    compressible hyperelastic models (see
    :mod:`femtoolkit.materials.mooney_rivlin`).

    Args:
        right_cauchy_green: The symmetric 3x3 right Cauchy-Green tensor ``C``.

    Returns:
        ``I1_bar``, dimensionless.

    Raises:
        InvalidDeformationGradientError: If ``C`` is invalid.
    """
    jacobian = jacobian_from_right_cauchy_green(right_cauchy_green)
    return jacobian ** (-2.0 / 3.0) * first_invariant(right_cauchy_green)


def isochoric_second_invariant(right_cauchy_green: np.ndarray) -> float:
    """Compute the isochoric (volume-normalized) second invariant, ``I2_bar = J^(-4/3) * I2``.

    See :func:`isochoric_first_invariant` for the analogous derivation and
    physical meaning; ``I2_bar`` is likewise exactly ``3`` for any pure
    volumetric scaling of the identity.

    Args:
        right_cauchy_green: The symmetric 3x3 right Cauchy-Green tensor ``C``.

    Returns:
        ``I2_bar``, dimensionless.

    Raises:
        InvalidDeformationGradientError: If ``C`` is invalid.
    """
    jacobian = jacobian_from_right_cauchy_green(right_cauchy_green)
    return jacobian ** (-4.0 / 3.0) * second_invariant(right_cauchy_green)
