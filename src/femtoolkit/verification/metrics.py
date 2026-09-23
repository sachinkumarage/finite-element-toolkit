"""Error metrics comparing a numerical result against a reference value (Version 29).

**Verification** asks "are we solving the equations correctly?" -- and
answering that question quantitatively requires a precise definition of
"how wrong" a numerical result is. This module implements exactly the
handful of error metrics classical FEA verification uses, each a small,
pure function operating on plain floats or NumPy arrays; nothing here
depends on any part of the analysis/solver stack, so the same functions
serve verification (comparing against an analytical solution) and
validation (comparing against a reference dataset) alike.

.. code-block:: text

    absolute_error   e_abs  = |x_fea - x_ref|
    relative_error   e_rel  = |x_fea - x_ref| / max(|x_ref|, eps)
    l2_error         ||e||_2 = sqrt(sum((x_fea_i - x_ref_i)^2))
    relative_l2_error         = ||x_fea - x_ref||_2 / max(||x_ref||_2, eps)

Every relative metric takes a configurable ``epsilon`` floor for the
denominator, avoiding a division-by-zero when the reference value is
itself (exactly or nearly) zero -- e.g. a reaction force at a symmetric
load point, or a displacement at a fixed support.
"""

from __future__ import annotations

import numpy as np

from femtoolkit.exceptions import ValidationError

DEFAULT_EPSILON = 1e-12
"""Default floor for a relative-error denominator, avoiding division by
(near-)zero without needing every call site to supply one explicitly."""


def _validate_epsilon(epsilon: float) -> None:
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValidationError(f"epsilon must be a positive, finite number, got {epsilon}.")


def absolute_error(numerical: float, reference: float) -> float:
    """Compute the absolute error ``e_abs = |x_fea - x_ref|``.

    Args:
        numerical: The FEA (numerical) value.
        reference: The reference (analytical or validation-dataset) value.

    Returns:
        The absolute difference, always non-negative.
    """
    return abs(numerical - reference)


def relative_error(numerical: float, reference: float, epsilon: float = DEFAULT_EPSILON) -> float:
    """Compute the relative error ``e_rel = |x_fea - x_ref| / max(|x_ref|, eps)``.

    Args:
        numerical: The FEA (numerical) value.
        reference: The reference value.
        epsilon: Floor for the denominator, avoiding division by zero
            when ``reference`` is (near) zero. Must be positive.

    Returns:
        The relative error, always non-negative and dimensionless.

    Raises:
        ValidationError: If ``epsilon`` is not positive and finite.
    """
    _validate_epsilon(epsilon)
    return absolute_error(numerical, reference) / max(abs(reference), epsilon)


def l2_error(numerical: np.ndarray, reference: np.ndarray) -> float:
    """Compute the L2 (Euclidean) norm of the error vector, ``||x_fea - x_ref||_2``.

    Args:
        numerical: The FEA (numerical) vector.
        reference: The reference vector, same shape as ``numerical``.

    Returns:
        The L2 norm of ``numerical - reference``, always non-negative.

    Raises:
        ValidationError: If ``numerical`` and ``reference`` have
            different shapes.
    """
    numerical_array = np.asarray(numerical, dtype=float)
    reference_array = np.asarray(reference, dtype=float)
    if numerical_array.shape != reference_array.shape:
        raise ValidationError(
            f"l2_error requires matching shapes, got {numerical_array.shape} and "
            f"{reference_array.shape}."
        )
    return float(np.linalg.norm(numerical_array - reference_array))


def relative_l2_error(
    numerical: np.ndarray, reference: np.ndarray, epsilon: float = DEFAULT_EPSILON
) -> float:
    """Compute the relative L2 error, ``||x_fea - x_ref||_2 / max(||x_ref||_2, eps)``.

    Args:
        numerical: The FEA (numerical) vector.
        reference: The reference vector, same shape as ``numerical``.
        epsilon: Floor for the denominator. Must be positive.

    Returns:
        The relative L2 error, always non-negative and dimensionless.

    Raises:
        ValidationError: If ``epsilon`` is not positive and finite, or
            the two vectors have different shapes.
    """
    _validate_epsilon(epsilon)
    reference_norm = float(np.linalg.norm(np.asarray(reference, dtype=float)))
    return l2_error(numerical, reference) / max(reference_norm, epsilon)


def energy_norm_error(
    numerical_displacements: np.ndarray,
    reference_displacements: np.ndarray,
    stiffness: np.ndarray,
    epsilon: float = DEFAULT_EPSILON,
) -> float:
    """Compute the relative error in the structural energy norm.

    The energy norm of a displacement error ``e = u_fea - u_ref`` is
    ``||e||_E = sqrt(e^T K e)`` -- the strain energy the error vector
    itself would store in the structure. This is the standard norm for
    assessing displacement-based FEA convergence (it directly reflects
    the strain-energy interpretation of the stiffness matrix, unlike a
    plain L2 norm over displacement components, which mixes
    incompatible physical quantities when DOFs represent different
    directions).

    Args:
        numerical_displacements: The FEA (numerical) displacement vector.
        reference_displacements: The reference displacement vector, same
            shape as ``numerical_displacements``.
        stiffness: The global (or reduced) stiffness matrix, square with
            side length matching the displacement vectors. Must be
            symmetric positive semi-definite for the result to be a
            valid norm (not checked here -- any assembled, properly
            constrained stiffness matrix in this toolkit satisfies it).
        epsilon: Floor for the denominator. Must be positive.

    Returns:
        The relative energy-norm error,
        ``sqrt(e^T K e) / max(sqrt(u_ref^T K u_ref), eps)``.

    Raises:
        ValidationError: If ``epsilon`` is not positive and finite, the
            two displacement vectors have different shapes, or
            ``stiffness`` is not square with a side length matching the
            displacement vectors.
    """
    _validate_epsilon(epsilon)
    numerical_array = np.asarray(numerical_displacements, dtype=float)
    reference_array = np.asarray(reference_displacements, dtype=float)
    if numerical_array.shape != reference_array.shape:
        raise ValidationError(
            f"energy_norm_error requires matching displacement shapes, got "
            f"{numerical_array.shape} and {reference_array.shape}."
        )
    n = numerical_array.shape[0]
    if stiffness.shape != (n, n):
        raise ValidationError(
            f"energy_norm_error requires a square stiffness matrix of shape ({n}, {n}), "
            f"got {stiffness.shape}."
        )

    error = numerical_array - reference_array
    error_energy = float(error.T @ stiffness @ error)
    reference_energy = float(reference_array.T @ stiffness @ reference_array)
    # Strain energy is a quadratic form and can be numerically negative
    # by a floating-point rounding error when the true value is exactly
    # zero (e.g. a reference vector of all zeros); clip to zero before
    # taking the square root rather than letting a tiny negative value
    # raise on ``sqrt``.
    error_norm = np.sqrt(max(error_energy, 0.0))
    reference_norm = np.sqrt(max(reference_energy, 0.0))
    return error_norm / max(reference_norm, epsilon)


__all__ = [
    "DEFAULT_EPSILON",
    "absolute_error",
    "energy_norm_error",
    "l2_error",
    "relative_error",
    "relative_l2_error",
]
