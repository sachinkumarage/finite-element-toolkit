"""Newton-Raphson convergence criteria.

Two independent ways to decide a Newton-Raphson iteration has converged
are supported, both expressed as **dimensionless ratios** (never an
absolute quantity, which would depend on the problem's own force/length
scale and could not be compared against a single fixed tolerance across
different models):

.. code-block:: text

    residual norm:       ||R|| / ||F_ext|| < tolerance
    displacement norm:    ||du|| / ||u||   < tolerance

The **residual criterion** checks how far the current trial
displacement is from satisfying equilibrium (``R = F_ext - F_int``)
relative to the applied load's own size -- the more physically direct
measure ("are internal and external forces balanced"), and this
module's default. The **displacement criterion** instead checks how
much the last iteration still changed the displacement relative to the
displacement's own size -- useful when the residual is a poor scale
reference (e.g. a very small applied load).

**Avoiding division by (near) zero.** Both ratios divide by a quantity
that can legitimately be zero or extremely small (the very first load
increment, before any load has been applied, gives ``u = 0``; a model
with genuinely zero net external force at some step gives
``F_ext = 0``). Dividing by such a value would produce ``inf``/``nan``
and silently break convergence checking. Both ratio functions instead
substitute a floor value (:data:`NORM_FLOOR`) as the denominator
whenever the natural denominator falls below it -- equivalent to
falling back to an *absolute* (rather than relative) comparison in that
regime, which is safe because :data:`NORM_FLOOR` is small enough that a
numerator larger than it is a genuine, physically meaningful residual.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from femtoolkit.exceptions import ValidationError

ConvergenceCriterion = Literal["residual", "displacement"]

NORM_FLOOR: float = 1e-12
"""Minimum denominator used by :func:`residual_norm_ratio`/
:func:`displacement_correction_ratio`, substituted whenever the natural
denominator (``||F_ext||`` or ``||u||``) falls below it, to avoid
dividing by (near) zero.
"""


def residual_norm_ratio(residual: np.ndarray, external_force: np.ndarray) -> float:
    """Compute ``||R|| / ||F_ext||``, with a safe floor when ``||F_ext|| ~= 0``.

    Args:
        residual: The residual vector ``R = F_ext - F_int`` (typically
            restricted to the free DOFs).
        external_force: The applied external force vector ``F_ext``
            (same DOF selection as ``residual``).

    Returns:
        The dimensionless residual norm ratio.
    """
    denominator = np.linalg.norm(external_force)
    if denominator < NORM_FLOOR:
        denominator = 1.0
    return float(np.linalg.norm(residual) / denominator)


def displacement_correction_ratio(delta_u: np.ndarray, u: np.ndarray) -> float:
    """Compute ``||delta_u|| / ||u||``, with a safe floor when ``||u|| ~= 0``.

    Args:
        delta_u: The Newton-Raphson displacement correction from the
            most recent iteration (typically restricted to the free DOFs).
        u: The current trial displacement (same DOF selection as ``delta_u``).

    Returns:
        The dimensionless displacement correction ratio.
    """
    denominator = np.linalg.norm(u)
    if denominator < NORM_FLOOR:
        denominator = 1.0
    return float(np.linalg.norm(delta_u) / denominator)


def has_converged(
    criterion: ConvergenceCriterion,
    *,
    residual: np.ndarray,
    external_force: np.ndarray,
    delta_u: np.ndarray,
    u: np.ndarray,
    tolerance: float,
) -> bool:
    """Evaluate the configured convergence criterion against ``tolerance``.

    Args:
        criterion: ``"residual"`` (default, see
            :func:`residual_norm_ratio`) or ``"displacement"`` (see
            :func:`displacement_correction_ratio`).
        residual: The residual vector ``R = F_ext - F_int`` (free DOFs).
        external_force: The applied external force vector (free DOFs).
        delta_u: The most recent Newton-Raphson displacement correction
            (free DOFs).
        u: The current trial displacement (free DOFs).
        tolerance: The convergence tolerance to compare the chosen
            ratio against. Must be positive.

    Returns:
        ``True`` if the configured ratio is below ``tolerance``.

    Raises:
        ValidationError: If ``criterion`` is not ``"residual"`` or
            ``"displacement"``.
    """
    if criterion == "residual":
        return residual_norm_ratio(residual, external_force) < tolerance
    if criterion == "displacement":
        return displacement_correction_ratio(delta_u, u) < tolerance
    raise ValidationError(
        f'convergence criterion must be "residual" or "displacement", got {criterion!r}.'
    )
