"""Finite-difference sensitivity analysis (Version 30).

**Engineering concept.** A dimensionless sensitivity, ``S = (dy/y) /
(dp/p)``, answers "for a 1% change in parameter p, what percentage
change in result y do I get?" It is computed here from exactly two
evaluated points (a first-order finite-difference approximation, not a
derivative computed in closed form), so it describes the *local*, *
secant* behavior between those two points -- not necessarily the
behavior at any other point, and not necessarily linear beyond them.

**Explicit limitations (this is a foundation, not a UQ framework).**
This module has no notion of a probability distribution, confidence
interval, or Monte Carlo sampling -- every sensitivity value here is a
single deterministic finite-difference ratio between two deterministic
FEA runs. It also assumes the underlying response is smooth enough that
a secant is a reasonable local approximation; a sensitivity computed
across two widely separated points on a strongly nonlinear response
(e.g. spanning a buckling or yield transition) can be misleading. See
the Version 31 preview (``README.md``) for where probabilistic
uncertainty quantification is planned.
"""

from __future__ import annotations

from dataclasses import dataclass

from femtoolkit.exceptions import ValidationError
from femtoolkit.studies.comparison import DEFAULT_EPSILON


@dataclass(frozen=True)
class SensitivityResult:
    """One finite-difference sensitivity computed between two evaluated points.

    Attributes:
        parameter_label: The varied parameter's name.
        quantity_label: The observed result quantity's name.
        p1: The first (reference) parameter value.
        p2: The second parameter value.
        y1: The result quantity at ``p1``.
        y2: The result quantity at ``p2``.
        sensitivity: ``(y2-y1)/max(|y1|,eps) / ((p2-p1)/max(|p1|,eps))``.
    """

    parameter_label: str
    quantity_label: str
    p1: float
    p2: float
    y1: float
    y2: float
    sensitivity: float


def compute_sensitivity(
    p1: float,
    p2: float,
    y1: float,
    y2: float,
    parameter_label: str = "parameter",
    quantity_label: str = "quantity",
    epsilon: float = DEFAULT_EPSILON,
) -> SensitivityResult:
    """Compute the finite-difference sensitivity of ``y`` to ``p`` between two points.

    An epsilon floor (see :data:`~femtoolkit.studies.comparison.DEFAULT_EPSILON`)
    is applied to a near-zero *reference value* (``p1`` or ``y1``) so a
    baseline of exactly (or nearly) zero does not produce a division by
    zero -- this is the same convention
    :mod:`femtoolkit.verification.metrics` already uses for relative
    error. A near-zero *actual parameter change* (``p2 == p1``) is a
    different situation -- no change was actually made, so there is
    nothing to compute a sensitivity from -- and is treated as a caller
    error rather than silently epsilon-floored, since flooring it would
    produce an arbitrarily large, misleading sensitivity rather than a
    merely imprecise one.

    Args:
        p1: The first (reference) parameter value.
        p2: The second parameter value. Must differ from ``p1``.
        y1: The result quantity evaluated at ``p1``.
        y2: The result quantity evaluated at ``p2``.
        parameter_label: A human-readable name for the parameter.
        quantity_label: A human-readable name for the result quantity.
        epsilon: The denominator floor for near-zero reference values.

    Returns:
        A :class:`SensitivityResult`.

    Raises:
        ValidationError: If ``p2 == p1`` (no actual parameter change).
    """
    if p2 == p1:
        raise ValidationError(
            f"Cannot compute the sensitivity of {quantity_label!r} to {parameter_label!r}: "
            f"p1 and p2 are equal ({p1!r}); no actual parameter change occurred."
        )

    delta_p_over_p = (p2 - p1) / max(abs(p1), epsilon)
    delta_y_over_y = (y2 - y1) / max(abs(y1), epsilon)
    sensitivity = delta_y_over_y / delta_p_over_p

    return SensitivityResult(
        parameter_label=parameter_label,
        quantity_label=quantity_label,
        p1=p1,
        p2=p2,
        y1=y1,
        y2=y2,
        sensitivity=sensitivity,
    )


def compute_sensitivity_series(
    parameter_values: list[float],
    quantity_values: list[float],
    parameter_label: str = "parameter",
    quantity_label: str = "quantity",
    epsilon: float = DEFAULT_EPSILON,
) -> list[SensitivityResult]:
    """Compute consecutive pairwise sensitivities across a full parameter sweep.

    Mirrors :class:`~femtoolkit.verification.convergence.MeshConvergenceStudy`'s
    point-to-point relative-change pattern: each result compares one
    point in the sweep to the point immediately before it, so the
    returned list traces how the local sensitivity evolves across the
    sweep rather than collapsing it to a single number.

    Args:
        parameter_values: The swept parameter values, in sweep order.
        quantity_values: The corresponding result quantity values, same
            length and order as ``parameter_values``.
        parameter_label: A human-readable name for the parameter.
        quantity_label: A human-readable name for the result quantity.
        epsilon: The denominator floor for near-zero reference values.

    Returns:
        ``len(parameter_values) - 1`` consecutive
        :class:`SensitivityResult` entries.

    Raises:
        ValidationError: If the two input lists differ in length, or if
            fewer than two points are given.
    """
    if len(parameter_values) != len(quantity_values):
        raise ValidationError(
            "parameter_values and quantity_values must have the same length "
            f"({len(parameter_values)} != {len(quantity_values)})."
        )
    if len(parameter_values) < 2:
        raise ValidationError("compute_sensitivity_series requires at least two points.")

    return [
        compute_sensitivity(
            parameter_values[index],
            parameter_values[index + 1],
            quantity_values[index],
            quantity_values[index + 1],
            parameter_label=parameter_label,
            quantity_label=quantity_label,
            epsilon=epsilon,
        )
        for index in range(len(parameter_values) - 1)
    ]
