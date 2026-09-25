"""Descriptive statistics for a Monte Carlo output sample (Version 31).

Turns the raw array of one output quantity's successful-run values into
a small, structured summary: count, mean, median, standard deviation,
variance, range, coefficient of variation, and configurable percentiles
-- plus a running-mean/standard-error convergence trace
(:func:`compute_convergence_series`) so a caller can see whether the
estimate has settled or is still moving as more samples are added.
Percentiles here describe the *sampled output distribution itself* --
see :mod:`femtoolkit.uncertainty.confidence` for the different question
of how precisely the *mean* is known.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from femtoolkit.exceptions import ValidationError

DEFAULT_PERCENTILES: tuple[float, ...] = (5.0, 25.0, 50.0, 75.0, 95.0)
DEFAULT_EPSILON = 1e-12


@dataclass(frozen=True)
class OutputStatistics:
    """Descriptive statistics for one output quantity's successful Monte Carlo samples.

    Every statistic is ``None`` when it cannot be meaningfully computed
    (no successful samples at all, or a standard deviation/variance/CV
    that needs at least two samples) -- never a fabricated placeholder
    value.

    Attributes:
        quantity_label: The summarized quantity's name.
        units: A units string for display.
        n_requested: How many samples the study requested.
        n_successful: How many runs succeeded and contributed a value.
        n_failed: How many runs failed (solver/validation error).
        n_invalid: How many samples were rejected before execution for
            violating a physical bound.
        mean: The sample mean, or ``None`` if ``n_successful == 0``.
        median: The sample median, or ``None`` if ``n_successful == 0``.
        std: The sample standard deviation (``ddof=1``), or ``None`` if
            ``n_successful < 2``.
        variance: ``std ** 2``, or ``None`` under the same condition.
        minimum: The sample minimum, or ``None`` if ``n_successful == 0``.
        maximum: The sample maximum, or ``None`` if ``n_successful == 0``.
        coefficient_of_variation: ``std / max(|mean|, epsilon)``, or
            ``None`` if ``std`` is ``None``.
        percentiles: ``{percentile: value}`` for each requested
            percentile, empty if ``n_successful == 0``.
    """

    quantity_label: str
    units: str
    n_requested: int
    n_successful: int
    n_failed: int
    n_invalid: int
    mean: float | None
    median: float | None
    std: float | None
    variance: float | None
    minimum: float | None
    maximum: float | None
    coefficient_of_variation: float | None
    percentiles: dict[float, float] = field(default_factory=dict)


def compute_output_statistics(
    values: np.ndarray,
    quantity_label: str,
    units: str = "",
    n_requested: int | None = None,
    n_failed: int = 0,
    n_invalid: int = 0,
    percentiles: tuple[float, ...] = DEFAULT_PERCENTILES,
    epsilon: float = DEFAULT_EPSILON,
) -> OutputStatistics:
    """Compute descriptive statistics for one output quantity's successful samples.

    Args:
        values: The successful runs' extracted output values.
        quantity_label: A human-readable name for the quantity.
        units: A units string for display.
        n_requested: How many samples the study requested; defaults to
            ``len(values) + n_failed + n_invalid`` if not given.
        n_failed: How many runs failed.
        n_invalid: How many samples were rejected before execution.
        percentiles: Which percentiles (0-100) to compute.
        epsilon: The denominator floor for the coefficient of variation.

    Returns:
        An :class:`OutputStatistics` summary.

    Raises:
        ValidationError: If any requested percentile is outside ``[0, 100]``.
    """
    for percentile in percentiles:
        if not (0.0 <= percentile <= 100.0):
            raise ValidationError(f"Percentiles must lie in [0, 100], got {percentile!r}.")

    values = np.asarray(values, dtype=float)
    n_successful = int(values.size)
    total_requested = (
        n_requested if n_requested is not None else n_successful + n_failed + n_invalid
    )

    if n_successful == 0:
        return OutputStatistics(
            quantity_label=quantity_label,
            units=units,
            n_requested=total_requested,
            n_successful=0,
            n_failed=n_failed,
            n_invalid=n_invalid,
            mean=None,
            median=None,
            std=None,
            variance=None,
            minimum=None,
            maximum=None,
            coefficient_of_variation=None,
            percentiles={},
        )

    mean = float(np.mean(values))
    median = float(np.median(values))
    minimum = float(np.min(values))
    maximum = float(np.max(values))
    percentile_values = {p: float(np.percentile(values, p)) for p in percentiles}

    std: float | None = None
    variance: float | None = None
    coefficient_of_variation: float | None = None
    if n_successful >= 2:
        std = float(np.std(values, ddof=1))
        variance = std**2
        coefficient_of_variation = std / max(abs(mean), epsilon)

    return OutputStatistics(
        quantity_label=quantity_label,
        units=units,
        n_requested=total_requested,
        n_successful=n_successful,
        n_failed=n_failed,
        n_invalid=n_invalid,
        mean=mean,
        median=median,
        std=std,
        variance=variance,
        minimum=minimum,
        maximum=maximum,
        coefficient_of_variation=coefficient_of_variation,
        percentiles=percentile_values,
    )


@dataclass(frozen=True)
class ConvergencePoint:
    """The running estimate of a Monte Carlo mean after ``n`` samples.

    Attributes:
        n: How many samples (in draw order) this point includes.
        running_mean: The mean of the first ``n`` samples.
        standard_error: ``std(ddof=1) / sqrt(n)`` of the first ``n``
            samples, or ``None`` for ``n < 2``.
    """

    n: int
    running_mean: float
    standard_error: float | None


def compute_convergence_series(values: np.ndarray) -> list[ConvergencePoint]:
    """Trace the running mean and standard error as samples accumulate, in draw order.

    A simple, honest convergence diagnostic: whether the running mean
    has visibly settled (and the standard error has shrunk toward zero)
    by the end of the series is evidence the sample count was
    sufficient for that quantity -- completion of the simulation runs
    alone is not.

    Args:
        values: The successful runs' output values, in the order they
            were drawn (not sorted).

    Returns:
        One :class:`ConvergencePoint` per prefix length ``1..len(values)``.

    Raises:
        ValidationError: If ``values`` is empty.
    """
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        raise ValidationError("compute_convergence_series requires at least one value.")

    points: list[ConvergencePoint] = []
    for n in range(1, values.size + 1):
        subset = values[:n]
        running_mean = float(np.mean(subset))
        standard_error = float(np.std(subset, ddof=1) / math.sqrt(n)) if n >= 2 else None
        points.append(
            ConvergencePoint(n=n, running_mean=running_mean, standard_error=standard_error)
        )
    return points


__all__ = [
    "DEFAULT_EPSILON",
    "DEFAULT_PERCENTILES",
    "ConvergencePoint",
    "OutputStatistics",
    "compute_convergence_series",
    "compute_output_statistics",
]
