"""Confidence intervals for an estimated Monte Carlo mean (Version 31).

**This answers a different question than a percentile.**
:mod:`femtoolkit.uncertainty.statistics`'s percentiles describe the
*sampled output distribution itself* -- "95% of successful samples fall
below this value." A confidence interval instead describes how
precisely the *mean* of that distribution has been estimated from a
finite number of samples -- "if this study were repeated many times,
95% of the resulting confidence intervals would contain the true
population mean." Calling a 95th percentile a "95% confidence interval"
conflates these two different statistical statements; this module keeps
them in separate types and separate functions.

The interval is computed with the Student-``t`` distribution rather
than the ``z`` (normal) approximation, since the population standard
deviation is never actually known here -- only estimated from the same
finite sample -- which is exactly the situation the ``t`` distribution
is built for. As the sample count grows, the ``t`` distribution
converges to the normal one, so this is never less appropriate than the
``z`` approximation and is more appropriate for a small sample.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from femtoolkit.exceptions import ValidationError


@dataclass(frozen=True)
class ConfidenceInterval:
    """A confidence interval for an estimated population mean.

    Attributes:
        quantity_label: The summarized quantity's name.
        confidence_level: The confidence level used (e.g. ``0.95``).
        mean: The sample mean (the point estimate).
        lower: The interval's lower bound.
        upper: The interval's upper bound.
        standard_error: ``std(ddof=1) / sqrt(n)``.
        degrees_of_freedom: ``n - 1``.
        n_samples: How many samples the interval was computed from.
    """

    quantity_label: str
    confidence_level: float
    mean: float
    lower: float
    upper: float
    standard_error: float
    degrees_of_freedom: int
    n_samples: int


def confidence_interval_mean(
    values: np.ndarray, quantity_label: str = "quantity", confidence_level: float = 0.95
) -> ConfidenceInterval:
    """Compute a Student-t confidence interval for the population mean.

    ``mean +/- t(alpha/2, n-1) * standard_error``.

    Args:
        values: The successful runs' output values.
        quantity_label: A human-readable name for the quantity.
        confidence_level: The desired confidence level, in ``(0, 1)``
            (e.g. ``0.95`` for a 95% interval).

    Returns:
        A :class:`ConfidenceInterval`.

    Raises:
        ValidationError: If ``confidence_level`` is not in ``(0, 1)``,
            or fewer than two values are given (a standard error, and
            therefore an interval, needs at least two samples).
    """
    if not (0.0 < confidence_level < 1.0):
        raise ValidationError(f"confidence_level must lie in (0, 1), got {confidence_level!r}.")

    values = np.asarray(values, dtype=float)
    n = int(values.size)
    if n < 2:
        raise ValidationError(
            f"confidence_interval_mean requires at least 2 samples, got {n}."
        )

    from scipy import stats

    mean = float(np.mean(values))
    standard_error = float(np.std(values, ddof=1) / math.sqrt(n))
    degrees_of_freedom = n - 1
    t_value = float(stats.t.ppf(0.5 + confidence_level / 2.0, df=degrees_of_freedom))
    margin = t_value * standard_error

    return ConfidenceInterval(
        quantity_label=quantity_label,
        confidence_level=confidence_level,
        mean=mean,
        lower=mean - margin,
        upper=mean + margin,
        standard_error=standard_error,
        degrees_of_freedom=degrees_of_freedom,
        n_samples=n,
    )


__all__ = ["ConfidenceInterval", "confidence_interval_mean"]
