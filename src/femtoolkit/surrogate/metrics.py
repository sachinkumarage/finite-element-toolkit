"""Surrogate accuracy metrics (Version 35).

Every metric here operates on plain 1D arrays of actual vs. predicted
values, in physical (already inverse-scaled) units -- none of them
knows about a particular surrogate model or dataset, so they are
equally usable for a polynomial surrogate, an RBF surrogate, or a POD
reconstruction.

.. math::

    MAE = \\frac{1}{n}\\sum_i |y_i - \\hat y_i|

    RMSE = \\sqrt{\\frac{1}{n}\\sum_i (y_i - \\hat y_i)^2}

    e_i = \\frac{|y_i - \\hat y_i|}{|y_i| + \\epsilon}

    R^2 = 1 - \\frac{\\sum_i (y_i - \\hat y_i)^2}{\\sum_i (y_i - \\bar y)^2}
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFAULT_EPSILON = 1e-12
"""The small constant added to a relative-error denominator to avoid division by zero."""


def mean_absolute_error(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean absolute error between ``actual`` and ``predicted``."""
    actual, predicted = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    return float(np.mean(np.abs(actual - predicted)))


def root_mean_square_error(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Root mean square error between ``actual`` and ``predicted``."""
    actual, predicted = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def relative_error(
    actual: np.ndarray, predicted: np.ndarray, epsilon: float = DEFAULT_EPSILON
) -> np.ndarray:
    """Per-sample relative error ``|y - y_hat| / (|y| + epsilon)``.

    Args:
        actual: The true values.
        predicted: The predicted values.
        epsilon: A small constant preventing division by (near-)zero
            true values -- never raises or produces ``NaN``/``inf`` for
            a zero-response sample.

    Returns:
        A 1D array of non-negative relative errors, same length as
        ``actual``.
    """
    actual, predicted = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    return np.abs(actual - predicted) / (np.abs(actual) + epsilon)


def r_squared(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Coefficient of determination :math:`R^2`.

    Args:
        actual: The true values.
        predicted: The predicted values.

    Returns:
        ``1 - SS_res / SS_tot``. Returns ``1.0`` if every ``actual``
        value is identical and the prediction matches it exactly
        (a perfect, if degenerate, fit); returns ``0.0`` for the same
        case when the prediction does not match, rather than dividing
        by zero.
    """
    actual, predicted = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    residual_sum_of_squares = float(np.sum((actual - predicted) ** 2))
    total_sum_of_squares = float(np.sum((actual - actual.mean()) ** 2))
    if total_sum_of_squares < 1e-300:
        return 1.0 if residual_sum_of_squares < 1e-300 else 0.0
    return 1.0 - residual_sum_of_squares / total_sum_of_squares


@dataclass(frozen=True)
class MetricSet:
    """A bundle of the standard accuracy metrics computed over one actual/predicted pair.

    Attributes:
        n_samples: How many samples the metrics were computed over.
        mae: Mean absolute error.
        rmse: Root mean square error.
        r2: Coefficient of determination.
        max_relative_error: The largest per-sample relative error.
        mean_relative_error: The average per-sample relative error.
    """

    n_samples: int
    mae: float
    rmse: float
    r2: float
    max_relative_error: float
    mean_relative_error: float

    def to_dict(self) -> dict[str, float | int]:
        """Return a plain, JSON-serializable representation of this metric set."""
        return {
            "n_samples": self.n_samples,
            "mae": self.mae,
            "rmse": self.rmse,
            "r2": self.r2,
            "max_relative_error": self.max_relative_error,
            "mean_relative_error": self.mean_relative_error,
        }


def compute_metrics(actual: np.ndarray, predicted: np.ndarray) -> MetricSet:
    """Compute the standard :class:`MetricSet` for one actual/predicted pair.

    Args:
        actual: The true values (1D).
        predicted: The predicted values (1D, same length as ``actual``).

    Returns:
        A :class:`MetricSet`.
    """
    actual, predicted = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    relative = relative_error(actual, predicted)
    return MetricSet(
        n_samples=int(actual.size),
        mae=mean_absolute_error(actual, predicted),
        rmse=root_mean_square_error(actual, predicted),
        r2=r_squared(actual, predicted),
        max_relative_error=float(relative.max()) if relative.size else 0.0,
        mean_relative_error=float(relative.mean()) if relative.size else 0.0,
    )


__all__ = [
    "DEFAULT_EPSILON",
    "MetricSet",
    "compute_metrics",
    "mean_absolute_error",
    "r_squared",
    "relative_error",
    "root_mean_square_error",
]
