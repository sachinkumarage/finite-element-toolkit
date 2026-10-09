"""Before/after model-accuracy comparison for digital twin model updating (Version 38).

Reuses :func:`~femtoolkit.surrogate.metrics.compute_metrics` (Version 35)
directly for both the baseline and the updated model -- no new accuracy
metric is introduced. The correction is never assumed to help: a model
update that makes agreement with measurements *worse* is reported honestly,
not hidden.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from femtoolkit.digital_twin.results import ModelUpdateResult
from femtoolkit.surrogate.metrics import MetricSet, compute_metrics


@dataclass(frozen=True)
class ModelAccuracyComparison:
    """Baseline-model vs. updated-model accuracy, against the same measured data.

    Attributes:
        before: :class:`~femtoolkit.surrogate.metrics.MetricSet` for the
            baseline (pre-update) predictions against the measured values.
        after: :class:`~femtoolkit.surrogate.metrics.MetricSet` for the
            updated (post-update) predictions against the measured values.
        before_max_absolute_error: The largest absolute residual before updating.
        after_max_absolute_error: The largest absolute residual after updating.
    """

    before: MetricSet
    after: MetricSet
    before_max_absolute_error: float
    after_max_absolute_error: float

    @property
    def improved(self) -> bool:
        """Whether the updated model's RMSE is strictly smaller than the baseline's."""
        return self.after.rmse < self.before.rmse


def compare_model_accuracy(result: ModelUpdateResult) -> ModelAccuracyComparison:
    """Compare baseline-model accuracy against updated-model accuracy.

    Args:
        result: A completed :class:`~femtoolkit.digital_twin.results.ModelUpdateResult`.

    Returns:
        A :class:`ModelAccuracyComparison`.
    """
    before = compute_metrics(result.measured_values, result.initial_predictions)
    after = compute_metrics(result.measured_values, result.updated_predictions)
    return ModelAccuracyComparison(
        before=before,
        after=after,
        before_max_absolute_error=float(np.max(np.abs(result.residuals_before))),
        after_max_absolute_error=float(np.max(np.abs(result.residuals_after))),
    )


__all__ = ["ModelAccuracyComparison", "compare_model_accuracy"]
