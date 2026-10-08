"""Comparing low-fidelity-only accuracy against fused-model accuracy (Version 37).

Reuses :func:`~femtoolkit.surrogate.metrics.compute_metrics` (Version 35)
directly for both comparisons -- no new accuracy metric is introduced. The whole
point of this module is to make explicit that the discrepancy correction is
**not assumed** to always help: a dataset where the fused model is *worse* than
the raw low-fidelity model is reported honestly, not hidden.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from femtoolkit.exceptions import ValidationError
from femtoolkit.multifidelity.dataset import MultiFidelityDataset
from femtoolkit.multifidelity.model import MultiFidelityModel
from femtoolkit.surrogate.metrics import MetricSet, compute_metrics


@dataclass(frozen=True)
class FidelityComparisonReport:
    """Low-fidelity-only vs. fused-model accuracy, against the same high-fidelity samples.

    Attributes:
        n_samples: How many paired samples the comparison was computed over.
        low_fidelity_metrics: Per-response :class:`~femtoolkit.surrogate.metrics.MetricSet`
            for the raw low-fidelity result against the high-fidelity result.
        fused_metrics: Per-response :class:`~femtoolkit.surrogate.metrics.MetricSet`
            for the fused prediction against the high-fidelity result.
    """

    n_samples: int
    low_fidelity_metrics: dict[str, MetricSet] = field(default_factory=dict)
    fused_metrics: dict[str, MetricSet] = field(default_factory=dict)

    def improves_on_low_fidelity(self, response_name: str) -> bool:
        """Whether the fused model's RMSE is smaller than the raw low-fidelity model's.

        Args:
            response_name: Which response to check.

        Returns:
            ``True`` if the fused model's RMSE against the high-fidelity result is
            strictly smaller than the low-fidelity model's own RMSE.

        Raises:
            ValidationError: If ``response_name`` was not part of this comparison.
        """
        known = set(self.low_fidelity_metrics) & set(self.fused_metrics)
        if response_name not in known:
            raise ValidationError(
                f"Response {response_name!r} is not part of this comparison; expected one "
                f"of {sorted(self.low_fidelity_metrics)}."
            )
        low_rmse = self.low_fidelity_metrics[response_name].rmse
        fused_rmse = self.fused_metrics[response_name].rmse
        return fused_rmse < low_rmse


def compare_fidelity_accuracy(
    model: MultiFidelityModel, dataset: MultiFidelityDataset
) -> FidelityComparisonReport:
    """Compare low-fidelity-only accuracy against fused-model accuracy on paired samples.

    Args:
        model: The multi-fidelity model to evaluate (its discrepancy surrogate may
            have been trained on a different split of ``dataset`` -- this function
            does not itself hold out data; pass only genuinely held-out samples for
            an honest accuracy estimate).
        dataset: The paired dataset to evaluate against.

    Returns:
        A :class:`FidelityComparisonReport`.

    Raises:
        ValidationError: If ``dataset`` has no paired samples.
    """
    paired = dataset.paired_samples()
    if not paired:
        raise ValidationError("compare_fidelity_accuracy() requires at least one paired sample.")

    low_fidelity_metrics: dict[str, MetricSet] = {}
    fused_metrics: dict[str, MetricSet] = {}
    for response_name in dataset.response_names:
        high_values = np.array([sample.high_result[response_name] for sample in paired])
        low_values = np.array([sample.low_result[response_name] for sample in paired])
        fused_values = np.array(
            [model.predict(sample.inputs).fused_prediction[response_name] for sample in paired]
        )
        low_fidelity_metrics[response_name] = compute_metrics(high_values, low_values)
        fused_metrics[response_name] = compute_metrics(high_values, fused_values)

    return FidelityComparisonReport(
        n_samples=len(paired),
        low_fidelity_metrics=low_fidelity_metrics,
        fused_metrics=fused_metrics,
    )


__all__ = ["FidelityComparisonReport", "compare_fidelity_accuracy"]
