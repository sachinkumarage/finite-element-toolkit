"""Surrogate validation: cross-validation, engineering tolerances, and the validation report.

(Version 35.)

**Statistical accuracy is not the same as engineering acceptability.**
A surrogate can report an excellent global RMSE/R^2 and still be
unacceptable for a safety-critical response if its *worst-case* error
on that one response exceeds what an engineer can tolerate. This module
therefore keeps two separate kinds of check side by side in
:class:`SurrogateValidationReport` -- generic statistical metrics
(:mod:`femtoolkit.surrogate.metrics`) and caller-defined
:class:`EngineeringTolerance` checks -- and never collapses them into a
single "safe"/"unsafe" verdict. A surrogate passing every statistical
metric with a good score can still be reported as failing its
engineering tolerances, and the report says so explicitly rather than
labeling the model safe by default.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from femtoolkit.exceptions import InsufficientSnapshotsError, ValidationError
from femtoolkit.surrogate.metrics import MetricSet, compute_metrics

if TYPE_CHECKING:
    from femtoolkit.surrogate.datasets import SnapshotDataset


@dataclass(frozen=True)
class EngineeringTolerance:
    """A caller-defined acceptable-error bound on one response quantity.

    Attributes:
        response_name: Which response this tolerance applies to.
        max_absolute_error: The largest acceptable absolute error in
            physical units, or ``None`` to not check absolute error.
        max_relative_error: The largest acceptable relative error
            (e.g. ``0.05`` for 5%), or ``None`` to not check it.
        description: A short, human-readable description (e.g.
            "Maximum displacement must be accurate to 5% for this
            safety-critical check.").
    """

    response_name: str
    max_absolute_error: float | None = None
    max_relative_error: float | None = None
    description: str = ""

    def __post_init__(self) -> None:
        if self.max_absolute_error is None and self.max_relative_error is None:
            raise ValidationError(
                f"EngineeringTolerance for {self.response_name!r} requires at least one of "
                "max_absolute_error/max_relative_error."
            )


@dataclass(frozen=True)
class ToleranceCheckResult:
    """The outcome of checking one :class:`EngineeringTolerance` against held-out predictions.

    Attributes:
        tolerance: The tolerance that was checked.
        observed_max_absolute_error: The worst observed absolute error.
        observed_max_relative_error: The worst observed relative error.
        passed: Whether every configured bound was satisfied.
    """

    tolerance: EngineeringTolerance
    observed_max_absolute_error: float
    observed_max_relative_error: float
    passed: bool


def check_engineering_tolerances(
    tolerances: list[EngineeringTolerance],
    actual_by_response: dict[str, np.ndarray],
    predicted_by_response: dict[str, np.ndarray],
) -> list[ToleranceCheckResult]:
    """Check every :class:`EngineeringTolerance` against held-out actual/predicted values.

    Args:
        tolerances: The tolerances to check.
        actual_by_response: True held-out values, keyed by response name.
        predicted_by_response: Predicted values, keyed by response name,
            same keys/lengths as ``actual_by_response``.

    Returns:
        One :class:`ToleranceCheckResult` per tolerance, in order.

    Raises:
        ValidationError: If a tolerance names a response not present in
            ``actual_by_response``.
    """
    from femtoolkit.surrogate.metrics import relative_error

    results: list[ToleranceCheckResult] = []
    for tolerance in tolerances:
        if tolerance.response_name not in actual_by_response:
            raise ValidationError(
                f"EngineeringTolerance names response {tolerance.response_name!r}, which is "
                f"not among the evaluated responses: {sorted(actual_by_response)}."
            )
        actual = np.asarray(actual_by_response[tolerance.response_name], dtype=float)
        predicted = np.asarray(predicted_by_response[tolerance.response_name], dtype=float)
        absolute = np.abs(actual - predicted)
        relative = relative_error(actual, predicted)
        max_absolute = float(absolute.max()) if absolute.size else 0.0
        max_relative = float(relative.max()) if relative.size else 0.0

        passed = True
        if tolerance.max_absolute_error is not None and max_absolute > tolerance.max_absolute_error:
            passed = False
        if tolerance.max_relative_error is not None and max_relative > tolerance.max_relative_error:
            passed = False

        results.append(
            ToleranceCheckResult(
                tolerance=tolerance,
                observed_max_absolute_error=max_absolute,
                observed_max_relative_error=max_relative,
                passed=passed,
            )
        )
    return results


@dataclass(frozen=True)
class CrossValidationResult:
    """The outcome of k-fold cross-validation for one response quantity.

    Attributes:
        response_name: Which response was cross-validated.
        k: The number of folds used.
        fold_scores: Each fold's held-out :math:`R^2` score, in fold order.
        mean_score: The mean of :attr:`fold_scores`.
        std_score: The sample standard deviation of :attr:`fold_scores`.
        seed: The random seed used to assign folds.
    """

    response_name: str
    k: int
    fold_scores: list[float]
    mean_score: float
    std_score: float
    seed: int | None

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable representation."""
        return {
            "response_name": self.response_name,
            "k": self.k,
            "fold_scores": list(self.fold_scores),
            "mean_score": self.mean_score,
            "std_score": self.std_score,
            "seed": self.seed,
        }


def k_fold_cross_validate(
    model_factory: Any,
    dataset: SnapshotDataset,
    response_name: str,
    k: int = 5,
    seed: int | None = None,
) -> CrossValidationResult:
    """Run k-fold cross-validation for one response of a surrogate model type.

    Args:
        model_factory: A zero-argument callable returning a fresh,
            unfitted :class:`~femtoolkit.surrogate.models.base.SurrogateModel`
            for each fold (so every fold trains an independent model
            instance -- no state leaks between folds).
        dataset: The dataset to cross-validate on.
        response_name: Which response to score (cross-validation is
            performed one response at a time, since different responses
            of the same surrogate may have very different accuracy).
        k: The number of folds. Kept deliberately simple: contiguous
            folds over a single random shuffle, not stratified.
        seed: A random seed; the same seed reproduces the same fold
            assignment.

    Returns:
        A :class:`CrossValidationResult`.

    Raises:
        InsufficientSnapshotsError: If the dataset has fewer than ``k``
            snapshots.
        ValidationError: If ``k`` is less than 2, or ``response_name``
            is not one of the dataset's response names.
    """
    if k < 2:
        raise ValidationError(f"k must be at least 2, got {k}.")
    if response_name not in dataset.response_names:
        raise ValidationError(
            f"Unknown response {response_name!r}; expected one of {dataset.response_names}."
        )
    if dataset.n_samples < k:
        raise InsufficientSnapshotsError(
            f"Dataset has only {dataset.n_samples} snapshot(s), fewer than k={k} folds."
        )

    response_index = dataset.response_names.index(response_name)
    x, y = dataset.to_arrays()
    y_column = y[:, response_index]

    rng = np.random.default_rng(seed)
    order = rng.permutation(dataset.n_samples)
    folds = np.array_split(order, k)

    scores: list[float] = []
    for fold_index in range(k):
        test_idx = folds[fold_index]
        train_idx = np.concatenate([folds[i] for i in range(k) if i != fold_index])
        model = model_factory()
        model.fit(
            x[train_idx],
            y_column[train_idx].reshape(-1, 1),
            feature_names=dataset.feature_names,
            response_names=[response_name],
        )
        predicted = model.predict(x[test_idx]).reshape(-1)
        scores.append(compute_metrics(y_column[test_idx], predicted).r2)

    return CrossValidationResult(
        response_name=response_name,
        k=k,
        fold_scores=scores,
        mean_score=float(np.mean(scores)),
        std_score=float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0,
        seed=seed,
    )


@dataclass
class SurrogateValidationReport:
    """The structured outcome of validating one trained surrogate model.

    .. code-block:: text

        model name
        dataset size
        input dimensions
        output dimensions
        training metrics
        validation metrics
        test metrics
        cross-validation metrics
        engineering tolerance checks
        warnings

    Attributes:
        model_name: The surrogate model's type name.
        dataset_id: The training dataset's identifier.
        dataset_version: The training dataset's version.
        n_samples: Total snapshots used (train + validation + test).
        n_features: Number of input dimensions.
        n_responses: Number of output dimensions.
        training_metrics: Per-response :class:`~femtoolkit.surrogate.metrics.MetricSet`
            on the training split.
        validation_metrics: Per-response metrics on the validation split
            (empty if no validation split was held out).
        test_metrics: Per-response metrics on the held-out test split.
        cross_validation: Per-response :class:`CrossValidationResult`,
            if cross-validation was run.
        tolerance_checks: Every :class:`ToleranceCheckResult` evaluated
            against the test split.
        warnings: Free-text warnings (e.g. a training-metric-only
            evaluation, a very small dataset, a failed tolerance check).
    """

    model_name: str
    dataset_id: str
    dataset_version: int
    n_samples: int
    n_features: int
    n_responses: int
    training_metrics: dict[str, MetricSet] = field(default_factory=dict)
    validation_metrics: dict[str, MetricSet] = field(default_factory=dict)
    test_metrics: dict[str, MetricSet] = field(default_factory=dict)
    cross_validation: dict[str, CrossValidationResult] = field(default_factory=dict)
    tolerance_checks: list[ToleranceCheckResult] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def meets_engineering_tolerances(self) -> bool:
        """Whether every configured :class:`EngineeringTolerance` was satisfied.

        Returns ``True`` (vacuously) if no tolerance was configured --
        this is never interpreted elsewhere as "the model is safe"; a
        caller must configure at least one tolerance for this property
        to carry engineering meaning. See the module docstring.
        """
        return all(check.passed for check in self.tolerance_checks)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable representation of this report."""
        return {
            "model_name": self.model_name,
            "dataset_id": self.dataset_id,
            "dataset_version": self.dataset_version,
            "n_samples": self.n_samples,
            "n_features": self.n_features,
            "n_responses": self.n_responses,
            "training_metrics": {k: v.to_dict() for k, v in self.training_metrics.items()},
            "validation_metrics": {k: v.to_dict() for k, v in self.validation_metrics.items()},
            "test_metrics": {k: v.to_dict() for k, v in self.test_metrics.items()},
            "cross_validation": {k: v.to_dict() for k, v in self.cross_validation.items()},
            "tolerance_checks": [
                {
                    "response_name": check.tolerance.response_name,
                    "observed_max_absolute_error": check.observed_max_absolute_error,
                    "observed_max_relative_error": check.observed_max_relative_error,
                    "passed": check.passed,
                }
                for check in self.tolerance_checks
            ],
            "warnings": list(self.warnings),
            "meets_engineering_tolerances": self.meets_engineering_tolerances,
        }


__all__ = [
    "CrossValidationResult",
    "EngineeringTolerance",
    "SurrogateValidationReport",
    "ToleranceCheckResult",
    "check_engineering_tolerances",
    "k_fold_cross_validate",
]
