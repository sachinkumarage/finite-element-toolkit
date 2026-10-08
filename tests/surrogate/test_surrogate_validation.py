"""Tests for femtoolkit.surrogate.validation."""

from __future__ import annotations

import numpy as np
import pytest

from femtoolkit.exceptions import InsufficientSnapshotsError, ValidationError
from femtoolkit.surrogate.datasets import Snapshot, SnapshotDataset
from femtoolkit.surrogate.models.polynomial import PolynomialRegressionSurrogate
from femtoolkit.surrogate.validation import (
    EngineeringTolerance,
    check_engineering_tolerances,
    k_fold_cross_validate,
)


def _linear_dataset(n: int = 20) -> SnapshotDataset:
    rng = np.random.default_rng(0)
    snapshots = []
    for i in range(n):
        x = rng.uniform(-5, 5)
        snapshots.append(
            Snapshot(snapshot_id=str(i), inputs={"x": x}, outputs={"y": 2.0 * x + 1.0})
        )
    return SnapshotDataset(feature_names=["x"], response_names=["y"], snapshots=snapshots)


def test_engineering_tolerance_requires_a_bound() -> None:
    with pytest.raises(ValidationError):
        EngineeringTolerance(response_name="y")


def test_check_engineering_tolerances_pass_and_fail() -> None:
    tolerance = EngineeringTolerance(response_name="y", max_absolute_error=0.5)
    actual = {"y": np.array([1.0, 2.0, 3.0])}
    close_predictions = {"y": np.array([1.1, 1.9, 3.05])}
    far_predictions = {"y": np.array([1.0, 5.0, 3.0])}

    passing = check_engineering_tolerances([tolerance], actual, close_predictions)
    assert passing[0].passed is True

    failing = check_engineering_tolerances([tolerance], actual, far_predictions)
    assert failing[0].passed is False


def test_check_engineering_tolerances_unknown_response_raises() -> None:
    tolerance = EngineeringTolerance(response_name="unknown", max_absolute_error=1.0)
    with pytest.raises(ValidationError):
        check_engineering_tolerances([tolerance], {"y": np.array([1.0])}, {"y": np.array([1.0])})


def test_k_fold_cross_validate_perfect_linear_fit() -> None:
    dataset = _linear_dataset(20)
    result = k_fold_cross_validate(
        model_factory=lambda: PolynomialRegressionSurrogate(degree=1),
        dataset=dataset,
        response_name="y",
        k=5,
        seed=0,
    )
    assert result.k == 5
    assert len(result.fold_scores) == 5
    assert result.mean_score > 0.99


def test_k_fold_cross_validate_requires_enough_snapshots() -> None:
    dataset = _linear_dataset(3)
    with pytest.raises(InsufficientSnapshotsError):
        k_fold_cross_validate(
            model_factory=lambda: PolynomialRegressionSurrogate(degree=1),
            dataset=dataset,
            response_name="y",
            k=5,
        )


def test_k_fold_cross_validate_rejects_small_k() -> None:
    dataset = _linear_dataset(10)
    with pytest.raises(ValidationError):
        k_fold_cross_validate(
            model_factory=lambda: PolynomialRegressionSurrogate(degree=1),
            dataset=dataset,
            response_name="y",
            k=1,
        )


def test_k_fold_cross_validate_reproducible_with_same_seed() -> None:
    dataset = _linear_dataset(20)
    first = k_fold_cross_validate(
        model_factory=lambda: PolynomialRegressionSurrogate(degree=1),
        dataset=dataset, response_name="y", k=4, seed=7,
    )
    second = k_fold_cross_validate(
        model_factory=lambda: PolynomialRegressionSurrogate(degree=1),
        dataset=dataset, response_name="y", k=4, seed=7,
    )
    assert first.fold_scores == second.fold_scores
