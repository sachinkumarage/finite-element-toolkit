"""Tests for femtoolkit.surrogate.metrics."""

from __future__ import annotations

import numpy as np

from femtoolkit.surrogate.metrics import (
    compute_metrics,
    mean_absolute_error,
    r_squared,
    relative_error,
    root_mean_square_error,
)


def test_metrics_zero_for_perfect_prediction() -> None:
    actual = np.array([1.0, 2.0, 3.0])
    assert mean_absolute_error(actual, actual) == 0.0
    assert root_mean_square_error(actual, actual) == 0.0
    assert r_squared(actual, actual) == 1.0
    np.testing.assert_allclose(relative_error(actual, actual), 0.0)


def test_mae_rmse_known_values() -> None:
    actual = np.array([0.0, 0.0])
    predicted = np.array([3.0, 4.0])
    assert mean_absolute_error(actual, predicted) == 3.5
    assert root_mean_square_error(actual, predicted) == np.sqrt((9 + 16) / 2)


def test_relative_error_handles_zero_response_safely() -> None:
    actual = np.array([0.0])
    predicted = np.array([0.001])
    result = relative_error(actual, predicted)
    assert np.isfinite(result).all()
    assert result[0] > 0.0


def test_r_squared_zero_variance_edge_cases() -> None:
    actual = np.array([5.0, 5.0, 5.0])
    assert r_squared(actual, actual) == 1.0
    assert r_squared(actual, np.array([5.0, 5.0, 6.0])) == 0.0


def test_compute_metrics_bundle() -> None:
    actual = np.array([1.0, 2.0, 3.0, 4.0])
    predicted = np.array([1.1, 1.9, 3.2, 3.8])
    metrics = compute_metrics(actual, predicted)
    assert metrics.n_samples == 4
    assert metrics.mae > 0.0
    assert metrics.rmse > 0.0
    assert 0.0 < metrics.r2 <= 1.0
    assert metrics.max_relative_error >= metrics.mean_relative_error
    as_dict = metrics.to_dict()
    assert as_dict["n_samples"] == 4
