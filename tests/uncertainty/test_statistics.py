"""Tests for femtoolkit.uncertainty.statistics."""

import numpy as np
import pytest

from femtoolkit.exceptions import ValidationError
from femtoolkit.uncertainty.statistics import compute_convergence_series, compute_output_statistics


def test_compute_output_statistics_known_dataset() -> None:
    values = np.arange(1, 11, dtype=float)
    stats = compute_output_statistics(values, "x")
    assert stats.mean == pytest.approx(5.5)
    assert stats.median == pytest.approx(5.5)
    assert stats.minimum == 1.0
    assert stats.maximum == 10.0
    assert stats.std == pytest.approx(np.std(values, ddof=1))
    assert stats.percentiles[50.0] == pytest.approx(np.percentile(values, 50))
    assert stats.n_successful == 10


def test_compute_output_statistics_coefficient_of_variation() -> None:
    values = np.array([90.0, 100.0, 110.0])
    stats = compute_output_statistics(values, "x")
    assert stats.coefficient_of_variation == pytest.approx(stats.std / 100.0)


def test_compute_output_statistics_empty_returns_none_fields() -> None:
    stats = compute_output_statistics(np.array([]), "x", n_requested=5, n_failed=3, n_invalid=2)
    assert stats.mean is None
    assert stats.std is None
    assert stats.percentiles == {}
    assert stats.n_requested == 5
    assert stats.n_failed == 3
    assert stats.n_invalid == 2


def test_compute_output_statistics_single_sample_has_no_std() -> None:
    stats = compute_output_statistics(np.array([42.0]), "x")
    assert stats.mean == 42.0
    assert stats.std is None
    assert stats.variance is None
    assert stats.coefficient_of_variation is None


def test_compute_output_statistics_rejects_invalid_percentile() -> None:
    with pytest.raises(ValidationError):
        compute_output_statistics(np.array([1.0, 2.0]), "x", percentiles=(150.0,))


def test_compute_output_statistics_n_requested_defaults_to_total() -> None:
    stats = compute_output_statistics(np.array([1.0, 2.0]), "x", n_failed=1, n_invalid=1)
    assert stats.n_requested == 4


def test_compute_convergence_series_known_dataset() -> None:
    values = np.array([10.0, 20.0, 30.0])
    series = compute_convergence_series(values)
    assert len(series) == 3
    assert series[0].n == 1 and series[0].running_mean == 10.0 and series[0].standard_error is None
    assert series[1].running_mean == pytest.approx(15.0)
    assert series[1].standard_error is not None
    assert series[2].running_mean == pytest.approx(20.0)


def test_compute_convergence_series_rejects_empty() -> None:
    with pytest.raises(ValidationError):
        compute_convergence_series(np.array([]))
