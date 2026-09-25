"""Tests for femtoolkit.uncertainty.correlation."""

import numpy as np
import pytest

from femtoolkit.exceptions import ValidationError
from femtoolkit.uncertainty.correlation import (
    correlation_summary,
    pearson_correlation,
    spearman_correlation,
)


def test_pearson_correlation_perfect_positive_linear() -> None:
    x = np.arange(1, 21, dtype=float)
    y = 3.0 * x + 7.0
    assert pearson_correlation(x, y) == pytest.approx(1.0)


def test_pearson_correlation_perfect_negative_linear() -> None:
    x = np.arange(1, 21, dtype=float)
    y = -2.0 * x + 1.0
    assert pearson_correlation(x, y) == pytest.approx(-1.0)


def test_pearson_correlation_near_zero_for_unrelated_data() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=5000)
    y = rng.normal(size=5000)
    assert abs(pearson_correlation(x, y)) < 0.05


def test_pearson_correlation_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValidationError):
        pearson_correlation(np.array([1.0, 2.0]), np.array([1.0]))


def test_pearson_correlation_rejects_too_few_samples() -> None:
    with pytest.raises(ValidationError):
        pearson_correlation(np.array([1.0]), np.array([1.0]))


def test_pearson_correlation_rejects_zero_variance() -> None:
    with pytest.raises(ValidationError):
        pearson_correlation(np.array([1.0, 1.0, 1.0]), np.array([1.0, 2.0, 3.0]))


def test_spearman_correlation_perfect_for_monotonic_nonlinear_relationship() -> None:
    x = np.arange(1, 21, dtype=float)
    y = x**3
    assert spearman_correlation(x, y) == pytest.approx(1.0)


def test_spearman_captures_monotonic_relation_pearson_understates() -> None:
    x = np.arange(1, 21, dtype=float)
    y = x**5
    rho = spearman_correlation(x, y)
    r = pearson_correlation(x, y)
    assert rho == pytest.approx(1.0)
    assert r < 1.0


def test_correlation_summary_sorted_by_descending_magnitude() -> None:
    x = np.arange(1, 51, dtype=float)
    strong = 5.0 * x
    rng = np.random.default_rng(1)
    weak = rng.normal(size=x.size)

    summary = correlation_summary(
        "Displacement",
        {
            "weak_param": ("Weak", x, weak),
            "strong_param": ("Strong", x, strong),
        },
    )
    assert summary[0].parameter_label == "Strong"
    assert abs(summary[0].pearson_r) > abs(summary[1].pearson_r)
    assert all(result.quantity_label == "Displacement" for result in summary)
