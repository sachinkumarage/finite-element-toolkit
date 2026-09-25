"""Tests for femtoolkit.uncertainty.confidence."""

import numpy as np
import pytest

from femtoolkit.exceptions import ValidationError
from femtoolkit.uncertainty.confidence import confidence_interval_mean


def test_confidence_interval_contains_true_mean_for_well_behaved_sample() -> None:
    rng = np.random.default_rng(0)
    sample = rng.normal(100.0, 10.0, size=1000)
    ci = confidence_interval_mean(sample, "y", confidence_level=0.95)
    assert ci.lower < 100.0 < ci.upper
    assert ci.lower < ci.mean < ci.upper


def test_confidence_interval_widens_for_higher_confidence_level() -> None:
    rng = np.random.default_rng(1)
    sample = rng.normal(0.0, 1.0, size=500)
    ci_90 = confidence_interval_mean(sample, "y", confidence_level=0.90)
    ci_99 = confidence_interval_mean(sample, "y", confidence_level=0.99)
    width_90 = ci_90.upper - ci_90.lower
    width_99 = ci_99.upper - ci_99.lower
    assert width_99 > width_90


def test_confidence_interval_degrees_of_freedom_and_n() -> None:
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    ci = confidence_interval_mean(values, "y")
    assert ci.n_samples == 5
    assert ci.degrees_of_freedom == 4
    assert ci.mean == pytest.approx(3.0)


def test_confidence_interval_rejects_invalid_confidence_level() -> None:
    values = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValidationError):
        confidence_interval_mean(values, "y", confidence_level=0.0)
    with pytest.raises(ValidationError):
        confidence_interval_mean(values, "y", confidence_level=1.0)


def test_confidence_interval_requires_at_least_two_samples() -> None:
    with pytest.raises(ValidationError):
        confidence_interval_mean(np.array([1.0]), "y")
    with pytest.raises(ValidationError):
        confidence_interval_mean(np.array([]), "y")
