"""Tests for femtoolkit.uncertainty.distributions."""

import numpy as np
import pytest

from femtoolkit.exceptions import ValidationError
from femtoolkit.uncertainty.distributions import (
    DeterministicDistribution,
    LognormalDistribution,
    NormalDistribution,
    UniformDistribution,
    distribution_from_dict,
    distribution_to_dict,
)

# --- DeterministicDistribution ---


def test_deterministic_sample_returns_constant_array() -> None:
    dist = DeterministicDistribution(5.0)
    samples = dist.sample(np.random.default_rng(0), 100)
    assert np.all(samples == 5.0)
    assert samples.shape == (100,)


def test_deterministic_mean_std_bounds() -> None:
    dist = DeterministicDistribution(7.5)
    assert dist.mean() == 7.5
    assert dist.std() == 0.0
    assert dist.bounds() == (7.5, 7.5)


def test_deterministic_ppf_is_constant() -> None:
    dist = DeterministicDistribution(3.0)
    result = dist.ppf(np.array([0.0, 0.25, 0.5, 1.0]))
    assert np.all(result == 3.0)


def test_deterministic_ppf_rejects_out_of_range_quantile() -> None:
    dist = DeterministicDistribution(3.0)
    with pytest.raises(ValidationError):
        dist.ppf(np.array([1.5]))


def test_deterministic_pdf_raises_not_implemented() -> None:
    dist = DeterministicDistribution(3.0)
    with pytest.raises(NotImplementedError):
        dist.pdf(np.array([3.0]))


def test_deterministic_rejects_non_finite_value() -> None:
    with pytest.raises(ValidationError):
        DeterministicDistribution(float("inf"))


# --- UniformDistribution ---


def test_uniform_requires_low_less_than_high() -> None:
    with pytest.raises(ValidationError):
        UniformDistribution(5.0, 5.0)
    with pytest.raises(ValidationError):
        UniformDistribution(5.0, 1.0)


def test_uniform_mean_and_std_match_closed_form() -> None:
    dist = UniformDistribution(1.0, 3.0)
    assert dist.mean() == pytest.approx(2.0)
    assert dist.std() == pytest.approx((3.0 - 1.0) / np.sqrt(12.0))


def test_uniform_sample_stays_within_bounds() -> None:
    dist = UniformDistribution(-2.0, 5.0)
    samples = dist.sample(np.random.default_rng(1), 5000)
    assert samples.min() >= -2.0
    assert samples.max() <= 5.0


def test_uniform_sample_statistics_converge_to_closed_form() -> None:
    dist = UniformDistribution(10.0, 20.0)
    samples = dist.sample(np.random.default_rng(2), 200_000)
    assert samples.mean() == pytest.approx(dist.mean(), abs=0.05)
    assert samples.std() == pytest.approx(dist.std(), abs=0.05)


def test_uniform_ppf_endpoints() -> None:
    dist = UniformDistribution(1.0, 3.0)
    result = dist.ppf(np.array([0.0, 0.5, 1.0]))
    assert result.tolist() == pytest.approx([1.0, 2.0, 3.0])


def test_uniform_pdf_zero_outside_support() -> None:
    dist = UniformDistribution(0.0, 2.0)
    density = dist.pdf(np.array([-1.0, 1.0, 3.0]))
    assert density[0] == 0.0
    assert density[1] == pytest.approx(0.5)
    assert density[2] == 0.0


# --- NormalDistribution ---


def test_normal_requires_positive_std() -> None:
    with pytest.raises(ValidationError):
        NormalDistribution(0.0, 0.0)
    with pytest.raises(ValidationError):
        NormalDistribution(0.0, -1.0)


def test_normal_mean_std_and_bounds() -> None:
    dist = NormalDistribution(200e9, 5e9)
    assert dist.mean() == 200e9
    assert dist.std() == 5e9
    assert dist.bounds() == (None, None)


def test_normal_ppf_median_equals_mean() -> None:
    dist = NormalDistribution(100.0, 10.0)
    assert dist.ppf(np.array([0.5]))[0] == pytest.approx(100.0, abs=1e-6)


def test_normal_sample_statistics_converge() -> None:
    dist = NormalDistribution(50.0, 4.0)
    samples = dist.sample(np.random.default_rng(3), 200_000)
    assert samples.mean() == pytest.approx(50.0, abs=0.1)
    assert samples.std() == pytest.approx(4.0, abs=0.1)


# --- LognormalDistribution ---


def test_lognormal_requires_positive_sigma() -> None:
    with pytest.raises(ValidationError):
        LognormalDistribution(0.0, 0.0)


def test_lognormal_bounds_are_nonnegative() -> None:
    dist = LognormalDistribution(0.0, 0.2)
    assert dist.bounds() == (0.0, None)


def test_lognormal_from_mean_std_matches_requested_moments() -> None:
    dist = LognormalDistribution.from_mean_std(200e9, 10e9)
    assert dist.mean() == pytest.approx(200e9, rel=1e-6)
    assert dist.std() == pytest.approx(10e9, rel=1e-6)


def test_lognormal_from_mean_std_requires_positive_inputs() -> None:
    with pytest.raises(ValidationError):
        LognormalDistribution.from_mean_std(-1.0, 1.0)
    with pytest.raises(ValidationError):
        LognormalDistribution.from_mean_std(1.0, -1.0)


def test_lognormal_samples_are_always_positive() -> None:
    dist = LognormalDistribution.from_mean_std(100.0, 50.0)
    samples = dist.sample(np.random.default_rng(4), 10_000)
    assert np.all(samples > 0.0)


def test_lognormal_sample_statistics_converge() -> None:
    dist = LognormalDistribution.from_mean_std(200.0, 20.0)
    samples = dist.sample(np.random.default_rng(5), 300_000)
    assert samples.mean() == pytest.approx(200.0, rel=0.02)
    assert samples.std() == pytest.approx(20.0, rel=0.05)


# --- serialization ---


@pytest.mark.parametrize(
    "dist",
    [
        DeterministicDistribution(5.0),
        UniformDistribution(1.0, 3.0),
        NormalDistribution(200e9, 5e9),
        LognormalDistribution(0.0, 0.2),
    ],
)
def test_distribution_round_trips_through_dict(dist) -> None:
    data = distribution_to_dict(dist)
    assert "type" in data
    restored = distribution_from_dict(data)
    assert restored == dist


def test_distribution_from_dict_rejects_unknown_type() -> None:
    with pytest.raises(ValidationError):
        distribution_from_dict({"type": "not_a_real_distribution"})
