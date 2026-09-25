"""Tests for femtoolkit.uncertainty.sampling."""

import numpy as np
import pytest
from scipy.stats import norm

from femtoolkit.exceptions import ValidationError
from femtoolkit.uncertainty.distributions import NormalDistribution, UniformDistribution
from femtoolkit.uncertainty.parameters import UncertainParameter
from femtoolkit.uncertainty.sampling import (
    generate_latin_hypercube_samples,
    generate_random_samples,
    generate_samples,
)

_PARAMETERS = [
    UncertainParameter(
        path="material.youngs_modulus", label="E", distribution=NormalDistribution(200e9, 5e9)
    ),
    UncertainParameter(
        path="loads.0.magnitude", label="Load", distribution=UniformDistribution(-2000.0, -1000.0)
    ),
]


def test_generate_random_samples_shape_and_columns() -> None:
    sample_set = generate_random_samples(_PARAMETERS, 50, seed=1)
    assert sample_set.n_samples == 50
    assert sample_set.values.shape == (50, 2)
    assert sample_set.parameter_paths == ["material.youngs_modulus", "loads.0.magnitude"]
    assert sample_set.method == "random"


def test_generate_random_samples_reproducible_with_same_seed() -> None:
    a = generate_random_samples(_PARAMETERS, 30, seed=42)
    b = generate_random_samples(_PARAMETERS, 30, seed=42)
    assert np.array_equal(a.values, b.values)


def test_generate_random_samples_differs_with_different_seed() -> None:
    a = generate_random_samples(_PARAMETERS, 30, seed=1)
    b = generate_random_samples(_PARAMETERS, 30, seed=2)
    assert not np.array_equal(a.values, b.values)


def test_generate_random_samples_requires_at_least_one_parameter() -> None:
    with pytest.raises(ValidationError):
        generate_random_samples([], 10)


def test_generate_random_samples_requires_positive_count() -> None:
    with pytest.raises(ValidationError):
        generate_random_samples(_PARAMETERS, 0)


def test_generate_latin_hypercube_samples_reproducible() -> None:
    a = generate_latin_hypercube_samples(_PARAMETERS, 20, seed=7)
    b = generate_latin_hypercube_samples(_PARAMETERS, 20, seed=7)
    assert np.array_equal(a.values, b.values)


def test_generate_latin_hypercube_samples_stratifies_each_parameter() -> None:
    """Each of the N probability intervals must contain exactly one sample."""
    n = 25
    sample_set = generate_latin_hypercube_samples(_PARAMETERS, n, seed=3)
    e_column = sample_set.column("material.youngs_modulus")
    quantiles = norm.cdf(e_column, loc=200e9, scale=5e9)
    bins = np.clip(np.floor(quantiles * n).astype(int), 0, n - 1)
    counts = np.bincount(bins, minlength=n)
    assert np.all(counts == 1)


def test_generate_samples_dispatches_to_random_and_lhs() -> None:
    random_set = generate_samples(_PARAMETERS, 10, method="random", seed=1)
    assert random_set.method == "random"
    lhs_set = generate_samples(_PARAMETERS, 10, method="latin_hypercube", seed=1)
    assert lhs_set.method == "latin_hypercube"


def test_generate_samples_rejects_unknown_method() -> None:
    with pytest.raises(ValidationError):
        generate_samples(_PARAMETERS, 10, method="not_a_method")


def test_sample_set_row_and_column() -> None:
    sample_set = generate_random_samples(_PARAMETERS, 10, seed=1)
    row = sample_set.row(0)
    assert set(row) == {"material.youngs_modulus", "loads.0.magnitude"}
    column = sample_set.column("loads.0.magnitude")
    assert column.shape == (10,)
    assert row["loads.0.magnitude"] == pytest.approx(column[0])


def test_sample_set_column_rejects_unknown_path() -> None:
    sample_set = generate_random_samples(_PARAMETERS, 10, seed=1)
    with pytest.raises(ValidationError):
        sample_set.column("not.a.real.path")
