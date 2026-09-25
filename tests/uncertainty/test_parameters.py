"""Tests for femtoolkit.uncertainty.parameters."""

import pytest

from femtoolkit.exceptions import ValidationError
from femtoolkit.uncertainty.distributions import NormalDistribution, UniformDistribution
from femtoolkit.uncertainty.parameters import UncertainParameter, UncertaintyCategory


def test_creation_with_defaults() -> None:
    parameter = UncertainParameter(
        path="material.youngs_modulus",
        label="Young's Modulus",
        distribution=NormalDistribution(200e9, 5e9),
    )
    assert parameter.category == UncertaintyCategory.ALEATORY
    assert parameter.units == ""
    assert parameter.physical_lower_bound is None


def test_physical_bounds_combine_with_distribution_bounds() -> None:
    parameter = UncertainParameter(
        path="x", label="x", distribution=UniformDistribution(-5.0, 5.0), physical_lower_bound=0.0
    )
    assert parameter.effective_bounds() == (0.0, 5.0)


def test_effective_bounds_with_no_bounds_at_all() -> None:
    parameter = UncertainParameter(path="x", label="x", distribution=NormalDistribution(0.0, 1.0))
    assert parameter.effective_bounds() == (None, None)


def test_is_physically_valid_respects_lower_bound() -> None:
    parameter = UncertainParameter(
        path="material.youngs_modulus",
        label="E",
        distribution=NormalDistribution(200e9, 5e9),
        physical_lower_bound=0.0,
    )
    assert parameter.is_physically_valid(200e9)
    assert not parameter.is_physically_valid(0.0)
    assert not parameter.is_physically_valid(-1.0)


def test_is_physically_valid_rejects_non_finite() -> None:
    parameter = UncertainParameter(path="x", label="x", distribution=NormalDistribution(0.0, 1.0))
    assert not parameter.is_physically_valid(float("nan"))
    assert not parameter.is_physically_valid(float("inf"))


def test_physical_lower_bound_must_be_less_than_upper_bound() -> None:
    with pytest.raises(ValidationError):
        UncertainParameter(
            path="x",
            label="x",
            distribution=NormalDistribution(0.0, 1.0),
            physical_lower_bound=10.0,
            physical_upper_bound=5.0,
        )


def test_reference_value_must_be_physically_valid() -> None:
    with pytest.raises(ValidationError):
        UncertainParameter(
            path="material.youngs_modulus",
            label="E",
            distribution=NormalDistribution(200e9, 5e9),
            physical_lower_bound=0.0,
            reference_value=-5.0,
        )


def test_reference_value_within_bounds_is_accepted() -> None:
    parameter = UncertainParameter(
        path="material.youngs_modulus",
        label="E",
        distribution=NormalDistribution(200e9, 5e9),
        physical_lower_bound=0.0,
        reference_value=200e9,
    )
    assert parameter.reference_value == 200e9


def test_to_dict_from_dict_round_trip() -> None:
    parameter = UncertainParameter(
        path="material.youngs_modulus",
        label="Young's Modulus",
        distribution=NormalDistribution(200e9, 5e9),
        units="Pa",
        description="Steel elastic modulus.",
        category=UncertaintyCategory.EPISTEMIC,
        physical_lower_bound=0.0,
        reference_value=200e9,
    )
    restored = UncertainParameter.from_dict(parameter.to_dict())
    assert restored == parameter
