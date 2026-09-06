"""Tests for femtoolkit.materials.thermal_properties."""

import pytest

from femtoolkit.exceptions import ValidationError
from femtoolkit.materials.thermal_properties import (
    TemperatureDependentProperty,
    evaluate_thermal_property,
)


def test_evaluate_constant_property() -> None:
    assert evaluate_thermal_property(200e9, temperature=373.15) == pytest.approx(200e9)


def test_evaluate_user_defined_function_property() -> None:
    def linear_softening(temperature: float) -> float:
        return 200e9 - 1e8 * (temperature - 293.15)

    assert evaluate_thermal_property(linear_softening, temperature=393.15) == pytest.approx(
        200e9 - 1e8 * 100.0
    )


def test_tabulated_property_matches_table_at_exact_points() -> None:
    youngs_modulus = TemperatureDependentProperty(
        temperatures=(293.15, 373.15, 473.15), values=(200e9, 195e9, 185e9)
    )
    assert youngs_modulus(293.15) == pytest.approx(200e9)
    assert youngs_modulus(373.15) == pytest.approx(195e9)
    assert youngs_modulus(473.15) == pytest.approx(185e9)


def test_tabulated_property_interpolates_linearly_between_points() -> None:
    youngs_modulus = TemperatureDependentProperty(
        temperatures=(293.15, 373.15), values=(200e9, 195e9)
    )
    midpoint = (293.15 + 373.15) / 2.0
    assert youngs_modulus(midpoint) == pytest.approx((200e9 + 195e9) / 2.0)


def test_tabulated_alpha_property() -> None:
    alpha = TemperatureDependentProperty(temperatures=(293.15, 473.15), values=(11e-6, 14e-6))
    assert alpha(383.15) == pytest.approx((11e-6 + 14e-6) / 2.0)


def test_evaluate_thermal_property_dispatches_to_tabulated_property() -> None:
    alpha = TemperatureDependentProperty(temperatures=(293.15, 473.15), values=(11e-6, 14e-6))
    assert evaluate_thermal_property(alpha, temperature=293.15) == pytest.approx(11e-6)


def test_below_range_temperature_raises() -> None:
    youngs_modulus = TemperatureDependentProperty(
        temperatures=(293.15, 373.15), values=(200e9, 195e9)
    )
    with pytest.raises(ValidationError):
        youngs_modulus(200.0)


def test_above_range_temperature_raises() -> None:
    youngs_modulus = TemperatureDependentProperty(
        temperatures=(293.15, 373.15), values=(200e9, 195e9)
    )
    with pytest.raises(ValidationError):
        youngs_modulus(500.0)


def test_non_finite_temperature_raises() -> None:
    youngs_modulus = TemperatureDependentProperty(
        temperatures=(293.15, 373.15), values=(200e9, 195e9)
    )
    with pytest.raises(ValidationError):
        youngs_modulus(float("nan"))


def test_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValidationError):
        TemperatureDependentProperty(temperatures=(293.15, 373.15, 473.15), values=(200e9, 195e9))


def test_rejects_fewer_than_two_points() -> None:
    with pytest.raises(ValidationError):
        TemperatureDependentProperty(temperatures=(293.15,), values=(200e9,))


def test_rejects_non_increasing_temperatures() -> None:
    with pytest.raises(ValidationError):
        TemperatureDependentProperty(temperatures=(373.15, 293.15), values=(195e9, 200e9))


def test_rejects_non_finite_values() -> None:
    with pytest.raises(ValidationError):
        TemperatureDependentProperty(temperatures=(293.15, 373.15), values=(200e9, float("nan")))
