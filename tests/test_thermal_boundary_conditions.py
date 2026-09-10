"""Tests for femtoolkit.thermal.thermal_boundary_conditions."""

import pytest

from femtoolkit.exceptions import ValidationError
from femtoolkit.thermal.thermal_boundary_conditions import PrescribedHeatFlux, PrescribedTemperature


def test_prescribed_temperature_stores_value() -> None:
    bc = PrescribedTemperature(node_id=1, value=373.15)
    assert bc.node_id == 1
    assert bc.value == pytest.approx(373.15)


def test_prescribed_heat_flux_stores_value() -> None:
    bc = PrescribedHeatFlux(node_id=2, value=100.0)
    assert bc.node_id == 2
    assert bc.value == pytest.approx(100.0)


def test_prescribed_heat_flux_can_be_negative() -> None:
    """Negative means heat flowing out of the body."""
    bc = PrescribedHeatFlux(node_id=2, value=-50.0)
    assert bc.value == pytest.approx(-50.0)


def test_prescribed_temperature_rejects_non_finite_value() -> None:
    with pytest.raises(ValidationError):
        PrescribedTemperature(node_id=1, value=float("nan"))


def test_prescribed_temperature_rejects_infinite_value() -> None:
    with pytest.raises(ValidationError):
        PrescribedTemperature(node_id=1, value=float("inf"))


def test_prescribed_heat_flux_rejects_non_finite_value() -> None:
    with pytest.raises(ValidationError):
        PrescribedHeatFlux(node_id=1, value=float("nan"))
