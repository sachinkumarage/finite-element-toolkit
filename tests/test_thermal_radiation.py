"""Tests for femtoolkit.thermal.thermal_boundary_conditions: radiation (Version 21)."""

import pytest

from femtoolkit.exceptions import ValidationError
from femtoolkit.thermal.thermal_boundary_conditions import (
    STEFAN_BOLTZMANN_CONSTANT,
    RadiationBoundaryCondition,
    radiative_heat_flux,
)
from femtoolkit.thermal.thermal_surfaces import ThermalSurface

_SURFACE = ThermalSurface(element_id=1, local_face_index=0)


def test_stefan_boltzmann_constant_value() -> None:
    assert pytest.approx(5.670374419e-8) == STEFAN_BOLTZMANN_CONSTANT


def test_radiative_heat_flux_matches_stefan_boltzmann_formula() -> None:
    emissivity, surface_t, surrounding_t = 0.85, 500.0, 300.0
    expected = emissivity * STEFAN_BOLTZMANN_CONSTANT * (surface_t**4 - surrounding_t**4)
    assert radiative_heat_flux(emissivity, surface_t, surrounding_t) == pytest.approx(expected)


def test_radiative_heat_flux_zero_at_equal_temperatures() -> None:
    assert radiative_heat_flux(0.9, 350.0, 350.0) == pytest.approx(0.0)


def test_radiative_heat_flux_positive_when_hotter_than_surroundings() -> None:
    assert radiative_heat_flux(0.9, 500.0, 300.0) > 0.0


def test_radiative_heat_flux_negative_when_colder_than_surroundings() -> None:
    assert radiative_heat_flux(0.9, 250.0, 300.0) < 0.0


def test_radiative_heat_flux_scales_with_emissivity() -> None:
    low = radiative_heat_flux(0.2, 500.0, 300.0)
    high = radiative_heat_flux(0.8, 500.0, 300.0)
    assert high == pytest.approx(4.0 * low)


def test_radiative_heat_flux_rejects_negative_surface_temperature() -> None:
    with pytest.raises(ValidationError):
        radiative_heat_flux(0.9, -10.0, 300.0)


def test_radiative_heat_flux_rejects_negative_surrounding_temperature() -> None:
    with pytest.raises(ValidationError):
        radiative_heat_flux(0.9, 300.0, -10.0)


def test_radiative_heat_flux_accepts_absolute_zero() -> None:
    assert radiative_heat_flux(0.9, 0.0, 0.0) == pytest.approx(0.0)


def test_radiation_boundary_condition_stores_values() -> None:
    bc = RadiationBoundaryCondition(
        surface=_SURFACE, emissivity=0.85, surrounding_temperature=293.15
    )
    assert bc.emissivity == pytest.approx(0.85)
    assert bc.surrounding_temperature == pytest.approx(293.15)


@pytest.mark.parametrize("emissivity", [0.0, -0.1, 1.1, float("nan")])
def test_radiation_boundary_condition_rejects_invalid_emissivity(emissivity: float) -> None:
    with pytest.raises(ValidationError):
        RadiationBoundaryCondition(
            surface=_SURFACE, emissivity=emissivity, surrounding_temperature=300.0
        )


def test_radiation_boundary_condition_accepts_emissivity_of_one() -> None:
    bc = RadiationBoundaryCondition(surface=_SURFACE, emissivity=1.0, surrounding_temperature=300.0)
    assert bc.emissivity == pytest.approx(1.0)


@pytest.mark.parametrize("temperature", [-1.0, float("nan"), float("inf")])
def test_radiation_boundary_condition_rejects_invalid_surrounding_temperature(
    temperature: float,
) -> None:
    with pytest.raises(ValidationError):
        RadiationBoundaryCondition(
            surface=_SURFACE, emissivity=0.9, surrounding_temperature=temperature
        )


def test_radiation_boundary_condition_accepts_zero_kelvin_surroundings() -> None:
    bc = RadiationBoundaryCondition(surface=_SURFACE, emissivity=0.9, surrounding_temperature=0.0)
    assert bc.surrounding_temperature == pytest.approx(0.0)
