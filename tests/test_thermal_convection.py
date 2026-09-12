"""Tests for femtoolkit.thermal.thermal_boundary_conditions: convection (Version 21)."""

import pytest

from femtoolkit.analysis.dynamic_loads import ConstantLoad, SinusoidalLoad, StepLoad
from femtoolkit.exceptions import ValidationError
from femtoolkit.materials.thermal_properties import TemperatureDependentProperty
from femtoolkit.thermal.thermal_boundary_conditions import (
    ConvectionBoundaryCondition,
    convective_heat_flux,
)
from femtoolkit.thermal.thermal_surfaces import ThermalSurface

_SURFACE = ThermalSurface(element_id=1, local_face_index=0)


def test_convective_heat_flux_hotter_surface_loses_heat() -> None:
    flux = convective_heat_flux(
        coefficient=25.0, surface_temperature=350.0, ambient_temperature=300.0
    )
    assert flux == pytest.approx(25.0 * 50.0)


def test_convective_heat_flux_colder_surface_gains_heat() -> None:
    flux = convective_heat_flux(
        coefficient=25.0, surface_temperature=280.0, ambient_temperature=300.0
    )
    assert flux < 0.0
    assert flux == pytest.approx(25.0 * -20.0)


def test_convective_heat_flux_zero_when_at_ambient() -> None:
    assert convective_heat_flux(10.0, 300.0, 300.0) == pytest.approx(0.0)


def test_constant_convection_coefficient_stores_values() -> None:
    bc = ConvectionBoundaryCondition(
        surface=_SURFACE, convection_coefficient=25.0, ambient_temperature=293.15
    )
    assert not bc.is_temperature_dependent
    assert bc.convection_coefficient_at(400.0) == pytest.approx(25.0)
    assert bc.ambient_temperature_at(1000.0) == pytest.approx(293.15)


def test_temperature_dependent_convection_coefficient_via_callable() -> None:
    bc = ConvectionBoundaryCondition(
        surface=_SURFACE,
        convection_coefficient=lambda t: 10.0 + 0.1 * t,
        ambient_temperature=293.15,
    )
    assert bc.is_temperature_dependent
    assert bc.convection_coefficient_at(300.0) == pytest.approx(10.0 + 30.0)


def test_temperature_dependent_convection_coefficient_via_tabulated_property() -> None:
    table = TemperatureDependentProperty(temperatures=(280.0, 380.0), values=(10.0, 50.0))
    bc = ConvectionBoundaryCondition(
        surface=_SURFACE, convection_coefficient=table, ambient_temperature=293.15
    )
    assert bc.is_temperature_dependent
    assert bc.convection_coefficient_at(330.0) == pytest.approx(30.0)


def test_constant_ambient_temperature_ignores_time() -> None:
    bc = ConvectionBoundaryCondition(
        surface=_SURFACE, convection_coefficient=25.0, ambient_temperature=300.0
    )
    assert bc.ambient_temperature_at(0.0) == pytest.approx(300.0)
    assert bc.ambient_temperature_at(1e6) == pytest.approx(300.0)


def test_time_dependent_ambient_via_constant_load() -> None:
    bc = ConvectionBoundaryCondition(
        surface=_SURFACE, convection_coefficient=25.0, ambient_temperature=ConstantLoad(300.0)
    )
    assert bc.ambient_temperature_at(50.0) == pytest.approx(300.0)


def test_time_dependent_ambient_via_step_load_furnace_heating() -> None:
    bc = ConvectionBoundaryCondition(
        surface=_SURFACE,
        convection_coefficient=25.0,
        ambient_temperature=StepLoad(magnitude=800.0, step_time=100.0),
    )
    assert bc.ambient_temperature_at(50.0) == pytest.approx(0.0)
    assert bc.ambient_temperature_at(150.0) == pytest.approx(800.0)


def test_time_dependent_ambient_via_sinusoidal_cycle() -> None:
    bc = ConvectionBoundaryCondition(
        surface=_SURFACE,
        convection_coefficient=25.0,
        ambient_temperature=SinusoidalLoad(amplitude=10.0, angular_frequency=1.0),
    )
    assert bc.ambient_temperature_at(0.0) == pytest.approx(0.0)


def test_convection_rejects_non_positive_constant_coefficient() -> None:
    with pytest.raises(ValidationError):
        ConvectionBoundaryCondition(
            surface=_SURFACE, convection_coefficient=0.0, ambient_temperature=300.0
        )


def test_convection_rejects_non_finite_ambient_temperature() -> None:
    with pytest.raises(ValidationError):
        ConvectionBoundaryCondition(
            surface=_SURFACE, convection_coefficient=25.0, ambient_temperature=float("nan")
        )


def test_convection_coefficient_at_rejects_invalid_evaluated_value() -> None:
    bc = ConvectionBoundaryCondition(
        surface=_SURFACE, convection_coefficient=lambda t: -1.0, ambient_temperature=300.0
    )
    with pytest.raises(ValidationError):
        bc.convection_coefficient_at(300.0)
