"""Tests for femtoolkit.thermal.thermal_material.ThermalMaterial."""

import pytest

from femtoolkit.exceptions import ValidationError
from femtoolkit.materials.thermal_properties import TemperatureDependentProperty
from femtoolkit.thermal.thermal_material import ThermalMaterial


@pytest.fixture
def steel() -> ThermalMaterial:
    return ThermalMaterial(thermal_conductivity=50.0, density=7850.0, specific_heat=460.0)


def test_constant_conductivity(steel: ThermalMaterial) -> None:
    assert steel.conductivity_at(293.15) == pytest.approx(50.0)
    assert steel.conductivity_at(500.0) == pytest.approx(50.0)


def test_constant_specific_heat(steel: ThermalMaterial) -> None:
    assert steel.specific_heat_at(293.15) == pytest.approx(460.0)


def test_volumetric_heat_capacity(steel: ThermalMaterial) -> None:
    assert steel.volumetric_heat_capacity_at(293.15) == pytest.approx(7850.0 * 460.0)


def test_tabulated_conductivity_interpolates() -> None:
    material = ThermalMaterial(
        thermal_conductivity=TemperatureDependentProperty(
            temperatures=(293.15, 493.15), values=(50.0, 40.0)
        ),
        density=7850.0,
        specific_heat=460.0,
    )
    assert material.conductivity_at(293.15) == pytest.approx(50.0)
    assert material.conductivity_at(493.15) == pytest.approx(40.0)
    assert material.conductivity_at(393.15) == pytest.approx(45.0)


def test_tabulated_specific_heat_interpolates() -> None:
    material = ThermalMaterial(
        thermal_conductivity=50.0,
        density=7850.0,
        specific_heat=TemperatureDependentProperty(
            temperatures=(293.15, 493.15), values=(460.0, 520.0)
        ),
    )
    assert material.specific_heat_at(393.15) == pytest.approx(490.0)


def test_user_defined_function_conductivity() -> None:
    material = ThermalMaterial(
        thermal_conductivity=lambda t: 50.0 - 0.02 * (t - 293.15),
        density=7850.0,
        specific_heat=460.0,
    )
    assert material.conductivity_at(393.15) == pytest.approx(48.0)


def test_out_of_range_tabulated_conductivity_raises() -> None:
    material = ThermalMaterial(
        thermal_conductivity=TemperatureDependentProperty(
            temperatures=(293.15, 493.15), values=(50.0, 40.0)
        ),
        density=7850.0,
        specific_heat=460.0,
    )
    with pytest.raises(ValidationError):
        material.conductivity_at(1000.0)


@pytest.mark.parametrize("conductivity", [0.0, -1.0, float("nan")])
def test_rejects_non_positive_constant_conductivity(conductivity: float) -> None:
    with pytest.raises(ValidationError):
        ThermalMaterial(thermal_conductivity=conductivity, density=7850.0, specific_heat=460.0)


@pytest.mark.parametrize("specific_heat", [0.0, -1.0, float("nan")])
def test_rejects_non_positive_constant_specific_heat(specific_heat: float) -> None:
    with pytest.raises(ValidationError):
        ThermalMaterial(thermal_conductivity=50.0, density=7850.0, specific_heat=specific_heat)


@pytest.mark.parametrize("density", [0.0, -1.0, float("nan")])
def test_rejects_non_positive_density(density: float) -> None:
    with pytest.raises(ValidationError):
        ThermalMaterial(thermal_conductivity=50.0, density=density, specific_heat=460.0)


def test_evaluated_tabulated_conductivity_that_goes_non_positive_raises() -> None:
    """A tabulated property that resolves to a non-positive conductivity must still be caught."""
    material = ThermalMaterial(
        thermal_conductivity=TemperatureDependentProperty(
            temperatures=(293.15, 493.15), values=(50.0, -10.0)
        ),
        density=7850.0,
        specific_heat=460.0,
    )
    with pytest.raises(ValidationError):
        material.conductivity_at(493.15)
