"""Tests for femtoolkit.materials.thermoelastic."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from femtoolkit.continuum.constitutive import isotropic_3d_matrix
from femtoolkit.exceptions import ValidationError
from femtoolkit.materials.nonlinear import NonlinearMaterial
from femtoolkit.materials.thermal_properties import TemperatureDependentProperty
from femtoolkit.materials.thermoelastic import ThermoelasticMaterial3D


@pytest.fixture
def steel() -> ThermoelasticMaterial3D:
    return ThermoelasticMaterial3D(
        youngs_modulus=200e9,
        poisson_ratio=0.3,
        thermal_expansion_coefficient=12e-6,
        reference_temperature=293.15,
        density=7850.0,
    )


# --- Thermal strain ----------------------------------------------------------


def test_thermal_strain_is_zero_at_reference_temperature(steel: ThermoelasticMaterial3D) -> None:
    assert_allclose(steel.thermal_strain_voigt(293.15), np.zeros(6))


def test_thermal_strain_positive_for_heating(steel: ThermoelasticMaterial3D) -> None:
    thermal_strain = steel.thermal_strain_voigt(393.15)
    assert thermal_strain[0] == pytest.approx(12e-6 * 100.0)
    assert_allclose(thermal_strain[:3], thermal_strain[0])
    assert_allclose(thermal_strain[3:], np.zeros(3))


def test_thermal_strain_negative_for_cooling(steel: ThermoelasticMaterial3D) -> None:
    thermal_strain = steel.thermal_strain_voigt(193.15)
    assert thermal_strain[0] == pytest.approx(12e-6 * -100.0)
    assert thermal_strain[0] < 0.0


def test_thermal_strain_scales_linearly_with_delta_temperature(
    steel: ThermoelasticMaterial3D,
) -> None:
    small = steel.thermal_strain_voigt(303.15)[0]
    large = steel.thermal_strain_voigt(393.15)[0]
    assert large == pytest.approx(small * (100.0 / 10.0), rel=1e-10)


def test_no_thermal_shear_strain(steel: ThermoelasticMaterial3D) -> None:
    for temperature in (193.15, 293.15, 393.15, 573.15):
        assert_allclose(steel.thermal_strain_voigt(temperature)[3:], np.zeros(3))


# --- Free vs. constrained (material-level) -----------------------------------


def test_free_expansion_gives_zero_mechanical_stress(steel: ThermoelasticMaterial3D) -> None:
    """total_strain == thermal_strain (free expansion) -> mechanical strain 0 -> stress 0."""
    temperature = 393.15
    total_strain = steel.thermal_strain_voigt(temperature)
    stress = steel.stress_at(total_strain, temperature)
    assert_allclose(stress, np.zeros(6), atol=1e-6)


def test_fully_constrained_heating_develops_compressive_thermal_stress(
    steel: ThermoelasticMaterial3D,
) -> None:
    """total_strain == 0 (fully restrained), heating -> compressive thermal stress."""
    temperature = 393.15
    stress = steel.stress_at(np.zeros(6), temperature)
    expected_normal_stress = -200e9 * 12e-6 * 100.0 / (1.0 - 2.0 * 0.3)
    assert_allclose(stress[:3], expected_normal_stress, rtol=1e-10)
    assert_allclose(stress[3:], np.zeros(3), atol=1e-6)
    assert stress[0] < 0.0


def test_fully_constrained_cooling_develops_tensile_thermal_stress(
    steel: ThermoelasticMaterial3D,
) -> None:
    temperature = 193.15
    stress = steel.stress_at(np.zeros(6), temperature)
    assert stress[0] > 0.0


def test_combined_mechanical_and_thermal_strain(steel: ThermoelasticMaterial3D) -> None:
    """A mechanical strain superposed on a free thermal one produces exactly the
    stress that mechanical strain alone would (linearity/superposition)."""
    temperature = 393.15
    mechanical_only_strain = np.array([1e-4, 0, 0, 0, 0, 0])
    combined_total_strain = mechanical_only_strain + steel.thermal_strain_voigt(temperature)

    stress_combined = steel.stress_at(combined_total_strain, temperature)
    stress_mechanical_only = steel.stress_at(mechanical_only_strain, steel.reference_temperature)
    assert_allclose(stress_combined, stress_mechanical_only, rtol=1e-10)


# --- ThermoelasticMaterialAtTemperature / NonlinearMaterial integration -----


def test_at_temperature_returns_nonlinear_material(steel: ThermoelasticMaterial3D) -> None:
    bound = steel.at_temperature(393.15)
    assert isinstance(bound, NonlinearMaterial)
    assert bound.temperature == pytest.approx(393.15)


def test_initial_state_records_temperature(steel: ThermoelasticMaterial3D) -> None:
    bound = steel.at_temperature(393.15)
    state = bound.initial_state()
    assert_allclose(state.strain, np.zeros(6))
    assert_allclose(state.stress, np.zeros(6))
    assert state.temperature == pytest.approx(393.15)


def test_trial_state_matches_stress_at(steel: ThermoelasticMaterial3D) -> None:
    bound = steel.at_temperature(393.15)
    strain = np.array([1e-4, -3e-5, -3e-5, 2e-5, 0, 0])
    state = bound.trial_state(strain, bound.initial_state())
    assert_allclose(state.stress, steel.stress_at(strain, 393.15))
    assert state.temperature == pytest.approx(393.15)


def test_trial_state_ignores_committed_state(steel: ThermoelasticMaterial3D) -> None:
    bound = steel.at_temperature(393.15)
    strain = np.array([1e-4, 0, 0, 0, 0, 0])
    committed_a = bound.initial_state()
    committed_b = bound.trial_state(np.array([5e-4, 0, 0, 0, 0, 0]), committed_a)
    assert_allclose(
        bound.trial_state(strain, committed_a).stress, bound.trial_state(strain, committed_b).stress
    )


def test_tangent_modulus_equals_constitutive_matrix_at_temperature(
    steel: ThermoelasticMaterial3D,
) -> None:
    bound = steel.at_temperature(393.15)
    state = bound.trial_state(np.array([1e-4, 0, 0, 0, 0, 0]), bound.initial_state())
    assert_allclose(bound.tangent_modulus(state), isotropic_3d_matrix(200e9, 0.3))


def test_different_temperatures_give_different_materials(steel: ThermoelasticMaterial3D) -> None:
    cold = steel.at_temperature(293.15)
    hot = steel.at_temperature(393.15)
    strain = np.zeros(6)
    assert not np.allclose(
        cold.trial_state(strain, cold.initial_state()).stress,
        hot.trial_state(strain, hot.initial_state()).stress,
    )


# --- Temperature-dependent properties ----------------------------------------


def test_temperature_dependent_youngs_modulus_changes_constitutive_matrix() -> None:
    material = ThermoelasticMaterial3D(
        youngs_modulus=TemperatureDependentProperty(
            temperatures=(293.15, 393.15), values=(200e9, 190e9)
        ),
        poisson_ratio=0.3,
        thermal_expansion_coefficient=12e-6,
        reference_temperature=293.15,
        density=7850.0,
    )
    assert material.youngs_modulus_at(293.15) == pytest.approx(200e9)
    assert material.youngs_modulus_at(393.15) == pytest.approx(190e9)
    assert not np.allclose(
        material.constitutive_matrix_at(293.15), material.constitutive_matrix_at(393.15)
    )


def test_temperature_dependent_alpha_changes_thermal_strain() -> None:
    material = ThermoelasticMaterial3D(
        youngs_modulus=200e9,
        poisson_ratio=0.3,
        thermal_expansion_coefficient=TemperatureDependentProperty(
            temperatures=(293.15, 473.15), values=(11e-6, 14e-6)
        ),
        reference_temperature=293.15,
        density=7850.0,
    )
    strain_at_low_alpha_end = material.thermal_strain_voigt(293.15 + 1.0)
    strain_at_high_alpha_end = material.thermal_strain_voigt(473.15 - 1.0)
    assert strain_at_high_alpha_end[0] / (473.15 - 1.0 - 293.15) > strain_at_low_alpha_end[0] / 1.0


def test_out_of_range_tabulated_youngs_modulus_raises() -> None:
    material = ThermoelasticMaterial3D(
        youngs_modulus=TemperatureDependentProperty(
            temperatures=(293.15, 393.15), values=(200e9, 190e9)
        ),
        poisson_ratio=0.3,
        thermal_expansion_coefficient=12e-6,
        reference_temperature=293.15,
        density=7850.0,
    )
    with pytest.raises(ValidationError):
        material.youngs_modulus_at(1000.0)


# --- Validation ---------------------------------------------------------------


@pytest.mark.parametrize("youngs_modulus", [0.0, -1.0, float("nan")])
def test_rejects_non_positive_constant_youngs_modulus(youngs_modulus: float) -> None:
    with pytest.raises(ValidationError):
        ThermoelasticMaterial3D(
            youngs_modulus=youngs_modulus,
            poisson_ratio=0.3,
            thermal_expansion_coefficient=12e-6,
            reference_temperature=293.15,
            density=7850.0,
        )


@pytest.mark.parametrize("poisson_ratio", [-1.0, 0.5, 0.6, float("nan")])
def test_rejects_invalid_constant_poisson_ratio(poisson_ratio: float) -> None:
    with pytest.raises(ValidationError):
        ThermoelasticMaterial3D(
            youngs_modulus=200e9,
            poisson_ratio=poisson_ratio,
            thermal_expansion_coefficient=12e-6,
            reference_temperature=293.15,
            density=7850.0,
        )


def test_rejects_non_finite_constant_thermal_expansion_coefficient() -> None:
    with pytest.raises(ValidationError):
        ThermoelasticMaterial3D(
            youngs_modulus=200e9,
            poisson_ratio=0.3,
            thermal_expansion_coefficient=float("nan"),
            reference_temperature=293.15,
            density=7850.0,
        )


def test_rejects_non_finite_reference_temperature() -> None:
    with pytest.raises(ValidationError):
        ThermoelasticMaterial3D(
            youngs_modulus=200e9,
            poisson_ratio=0.3,
            thermal_expansion_coefficient=12e-6,
            reference_temperature=float("nan"),
            density=7850.0,
        )


@pytest.mark.parametrize("density", [0.0, -1.0, float("nan")])
def test_rejects_non_positive_density(density: float) -> None:
    with pytest.raises(ValidationError):
        ThermoelasticMaterial3D(
            youngs_modulus=200e9,
            poisson_ratio=0.3,
            thermal_expansion_coefficient=12e-6,
            reference_temperature=293.15,
            density=density,
        )


def test_negative_thermal_expansion_coefficient_is_valid() -> None:
    """Materials that contract on heating (e.g. some polymers/composites) are physically valid."""
    material = ThermoelasticMaterial3D(
        youngs_modulus=200e9,
        poisson_ratio=0.3,
        thermal_expansion_coefficient=-5e-6,
        reference_temperature=293.15,
        density=7850.0,
    )
    thermal_strain = material.thermal_strain_voigt(393.15)
    assert thermal_strain[0] < 0.0
