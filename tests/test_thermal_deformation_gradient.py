"""Tests for femtoolkit.continuum.thermal: the finite-strain thermal deformation foundation."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from femtoolkit.continuum.thermal import (
    elastic_deformation_gradient_from_thermal_split,
    thermal_deformation_gradient,
)
from femtoolkit.exceptions import InvalidDeformationGradientError
from femtoolkit.materials import MooneyRivlin3D, NeoHookean3D


def test_no_temperature_change_gives_identity() -> None:
    fth = thermal_deformation_gradient(thermal_expansion_coefficient=12e-6, delta_temperature=0.0)
    assert_allclose(fth, np.eye(3))


def test_heating_gives_expansion() -> None:
    fth = thermal_deformation_gradient(thermal_expansion_coefficient=12e-6, delta_temperature=100.0)
    assert fth[0, 0] > 1.0
    assert_allclose(fth, fth[0, 0] * np.eye(3))


def test_cooling_gives_contraction() -> None:
    fth = thermal_deformation_gradient(
        thermal_expansion_coefficient=12e-6, delta_temperature=-100.0
    )
    assert fth[0, 0] < 1.0


def test_thermal_stretch_matches_linear_expansion_formula() -> None:
    alpha, delta_temperature = 12e-6, 150.0
    fth = thermal_deformation_gradient(alpha, delta_temperature)
    expected_stretch = 1.0 + alpha * delta_temperature
    assert fth[0, 0] == pytest.approx(expected_stretch)


def test_is_isotropic_pure_volumetric_no_shape_change() -> None:
    fth = thermal_deformation_gradient(thermal_expansion_coefficient=20e-6, delta_temperature=50.0)
    off_diagonal = fth - np.diag(np.diag(fth))
    assert_allclose(off_diagonal, np.zeros((3, 3)))
    assert fth[0, 0] == fth[1, 1] == fth[2, 2]


def test_extreme_cooling_raises() -> None:
    with pytest.raises(InvalidDeformationGradientError):
        thermal_deformation_gradient(thermal_expansion_coefficient=1e-2, delta_temperature=-1000.0)


def test_negative_alpha_gives_contraction_on_heating() -> None:
    fth = thermal_deformation_gradient(thermal_expansion_coefficient=-5e-6, delta_temperature=100.0)
    assert fth[0, 0] < 1.0


# --- Elastic-part recovery and Version 17 compatibility ----------------------


def test_recovers_elastic_part_with_no_plasticity() -> None:
    fe_true = np.array([[1.05, 0.02, 0.0], [0.0, 0.97, 0.0], [0.0, 0.0, 1.02]])
    fth = thermal_deformation_gradient(12e-6, 100.0)
    f_total = fe_true @ fth

    fe_recovered = elastic_deformation_gradient_from_thermal_split(f_total, fth)
    assert_allclose(fe_recovered, fe_true, atol=1e-10)


def test_recovers_elastic_part_with_plasticity() -> None:
    fe_true = np.diag([1.03, 0.98, 0.99])
    fth = thermal_deformation_gradient(12e-6, 50.0)
    fp_true = np.diag([1.01, 0.995, 0.995])
    f_total = fe_true @ fth @ fp_true

    fe_recovered = elastic_deformation_gradient_from_thermal_split(f_total, fth, fp_true)
    assert_allclose(fe_recovered, fe_true, atol=1e-10)


def test_identity_thermal_split_reduces_to_total_deformation_gradient() -> None:
    f_total = np.array([[1.1, 0.0, 0.0], [0.0, 0.95, 0.0], [0.0, 0.0, 1.0]])
    fe_recovered = elastic_deformation_gradient_from_thermal_split(f_total, np.eye(3))
    assert_allclose(fe_recovered, f_total)


def test_compatible_with_neo_hookean_3d() -> None:
    """The recovered elastic F is usable by an unmodified Version 17 hyperelastic material."""
    rubber = NeoHookean3D(youngs_modulus=5e6, poisson_ratio=0.45)
    fe_true = np.array([[1.1, 0.03, 0.0], [0.0, 0.93, 0.0], [0.0, 0.0, 1.02]])
    fth = thermal_deformation_gradient(50e-6, 80.0)
    f_total = fe_true @ fth

    fe_recovered = elastic_deformation_gradient_from_thermal_split(f_total, fth)
    assert rubber.strain_energy_density(fe_recovered) == pytest.approx(
        rubber.strain_energy_density(fe_true), rel=1e-8
    )
    assert_allclose(
        rubber.second_piola_kirchhoff_stress(fe_recovered),
        rubber.second_piola_kirchhoff_stress(fe_true),
        rtol=1e-8,
    )


def test_compatible_with_mooney_rivlin_3d() -> None:
    rubber = MooneyRivlin3D(c10=0.4e6, c01=0.1e6, bulk_modulus=200e6)
    fe_true = np.diag([1.05, 0.98, 0.99])
    fth = thermal_deformation_gradient(30e-6, 60.0)
    f_total = fe_true @ fth

    fe_recovered = elastic_deformation_gradient_from_thermal_split(f_total, fth)
    assert rubber.strain_energy_density(fe_recovered) == pytest.approx(
        rubber.strain_energy_density(fe_true), rel=1e-8
    )


def test_reference_temperature_gives_stress_free_thermal_split() -> None:
    """At dT=0 (Fth=I), the elastic part is exactly the total F, and a hyperelastic
    material at F=I (no other deformation) reports zero energy/stress, unaffected
    by the (identity) thermal split."""
    rubber = NeoHookean3D(youngs_modulus=5e6, poisson_ratio=0.45)
    fth = thermal_deformation_gradient(12e-6, 0.0)
    fe_recovered = elastic_deformation_gradient_from_thermal_split(np.eye(3), fth)
    assert_allclose(fe_recovered, np.eye(3))
    assert rubber.strain_energy_density(fe_recovered) == pytest.approx(0.0, abs=1e-10)
