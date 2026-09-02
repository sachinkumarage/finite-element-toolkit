"""Tests for femtoolkit.continuum.tensor: Voigt/tensor conversions and invariants."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from femtoolkit.continuum.stress import von_mises_3d
from femtoolkit.continuum.tensor import (
    deviatoric_stress,
    hydrostatic_stress,
    j2_invariant,
    mean_stress,
    principal_stresses_3d,
    tensor_to_voigt_strain,
    tensor_to_voigt_stress,
    trace,
    voigt_strain_to_tensor,
    voigt_stress_to_tensor,
    von_mises_stress_from_tensor,
)


def test_voigt_stress_round_trip() -> None:
    stress_voigt = np.array([100.0, 200.0, 300.0, 40.0, 50.0, 60.0])
    tensor = voigt_stress_to_tensor(stress_voigt)

    assert tensor.shape == (3, 3)
    assert_allclose(tensor, tensor.T)  # symmetric
    assert_allclose(tensor_to_voigt_stress(tensor), stress_voigt)


def test_voigt_strain_round_trip() -> None:
    strain_voigt = np.array([1e-3, 2e-3, 3e-3, 4e-3, 5e-3, 6e-3])
    tensor = voigt_strain_to_tensor(strain_voigt)

    assert_allclose(tensor, tensor.T)
    assert_allclose(tensor_to_voigt_strain(tensor), strain_voigt)


def test_strain_shear_is_halved_in_tensor_form() -> None:
    """The critical engineering-vs-tensor-shear distinction (module docstring)."""
    gamma_xy, gamma_yz, gamma_xz = 0.02, 0.04, 0.06
    strain_voigt = np.array([0.0, 0.0, 0.0, gamma_xy, gamma_yz, gamma_xz])
    tensor = voigt_strain_to_tensor(strain_voigt)

    assert_allclose(tensor[0, 1], gamma_xy / 2.0)
    assert_allclose(tensor[1, 2], gamma_yz / 2.0)
    assert_allclose(tensor[0, 2], gamma_xz / 2.0)


def test_stress_conversion_applies_no_shear_factor() -> None:
    """Unlike strain, stress off-diagonals map directly (no factor of two)."""
    tau_xy = 50.0
    stress_voigt = np.array([0.0, 0.0, 0.0, tau_xy, 0.0, 0.0])
    tensor = voigt_stress_to_tensor(stress_voigt)

    assert_allclose(tensor[0, 1], tau_xy)


def test_trace() -> None:
    tensor = np.diag([1.0, 2.0, 3.0])
    assert trace(tensor) == pytest.approx(6.0)


def test_mean_stress() -> None:
    stress_tensor = voigt_stress_to_tensor([90.0, 60.0, 30.0, 0.0, 0.0, 0.0])
    assert mean_stress(stress_tensor) == pytest.approx(60.0)


def test_hydrostatic_and_deviatoric_decomposition() -> None:
    stress_voigt = np.array([100.0, 200.0, 300.0, 40.0, 50.0, 60.0])
    stress_tensor = voigt_stress_to_tensor(stress_voigt)

    hydro = hydrostatic_stress(stress_tensor)
    dev = deviatoric_stress(stress_tensor)

    # Decomposition reconstructs the original tensor exactly.
    assert_allclose(hydro + dev, stress_tensor)
    # Hydrostatic part is a pure, isotropic (scaled-identity) tensor.
    assert_allclose(hydro, mean_stress(stress_tensor) * np.eye(3))
    # Deviatoric part is traceless by construction.
    assert trace(dev) == pytest.approx(0.0, abs=1e-9)


def test_j2_invariant_of_pure_hydrostatic_stress_is_zero() -> None:
    stress_tensor = voigt_stress_to_tensor([50.0, 50.0, 50.0, 0.0, 0.0, 0.0])
    dev = deviatoric_stress(stress_tensor)
    assert j2_invariant(dev) == pytest.approx(0.0, abs=1e-9)


def test_von_mises_tensor_and_voigt_formulations_agree() -> None:
    stress_voigt = np.array([100e6, -50e6, 20e6, 30e6, -10e6, 5e6])
    stress_tensor = voigt_stress_to_tensor(stress_voigt)

    vm_tensor = von_mises_stress_from_tensor(stress_tensor)
    vm_voigt = von_mises_3d(*stress_voigt)

    assert vm_tensor == pytest.approx(vm_voigt, rel=1e-10)


def test_von_mises_of_pure_hydrostatic_stress_is_zero() -> None:
    """Direct check of the physical claim the J2 module docstring makes."""
    stress_tensor = voigt_stress_to_tensor([80e6, 80e6, 80e6, 0.0, 0.0, 0.0])
    assert von_mises_stress_from_tensor(stress_tensor) == pytest.approx(0.0, abs=1e-3)


def test_von_mises_of_pure_shear() -> None:
    tau = 100e6
    stress_tensor = voigt_stress_to_tensor([0.0, 0.0, 0.0, tau, 0.0, 0.0])
    assert von_mises_stress_from_tensor(stress_tensor) == pytest.approx(
        np.sqrt(3.0) * tau, rel=1e-10
    )


def test_von_mises_of_uniaxial_stress_equals_the_stress_magnitude() -> None:
    sigma = 250e6
    stress_tensor = voigt_stress_to_tensor([sigma, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert von_mises_stress_from_tensor(stress_tensor) == pytest.approx(sigma, rel=1e-10)


def test_principal_stresses_sorted_descending() -> None:
    stress_tensor = np.diag([10.0, 30.0, 20.0])
    sigma_1, sigma_2, sigma_3 = principal_stresses_3d(stress_tensor)

    assert sigma_1 >= sigma_2 >= sigma_3
    assert_allclose([sigma_1, sigma_2, sigma_3], [30.0, 20.0, 10.0])


def test_principal_stresses_of_pure_shear() -> None:
    """A pure XY shear tau has principal stresses (+tau, 0, -tau)."""
    tau = 40e6
    stress_tensor = voigt_stress_to_tensor([0.0, 0.0, 0.0, tau, 0.0, 0.0])
    sigma_1, sigma_2, sigma_3 = principal_stresses_3d(stress_tensor)

    assert_allclose([sigma_1, sigma_2, sigma_3], [tau, 0.0, -tau], atol=1e-6)


def test_principal_stresses_invariant_under_hydrostatic_shift() -> None:
    """Adding a hydrostatic stress shifts every principal stress by the same amount."""
    base = voigt_stress_to_tensor([100e6, -50e6, 20e6, 30e6, -10e6, 5e6])
    shift = 40e6
    shifted = base + shift * np.eye(3)

    base_principal = np.array(principal_stresses_3d(base))
    shifted_principal = np.array(principal_stresses_3d(shifted))

    assert_allclose(shifted_principal, base_principal + shift, rtol=1e-8)
