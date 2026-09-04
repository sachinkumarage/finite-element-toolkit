"""Tests for femtoolkit.materials.mooney_rivlin: compressible 2-parameter Mooney-Rivlin material."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from femtoolkit.continuum.tensor import tensor_to_voigt_strain
from femtoolkit.exceptions import ValidationError
from femtoolkit.materials.hyperelastic import HyperelasticMaterial
from femtoolkit.materials.mooney_rivlin import MooneyRivlin3D
from femtoolkit.materials.nonlinear import NonlinearMaterial


def _rotation_matrix(axis: np.ndarray, theta: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    k = np.array(
        [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
    )
    return np.eye(3) + np.sin(theta) * k + (1 - np.cos(theta)) * (k @ k)


@pytest.fixture
def rubber() -> MooneyRivlin3D:
    return MooneyRivlin3D(c10=0.4e6, c01=0.1e6, bulk_modulus=200e6)


def test_is_a_hyperelastic_and_nonlinear_material(rubber: MooneyRivlin3D) -> None:
    assert isinstance(rubber, HyperelasticMaterial)
    assert isinstance(rubber, NonlinearMaterial)


def test_reference_state_zero_energy_and_stress(rubber: MooneyRivlin3D) -> None:
    """Regression test for the naive raw-invariant reference-state bug caught during planning."""
    assert rubber.strain_energy_density(np.eye(3)) == pytest.approx(0.0, abs=1e-8)
    assert_allclose(rubber.second_piola_kirchhoff_stress(np.eye(3)), np.zeros(6), atol=1e-3)


def test_reference_state_holds_for_nonzero_c01() -> None:
    """The specific bug: a naive raw-invariant form is only stress-free at F=I by accident.

    Using c01 much larger than c10 would expose a naive (non-decoupled)
    implementation immediately, since dI2/dC|C=I = 2I != 0 contributes a
    spurious residual stress proportional to c01.
    """
    material = MooneyRivlin3D(c10=0.1e6, c01=0.9e6, bulk_modulus=200e6)
    assert material.strain_energy_density(np.eye(3)) == pytest.approx(0.0, abs=1e-8)
    assert_allclose(material.second_piola_kirchhoff_stress(np.eye(3)), np.zeros(6), atol=1e-3)


@pytest.mark.parametrize("theta", [0.3, 1.1, 2.4, -0.9])
def test_pure_rotation_gives_zero_energy_and_stress(rubber: MooneyRivlin3D, theta: float) -> None:
    """Mandatory objectivity test: F = R (a pure rotation) must give W=0, S=0."""
    r = _rotation_matrix(np.array([0.3, -0.7, 0.4]), theta)
    assert_allclose(r.T @ r, np.eye(3), atol=1e-10)

    assert rubber.strain_energy_density(r) == pytest.approx(0.0, abs=1e-3)
    assert_allclose(rubber.second_piola_kirchhoff_stress(r), np.zeros(6), atol=10.0)


def test_simple_shear_gives_nonzero_shear_stress(rubber: MooneyRivlin3D) -> None:
    gamma = 0.2
    f = np.array([[1.0, gamma, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    stress = rubber.second_piola_kirchhoff_stress(f)
    assert abs(stress[3]) > 1.0


def test_volumetric_deformation_jacobian(rubber: MooneyRivlin3D) -> None:
    alpha = 1.15
    f = alpha * np.eye(3)
    jacobian = np.linalg.det(f)
    assert jacobian == pytest.approx(alpha**3)
    stress = rubber.second_piola_kirchhoff_stress(f)
    assert not np.allclose(stress[:3], 0.0, atol=1.0)


def test_c01_zero_reduces_toward_neo_hookean_like_shear_response() -> None:
    """With c01=0, the deviatoric response is governed only by I1_bar, matching
    the qualitative Neo-Hookean-like shear stiffness relation shear_modulus = 2*c10."""
    material = MooneyRivlin3D(c10=0.5e6, c01=0.0, bulk_modulus=200e6)
    assert material.initial_shear_modulus == pytest.approx(2.0 * 0.5e6)


def test_material_tangent_is_positive_definite_at_realistic_state(rubber: MooneyRivlin3D) -> None:
    stretch = 1.4
    lateral = stretch ** (-0.5)
    f = np.diag([stretch, lateral, lateral])
    tangent = rubber.material_tangent(f)
    eigenvalues = np.linalg.eigvalsh(tangent)
    assert np.all(eigenvalues > 0)


def test_tangent_matches_finite_difference_of_stress(rubber: MooneyRivlin3D) -> None:
    f = np.array([[1.1, 0.03, 0.0], [0.0, 0.95, 0.0], [0.0, 0.0, 1.02]])
    strain_voigt = tensor_to_voigt_strain(0.5 * (f.T @ f - np.eye(3)))
    tangent = rubber._tangent_from_strain_voigt(strain_voigt)

    delta = np.zeros(6)
    delta[0] = 1e-5
    stress_plus = rubber._stress_from_strain_voigt(strain_voigt + delta)
    stress_minus = rubber._stress_from_strain_voigt(strain_voigt - delta)
    finite_difference_column = (stress_plus - stress_minus) / (2 * 1e-5)

    assert_allclose(tangent[:, 0], finite_difference_column, rtol=1e-2, atol=100.0)


def test_energy_consistency_against_stress_work(rubber: MooneyRivlin3D) -> None:
    """W(F+dF) - W(F) must match the stress work S:dE to first order (spec section 16)."""
    f = np.array([[1.1, 0.02, 0.0], [0.0, 0.93, 0.0], [0.0, 0.0, 1.01]])
    c = f.T @ f
    e = 0.5 * (c - np.eye(3))

    delta_e = np.zeros((3, 3))
    delta_e[0, 0] = 1e-6

    energy_before = rubber._energy_from_right_cauchy_green(c)
    c_after = c + 2.0 * delta_e
    energy_after = rubber._energy_from_right_cauchy_green(c_after)

    strain_voigt = tensor_to_voigt_strain(e)
    stress_voigt = rubber._stress_from_strain_voigt(strain_voigt)
    from femtoolkit.continuum.tensor import voigt_stress_to_tensor

    stress_tensor = voigt_stress_to_tensor(stress_voigt)
    stress_work = float(np.tensordot(stress_tensor, delta_e))

    assert (energy_after - energy_before) == pytest.approx(stress_work, rel=1e-2, abs=1e-3)


@pytest.mark.parametrize("c10", [0.0, -1.0, float("nan")])
def test_rejects_non_positive_c10(c10: float) -> None:
    with pytest.raises(ValidationError):
        MooneyRivlin3D(c10=c10, c01=0.1e6, bulk_modulus=200e6)


@pytest.mark.parametrize("c01", [-1.0, float("nan")])
def test_rejects_negative_c01(c01: float) -> None:
    with pytest.raises(ValidationError):
        MooneyRivlin3D(c10=0.4e6, c01=c01, bulk_modulus=200e6)


@pytest.mark.parametrize("bulk_modulus", [0.0, -1.0, float("nan")])
def test_rejects_non_positive_bulk_modulus(bulk_modulus: float) -> None:
    with pytest.raises(ValidationError):
        MooneyRivlin3D(c10=0.4e6, c01=0.1e6, bulk_modulus=bulk_modulus)


def test_initial_state_and_trial_state(rubber: MooneyRivlin3D) -> None:
    state0 = rubber.initial_state()
    assert_allclose(state0.strain, np.zeros(6))
    assert_allclose(state0.stress, np.zeros(6))

    strain = np.array([0.05, -0.01, -0.01, 0.02, 0.0, 0.0])
    state1 = rubber.trial_state(strain, state0)
    assert_allclose(state1.strain, strain)
    assert not np.allclose(state1.stress, 0.0)


def test_trial_state_is_path_independent(rubber: MooneyRivlin3D) -> None:
    strain = np.array([0.03, -0.005, -0.005, 0.0, 0.0, 0.0])
    committed_a = rubber.initial_state()
    committed_b = rubber.trial_state(np.array([0.1, 0, 0, 0, 0, 0]), committed_a)

    state_from_a = rubber.trial_state(strain, committed_a)
    state_from_b = rubber.trial_state(strain, committed_b)
    assert_allclose(state_from_a.stress, state_from_b.stress)
