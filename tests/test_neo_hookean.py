"""Tests for femtoolkit.materials.neo_hookean: compressible Neo-Hookean hyperelasticity."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from femtoolkit.continuum.tensor import tensor_to_voigt_strain
from femtoolkit.exceptions import ValidationError
from femtoolkit.materials.hyperelastic import HyperelasticMaterial
from femtoolkit.materials.neo_hookean import NeoHookean3D
from femtoolkit.materials.nonlinear import NonlinearMaterial


def _rotation_matrix(axis: np.ndarray, theta: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    k = np.array(
        [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
    )
    return np.eye(3) + np.sin(theta) * k + (1 - np.cos(theta)) * (k @ k)


@pytest.fixture
def rubber() -> NeoHookean3D:
    return NeoHookean3D(youngs_modulus=5.0e6, poisson_ratio=0.45)


def test_is_a_hyperelastic_and_nonlinear_material(rubber: NeoHookean3D) -> None:
    assert isinstance(rubber, HyperelasticMaterial)
    assert isinstance(rubber, NonlinearMaterial)


def test_reference_state_zero_energy_and_stress(rubber: NeoHookean3D) -> None:
    assert rubber.strain_energy_density(np.eye(3)) == pytest.approx(0.0, abs=1e-8)
    assert_allclose(rubber.second_piola_kirchhoff_stress(np.eye(3)), np.zeros(6), atol=1e-6)


@pytest.mark.parametrize("theta", [0.3, 1.1, 2.4, -0.9])
def test_pure_rotation_gives_zero_energy_and_stress(rubber: NeoHookean3D, theta: float) -> None:
    """Mandatory objectivity test: F = R (a pure rotation) must give W=0, S=0."""
    r = _rotation_matrix(np.array([0.3, -0.7, 0.4]), theta)
    assert_allclose(r.T @ r, np.eye(3), atol=1e-10)

    assert rubber.strain_energy_density(r) == pytest.approx(0.0, abs=1e-6)
    assert_allclose(rubber.second_piola_kirchhoff_stress(r), np.zeros(6), atol=1e-4)
    assert_allclose(rubber.cauchy_stress(r), np.zeros((3, 3)), atol=1e-4)


def test_simple_shear_gives_nonzero_shear_stress(rubber: NeoHookean3D) -> None:
    gamma = 0.2
    f = np.array([[1.0, gamma, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    stress = rubber.second_piola_kirchhoff_stress(f)
    assert abs(stress[3]) > 1.0  # S_xy index in [Sxx,Syy,Szz,Sxy,Syz,Sxz]


def test_volumetric_deformation_jacobian(rubber: NeoHookean3D) -> None:
    alpha = 1.2
    f = alpha * np.eye(3)
    jacobian = np.linalg.det(f)
    assert jacobian == pytest.approx(alpha**3)
    # Pure dilation is not stress-free (K > 0 resists volume change).
    stress = rubber.second_piola_kirchhoff_stress(f)
    assert not np.allclose(stress[:3], 0.0, atol=1.0)


def test_uniaxial_extension_matches_analytical_stretch_curve(rubber: NeoHookean3D) -> None:
    """Classic incompressible stress difference: sigma_11-sigma_22 = mu*(lambda^2-1/lambda).

    An isochoric lateral stretch ``lambda^(-1/2)`` makes ``J = lambda *
    lambda^(-1) = 1`` exactly, so the volumetric (``ln J``) term of this
    compressible formula vanishes exactly and ``S = mu*(I - C^-1))``. This
    is *not* the same as the truly incompressible (Lagrange-multiplier)
    formulation used to derive the classic curve -- here the lateral
    Cauchy stress is not independently zero -- but the *difference*
    ``sigma_11 - sigma_22`` cancels the extra volumetric-like terms
    identically and reproduces the classic closed-form result exactly.
    """
    mu = rubber.shear_modulus

    for stretch in (1.1, 1.3, 1.6):
        lateral = stretch ** (-0.5)
        f = np.diag([stretch, lateral, lateral])
        sigma = rubber.cauchy_stress(f)
        expected_difference = mu * (stretch**2 - 1.0 / stretch)
        assert sigma[0, 0] - sigma[1, 1] == pytest.approx(expected_difference, rel=1e-6)


def test_analytical_stress_matches_numerical_base_class_default(rubber: NeoHookean3D) -> None:
    """Cross-check NeoHookean3D's analytical override against the base class's numerical default."""
    f = np.array([[1.15, 0.05, 0.0], [0.0, 0.92, -0.03], [0.02, 0.0, 1.05]])
    strain_voigt = tensor_to_voigt_strain(0.5 * (f.T @ f - np.eye(3)))
    analytical = rubber._stress_from_strain_voigt(strain_voigt)
    numerical = HyperelasticMaterial._stress_from_strain_voigt(rubber, strain_voigt)
    assert_allclose(analytical, numerical, rtol=2e-3, atol=10.0)


def test_material_tangent_is_positive_definite_at_realistic_state(rubber: NeoHookean3D) -> None:
    stretch = 1.4
    lateral = stretch ** (-0.5)
    f = np.diag([stretch, lateral, lateral])
    tangent = rubber.material_tangent(f)
    eigenvalues = np.linalg.eigvalsh(tangent)
    assert np.all(eigenvalues > 0)


def test_tangent_matches_finite_difference_of_stress(rubber: NeoHookean3D) -> None:
    f = np.array([[1.1, 0.03, 0.0], [0.0, 0.95, 0.0], [0.0, 0.0, 1.02]])
    strain_voigt = tensor_to_voigt_strain(0.5 * (f.T @ f - np.eye(3)))
    tangent = rubber._tangent_from_strain_voigt(strain_voigt)

    delta = np.zeros(6)
    delta[0] = 1e-5
    stress_plus = rubber._stress_from_strain_voigt(strain_voigt + delta)
    stress_minus = rubber._stress_from_strain_voigt(strain_voigt - delta)
    finite_difference_column = (stress_plus - stress_minus) / (2 * 1e-5)

    assert_allclose(tangent[:, 0], finite_difference_column, rtol=1e-2, atol=100.0)


@pytest.mark.parametrize("youngs_modulus", [0.0, -1.0, float("nan")])
def test_rejects_non_positive_youngs_modulus(youngs_modulus: float) -> None:
    with pytest.raises(ValidationError):
        NeoHookean3D(youngs_modulus=youngs_modulus, poisson_ratio=0.3)


@pytest.mark.parametrize("poisson_ratio", [-1.0, 0.5, 0.6, float("nan")])
def test_rejects_invalid_poisson_ratio(poisson_ratio: float) -> None:
    with pytest.raises(ValidationError):
        NeoHookean3D(youngs_modulus=5.0e6, poisson_ratio=poisson_ratio)


def test_initial_state_and_trial_state(rubber: NeoHookean3D) -> None:
    state0 = rubber.initial_state()
    assert_allclose(state0.strain, np.zeros(6))
    assert_allclose(state0.stress, np.zeros(6))

    strain = np.array([0.05, -0.01, -0.01, 0.02, 0.0, 0.0])
    state1 = rubber.trial_state(strain, state0)
    assert_allclose(state1.strain, strain)
    assert not np.allclose(state1.stress, 0.0)


def test_trial_state_is_path_independent(rubber: NeoHookean3D) -> None:
    strain = np.array([0.03, -0.005, -0.005, 0.0, 0.0, 0.0])
    committed_a = rubber.initial_state()
    committed_b = rubber.trial_state(np.array([0.1, 0, 0, 0, 0, 0]), committed_a)

    state_from_a = rubber.trial_state(strain, committed_a)
    state_from_b = rubber.trial_state(strain, committed_b)
    assert_allclose(state_from_a.stress, state_from_b.stress)
