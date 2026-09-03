"""Tests for femtoolkit.materials.finite_strain: St. Venant-Kirchhoff materials."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from femtoolkit.continuum.constitutive import isotropic_3d_matrix
from femtoolkit.exceptions import ValidationError
from femtoolkit.materials.finite_strain import SaintVenantKirchhoff1D, SaintVenantKirchhoff3D
from femtoolkit.materials.nonlinear import NonlinearMaterial


@pytest.fixture
def steel_3d() -> SaintVenantKirchhoff3D:
    return SaintVenantKirchhoff3D(youngs_modulus=200e9, poisson_ratio=0.3)


def test_is_a_nonlinear_material(steel_3d: SaintVenantKirchhoff3D) -> None:
    assert isinstance(steel_3d, NonlinearMaterial)


def test_initial_state_is_zero(steel_3d: SaintVenantKirchhoff3D) -> None:
    state = steel_3d.initial_state()
    assert_allclose(state.strain, np.zeros(6))
    assert_allclose(state.stress, np.zeros(6))
    assert not state.yielded


def test_trial_state_matches_s_equals_c_e(steel_3d: SaintVenantKirchhoff3D) -> None:
    strain = np.array([0.01, -0.002, 0.003, 0.0015, -0.001, 0.0005])
    state = steel_3d.trial_state(strain, steel_3d.initial_state())

    expected = isotropic_3d_matrix(200e9, 0.3) @ strain
    assert_allclose(state.stress, expected)
    assert_allclose(state.strain, strain)


def test_trial_state_ignores_committed_state(steel_3d: SaintVenantKirchhoff3D) -> None:
    """St. Venant-Kirchhoff is path-independent (elastic): stress depends only on current E."""
    strain = np.array([0.005, 0.0, 0.0, 0.0, 0.0, 0.0])
    committed_a = steel_3d.initial_state()
    committed_b = steel_3d.trial_state(np.array([0.02, 0, 0, 0, 0, 0]), committed_a)

    state_from_a = steel_3d.trial_state(strain, committed_a)
    state_from_b = steel_3d.trial_state(strain, committed_b)
    assert_allclose(state_from_a.stress, state_from_b.stress)


def test_tangent_modulus_is_constant_and_equals_constitutive_matrix(
    steel_3d: SaintVenantKirchhoff3D,
) -> None:
    state_small = steel_3d.trial_state(np.array([1e-6, 0, 0, 0, 0, 0]), steel_3d.initial_state())
    state_large = steel_3d.trial_state(np.array([0.05, 0, 0, 0, 0, 0]), steel_3d.initial_state())

    tangent_small = steel_3d.tangent_modulus(state_small)
    tangent_large = steel_3d.tangent_modulus(state_large)
    expected = isotropic_3d_matrix(200e9, 0.3)

    assert_allclose(tangent_small, expected)
    assert_allclose(tangent_large, expected)
    assert_allclose(tangent_small, tangent_large)


@pytest.mark.parametrize("youngs_modulus", [0.0, -1.0])
def test_3d_rejects_non_positive_youngs_modulus(youngs_modulus: float) -> None:
    with pytest.raises(ValidationError):
        SaintVenantKirchhoff3D(youngs_modulus=youngs_modulus, poisson_ratio=0.3)


@pytest.mark.parametrize("poisson_ratio", [-1.0, 0.5, 0.6])
def test_3d_rejects_invalid_poisson_ratio(poisson_ratio: float) -> None:
    with pytest.raises(ValidationError):
        SaintVenantKirchhoff3D(youngs_modulus=200e9, poisson_ratio=poisson_ratio)


# --- 1D (truss) reduction --------------------------------------------------


def test_1d_matches_e_times_strain() -> None:
    material = SaintVenantKirchhoff1D(youngs_modulus=200e9)
    state = material.trial_state(0.01, material.initial_state())
    assert state.stress == pytest.approx(200e9 * 0.01)


def test_1d_tangent_is_constant_youngs_modulus() -> None:
    material = SaintVenantKirchhoff1D(youngs_modulus=200e9)
    state = material.trial_state(0.02, material.initial_state())
    assert material.tangent_modulus(state) == pytest.approx(200e9)


def test_1d_is_a_nonlinear_material() -> None:
    assert isinstance(SaintVenantKirchhoff1D(youngs_modulus=200e9), NonlinearMaterial)


@pytest.mark.parametrize("youngs_modulus", [0.0, -1.0])
def test_1d_rejects_non_positive_youngs_modulus(youngs_modulus: float) -> None:
    with pytest.raises(ValidationError):
        SaintVenantKirchhoff1D(youngs_modulus=youngs_modulus)
