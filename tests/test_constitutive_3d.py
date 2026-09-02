"""Tests for femtoolkit.continuum.constitutive.isotropic_3d_matrix (Version 15)."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from femtoolkit.continuum.constitutive import isotropic_3d_matrix
from femtoolkit.exceptions import ValidationError


def test_shape() -> None:
    d_matrix = isotropic_3d_matrix(youngs_modulus=200e9, poisson_ratio=0.3)
    assert d_matrix.shape == (6, 6)


def test_symmetric() -> None:
    d_matrix = isotropic_3d_matrix(youngs_modulus=200e9, poisson_ratio=0.3)
    assert_allclose(d_matrix, d_matrix.T)


@pytest.mark.parametrize("poisson_ratio", [-0.5, 0.0, 0.2, 0.3, 0.45, 0.499])
def test_positive_definite_for_valid_parameters(poisson_ratio: float) -> None:
    d_matrix = isotropic_3d_matrix(youngs_modulus=200e9, poisson_ratio=poisson_ratio)
    eigenvalues = np.linalg.eigvalsh(d_matrix)
    assert np.all(eigenvalues > 0.0)


def test_shear_block_is_diagonal_and_equals_shear_modulus() -> None:
    youngs_modulus, poisson_ratio = 200e9, 0.3
    shear_modulus = youngs_modulus / (2.0 * (1.0 + poisson_ratio))
    d_matrix = isotropic_3d_matrix(youngs_modulus, poisson_ratio)

    assert_allclose(d_matrix[3:, 3:], shear_modulus * np.eye(3))
    assert_allclose(d_matrix[:3, 3:], np.zeros((3, 3)))
    assert_allclose(d_matrix[3:, :3], np.zeros((3, 3)))


def test_hydrostatic_response_matches_bulk_modulus() -> None:
    """A uniform volumetric strain must produce a uniform stress consistent with K."""
    youngs_modulus, poisson_ratio = 210e9, 0.3
    d_matrix = isotropic_3d_matrix(youngs_modulus, poisson_ratio)

    epsilon_vol = 1e-3
    strain = np.array([epsilon_vol / 3.0, epsilon_vol / 3.0, epsilon_vol / 3.0, 0.0, 0.0, 0.0])
    stress = d_matrix @ strain

    bulk_modulus = youngs_modulus / (3.0 * (1.0 - 2.0 * poisson_ratio))
    expected_mean_stress = bulk_modulus * epsilon_vol
    assert stress[0] == pytest.approx(expected_mean_stress, rel=1e-10)
    assert_allclose(stress[:3], stress[0])
    assert_allclose(stress[3:], 0.0, atol=1e-6)


def test_uniaxial_strain_reproduces_youngs_modulus_and_poisson_ratio() -> None:
    """A general anisotropic 3D constraint check: for a strain state with
    only epsilon_xx nonzero, verify D reproduces the correct sigma_xx/epsilon_xx
    ratio is NOT simply E (that only holds under uniaxial *stress*, not strain) --
    instead cross-check against the known closed-form D entries directly."""
    youngs_modulus, poisson_ratio = 200e9, 0.3
    lame_lambda = youngs_modulus * poisson_ratio / ((1 + poisson_ratio) * (1 - 2 * poisson_ratio))
    shear_modulus = youngs_modulus / (2.0 * (1.0 + poisson_ratio))
    d_matrix = isotropic_3d_matrix(youngs_modulus, poisson_ratio)

    assert d_matrix[0, 0] == pytest.approx(lame_lambda + 2.0 * shear_modulus)
    assert d_matrix[0, 1] == pytest.approx(lame_lambda)
    assert d_matrix[1, 1] == pytest.approx(lame_lambda + 2.0 * shear_modulus)


@pytest.mark.parametrize("youngs_modulus", [0.0, -1.0, float("nan")])
def test_rejects_invalid_youngs_modulus(youngs_modulus: float) -> None:
    with pytest.raises(ValidationError):
        isotropic_3d_matrix(youngs_modulus, 0.3)


@pytest.mark.parametrize("poisson_ratio", [-1.0, 0.5, 1.0, float("nan")])
def test_rejects_invalid_poisson_ratio(poisson_ratio: float) -> None:
    with pytest.raises(ValidationError):
        isotropic_3d_matrix(200e9, poisson_ratio)
