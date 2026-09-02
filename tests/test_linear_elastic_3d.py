"""Tests for femtoolkit.materials.linear_elastic_3d.LinearElastic3D."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from femtoolkit.exceptions import ValidationError
from femtoolkit.materials import LinearElastic3D


def test_constitutive_matrix_shape_and_symmetry() -> None:
    steel = LinearElastic3D(youngs_modulus=200e9, poisson_ratio=0.3, density=7850.0)
    d_matrix = steel.constitutive_matrix

    assert d_matrix.shape == (6, 6)
    assert_allclose(d_matrix, d_matrix.T)
    assert np.all(np.linalg.eigvalsh(d_matrix) > 0)


def test_shear_and_bulk_modulus() -> None:
    steel = LinearElastic3D(youngs_modulus=200e9, poisson_ratio=0.25, density=7850.0)
    assert steel.shear_modulus == pytest.approx(200e9 / (2 * 1.25))
    assert steel.bulk_modulus == pytest.approx(200e9 / (3 * 0.5))


@pytest.mark.parametrize("youngs_modulus", [0.0, -1.0])
def test_rejects_non_positive_youngs_modulus(youngs_modulus: float) -> None:
    with pytest.raises(ValidationError):
        LinearElastic3D(youngs_modulus=youngs_modulus, poisson_ratio=0.3, density=7850.0)


@pytest.mark.parametrize("poisson_ratio", [-1.0, 0.5, 0.6])
def test_rejects_invalid_poisson_ratio(poisson_ratio: float) -> None:
    with pytest.raises(ValidationError):
        LinearElastic3D(youngs_modulus=200e9, poisson_ratio=poisson_ratio, density=7850.0)


@pytest.mark.parametrize("density", [0.0, -1.0])
def test_rejects_non_positive_density(density: float) -> None:
    with pytest.raises(ValidationError):
        LinearElastic3D(youngs_modulus=200e9, poisson_ratio=0.3, density=density)
