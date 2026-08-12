"""Tests for LinearElastic2D and the plane stress/strain constitutive matrices."""

import math

import numpy as np
import pytest
from numpy.testing import assert_allclose

from femtoolkit.continuum.constitutive import plane_strain_matrix, plane_stress_matrix
from femtoolkit.exceptions import ValidationError
from femtoolkit.materials import LinearElastic2D


def test_valid_material_creation() -> None:
    material = LinearElastic2D(youngs_modulus=210e9, poisson_ratio=0.3, formulation="plane_stress")

    assert material.youngs_modulus == 210e9
    assert material.poisson_ratio == 0.3
    assert material.formulation == "plane_stress"


@pytest.mark.parametrize("youngs_modulus", [0.0, -210e9, math.nan, math.inf, -math.inf])
def test_invalid_youngs_modulus_raises(youngs_modulus: float) -> None:
    with pytest.raises(ValidationError):
        LinearElastic2D(
            youngs_modulus=youngs_modulus, poisson_ratio=0.3, formulation="plane_stress"
        )


@pytest.mark.parametrize("poisson_ratio", [-1.0, 0.5, -1.5, 0.6, math.nan, math.inf])
def test_invalid_poisson_ratio_raises(poisson_ratio: float) -> None:
    with pytest.raises(ValidationError):
        LinearElastic2D(
            youngs_modulus=210e9, poisson_ratio=poisson_ratio, formulation="plane_stress"
        )


def test_invalid_formulation_raises() -> None:
    with pytest.raises(ValidationError):
        LinearElastic2D(youngs_modulus=210e9, poisson_ratio=0.3, formulation="plane_shear")  # type: ignore[arg-type]


def test_constitutive_matrix_dispatches_to_plane_stress() -> None:
    material = LinearElastic2D(youngs_modulus=1.0, poisson_ratio=0.3, formulation="plane_stress")

    assert_allclose(material.constitutive_matrix, plane_stress_matrix(1.0, 0.3))


def test_constitutive_matrix_dispatches_to_plane_strain() -> None:
    material = LinearElastic2D(youngs_modulus=1.0, poisson_ratio=0.3, formulation="plane_strain")

    assert_allclose(material.constitutive_matrix, plane_strain_matrix(1.0, 0.3))


def test_plane_stress_and_plane_strain_differ() -> None:
    plane_stress = plane_stress_matrix(200e9, 0.3)
    plane_strain = plane_strain_matrix(200e9, 0.3)

    assert not np.allclose(plane_stress, plane_strain)


# --- Validation Case 3: hand-derived constitutive matrices ---


def test_plane_stress_matrix_hand_derived_values() -> None:
    """E=1, v=0.3: D = 1/(1-0.09) * [[1,0.3,0],[0.3,1,0],[0,0,0.35]]."""
    d_matrix = plane_stress_matrix(youngs_modulus=1.0, poisson_ratio=0.3)

    factor = 1.0 / 0.91
    expected = np.array(
        [
            [factor * 1.0, factor * 0.3, 0.0],
            [factor * 0.3, factor * 1.0, 0.0],
            [0.0, 0.0, factor * 0.35],
        ]
    )
    assert_allclose(d_matrix, expected, rtol=1e-12)
    assert_allclose(d_matrix[0, 0], 1.098901098901099, rtol=1e-12)
    assert_allclose(d_matrix[0, 1], 0.32967032967033, rtol=1e-12)
    assert_allclose(d_matrix[2, 2], 0.38461538461538464, rtol=1e-12)


def test_plane_strain_matrix_hand_derived_values() -> None:
    """E=1, v=0.3: D = 1/((1.3)(0.4)) * [[0.7,0.3,0],[0.3,0.7,0],[0,0,0.2]]."""
    d_matrix = plane_strain_matrix(youngs_modulus=1.0, poisson_ratio=0.3)

    factor = 1.0 / (1.3 * 0.4)
    expected = np.array(
        [
            [factor * 0.7, factor * 0.3, 0.0],
            [factor * 0.3, factor * 0.7, 0.0],
            [0.0, 0.0, factor * 0.2],
        ]
    )
    assert_allclose(d_matrix, expected, rtol=1e-12)
    assert_allclose(d_matrix[0, 0], 1.3461538461538463, rtol=1e-12)
    assert_allclose(d_matrix[0, 1], 0.5769230769230769, rtol=1e-12)
    assert_allclose(d_matrix[2, 2], 0.38461538461538464, rtol=1e-12)


def test_plane_stress_matrix_is_symmetric() -> None:
    d_matrix = plane_stress_matrix(200e9, 0.3)

    assert_allclose(d_matrix, d_matrix.T)


def test_plane_strain_matrix_is_symmetric() -> None:
    d_matrix = plane_strain_matrix(200e9, 0.3)

    assert_allclose(d_matrix, d_matrix.T)


@pytest.mark.parametrize("youngs_modulus", [0.0, -200e9, math.nan, math.inf])
def test_plane_stress_invalid_youngs_modulus_raises(youngs_modulus: float) -> None:
    with pytest.raises(ValidationError):
        plane_stress_matrix(youngs_modulus, 0.3)


@pytest.mark.parametrize("poisson_ratio", [-1.0, 0.5, math.nan, math.inf])
def test_plane_strain_invalid_poisson_ratio_raises(poisson_ratio: float) -> None:
    with pytest.raises(ValidationError):
        plane_strain_matrix(200e9, poisson_ratio)
