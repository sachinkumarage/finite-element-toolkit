"""Tests for continuum stress recovery (femtoolkit.continuum.stress)."""

import math

import numpy as np
from numpy.testing import assert_allclose

from femtoolkit.continuum.stress import (
    principal_stresses_2d,
    stress_from_strain,
    von_mises_3d,
    von_mises_plane_strain,
    von_mises_plane_stress,
)


def test_stress_from_strain_matches_direct_matrix_multiplication() -> None:
    d_matrix = np.array([[2.0, 0.5, 0.0], [0.5, 2.0, 0.0], [0.0, 0.0, 0.75]])
    strain = [0.001, 0.0005, 0.0002]

    stress = stress_from_strain(d_matrix, strain)

    assert_allclose(stress, d_matrix @ np.array(strain))


def test_von_mises_uniaxial_stress_equals_the_stress_itself() -> None:
    """Under pure uniaxial stress, the von Mises equivalent stress equals
    the applied stress -- the defining sanity check for the formula.
    """
    assert_allclose(von_mises_plane_stress(100.0, 0.0, 0.0), 100.0)
    assert_allclose(von_mises_plane_stress(0.0, 100.0, 0.0), 100.0)


def test_von_mises_pure_shear() -> None:
    """Under pure shear, sigma_vm = sqrt(3) * tau."""
    assert_allclose(von_mises_plane_stress(0.0, 0.0, 50.0), math.sqrt(3) * 50.0)


def test_von_mises_zero_stress_is_zero() -> None:
    assert_allclose(von_mises_plane_stress(0.0, 0.0, 0.0), 0.0)


def test_von_mises_plane_stress_reduces_from_3d_formula_with_zero_sigma_z() -> None:
    sigma_x, sigma_y, tau_xy = 120.0, -40.0, 30.0

    assert_allclose(
        von_mises_plane_stress(sigma_x, sigma_y, tau_xy),
        von_mises_3d(sigma_x, sigma_y, 0.0, tau_xy),
    )


def test_von_mises_plane_strain_includes_out_of_plane_stress() -> None:
    """Plane strain must NOT match the plane-stress formula when sigma_z != 0."""
    sigma_x, sigma_y, tau_xy, poisson_ratio = 100.0, 20.0, 15.0, 0.3
    sigma_z = poisson_ratio * (sigma_x + sigma_y)

    plane_strain_value = von_mises_plane_strain(sigma_x, sigma_y, tau_xy, poisson_ratio)
    plane_stress_value = von_mises_plane_stress(sigma_x, sigma_y, tau_xy)
    expected = von_mises_3d(sigma_x, sigma_y, sigma_z, tau_xy)

    assert_allclose(plane_strain_value, expected)
    assert plane_strain_value != plane_stress_value


def test_von_mises_plane_strain_matches_plane_stress_when_poisson_ratio_is_zero() -> None:
    """With poisson_ratio=0, sigma_z=0, so plane strain reduces to plane stress."""
    sigma_x, sigma_y, tau_xy = 80.0, -10.0, 5.0

    assert_allclose(
        von_mises_plane_strain(sigma_x, sigma_y, tau_xy, poisson_ratio=0.0),
        von_mises_plane_stress(sigma_x, sigma_y, tau_xy),
    )


def test_principal_stresses_uniaxial() -> None:
    sigma_1, sigma_2 = principal_stresses_2d(100.0, 0.0, 0.0)

    assert_allclose(sigma_1, 100.0)
    assert_allclose(sigma_2, 0.0, atol=1e-12)


def test_principal_stresses_pure_shear() -> None:
    sigma_1, sigma_2 = principal_stresses_2d(0.0, 0.0, 50.0)

    assert_allclose(sigma_1, 50.0)
    assert_allclose(sigma_2, -50.0)


def test_principal_stresses_ordering() -> None:
    """sigma_1 must always be the larger (or equal) principal stress."""
    sigma_1, sigma_2 = principal_stresses_2d(30.0, 90.0, 40.0)

    assert sigma_1 >= sigma_2


def test_principal_stresses_average_equals_hydrostatic_average() -> None:
    sigma_x, sigma_y, tau_xy = 70.0, -20.0, 25.0

    sigma_1, sigma_2 = principal_stresses_2d(sigma_x, sigma_y, tau_xy)

    assert_allclose((sigma_1 + sigma_2) / 2.0, (sigma_x + sigma_y) / 2.0)
