"""Tests for femtoolkit.continuum.deformation: deformation gradient, Green-Lagrange strain."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from femtoolkit.continuum.deformation import (
    deformation_gradient,
    displacement_gradient,
    green_lagrange_strain_tensor,
    green_lagrange_strain_voigt,
    right_cauchy_green,
    validate_deformation_gradient,
)
from femtoolkit.exceptions import InvalidDeformationGradientError

# Reference-configuration shape gradients for the unit right tetrahedron
# (nodes at (0,0,0),(1,0,0),(0,1,0),(0,0,1)): dN1=(-1,-1,-1), dN2=(1,0,0),
# dN3=(0,1,0), dN4=(0,0,1).
_TET4_GRADIENTS = np.array([[-1.0, -1.0, -1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])


def test_zero_displacement_gives_identity_deformation_gradient() -> None:
    h = displacement_gradient(np.zeros(12), _TET4_GRADIENTS)
    f = deformation_gradient(h)
    assert_allclose(f, np.eye(3))
    assert_allclose(green_lagrange_strain_tensor(f), np.zeros((3, 3)))


def test_uniform_uniaxial_stretch() -> None:
    """Stretching every node's X coordinate by a factor lambda gives F = diag(lambda,1,1)."""
    lam = 1.3
    ref_coords = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    current_coords = ref_coords.copy()
    current_coords[:, 0] *= lam
    displacements = (current_coords - ref_coords).flatten()

    h = displacement_gradient(displacements, _TET4_GRADIENTS)
    f = deformation_gradient(h)
    assert_allclose(f, np.diag([lam, 1.0, 1.0]), atol=1e-12)

    e = green_lagrange_strain_tensor(f)
    expected_exx = 0.5 * (lam**2 - 1.0)
    assert e[0, 0] == pytest.approx(expected_exx)
    assert_allclose(e[1:, :], 0.0, atol=1e-12)
    assert_allclose(e[:, 1:], 0.0, atol=1e-12)


@pytest.mark.parametrize("axis", ["x", "y", "z"])
def test_rigid_rotation_gives_zero_green_lagrange_strain(axis: str) -> None:
    """The critical objectivity test: pure rotation must give E = 0 exactly."""
    theta = 1.1
    c, s = np.cos(theta), np.sin(theta)
    if axis == "x":
        rotation = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    elif axis == "y":
        rotation = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    else:
        rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    ref_coords = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    current_coords = ref_coords @ rotation.T
    displacements = (current_coords - ref_coords).flatten()

    h = displacement_gradient(displacements, _TET4_GRADIENTS)
    f = deformation_gradient(h)
    assert_allclose(f, rotation, atol=1e-10)

    e = green_lagrange_strain_tensor(f)
    assert_allclose(e, np.zeros((3, 3)), atol=1e-10)

    voigt = green_lagrange_strain_voigt(f)
    assert_allclose(voigt, np.zeros(6), atol=1e-10)


def test_rigid_translation_and_rotation_combined_gives_zero_strain() -> None:
    theta = 0.4
    rotation = np.array(
        [[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]]
    )
    translation = np.array([2.5, -1.3, 0.7])

    ref_coords = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    current_coords = ref_coords @ rotation.T + translation
    displacements = (current_coords - ref_coords).flatten()

    h = displacement_gradient(displacements, _TET4_GRADIENTS)
    f = deformation_gradient(h)
    e = green_lagrange_strain_tensor(f)
    assert_allclose(e, np.zeros((3, 3)), atol=1e-10)


def test_right_cauchy_green_of_rotation_is_identity() -> None:
    theta = 0.8
    rotation = np.array(
        [[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]]
    )
    c = right_cauchy_green(rotation)
    assert_allclose(c, np.eye(3), atol=1e-10)


def test_validate_deformation_gradient_accepts_valid_f() -> None:
    determinant = validate_deformation_gradient(np.eye(3))
    assert determinant == pytest.approx(1.0)


def test_validate_deformation_gradient_rejects_non_finite() -> None:
    bad_f = np.eye(3)
    bad_f[0, 0] = np.nan
    with pytest.raises(InvalidDeformationGradientError):
        validate_deformation_gradient(bad_f)


def test_validate_deformation_gradient_rejects_negative_determinant() -> None:
    inverted_f = np.diag([-1.0, 1.0, 1.0])
    with pytest.raises(InvalidDeformationGradientError):
        validate_deformation_gradient(inverted_f)


def test_validate_deformation_gradient_rejects_near_zero_determinant() -> None:
    collapsed_f = np.diag([1e-12, 1.0, 1.0])
    with pytest.raises(InvalidDeformationGradientError):
        validate_deformation_gradient(collapsed_f)


def test_green_lagrange_strain_voigt_doubles_shear_entries() -> None:
    """Sanity check that the Voigt conversion uses the engineering-shear convention."""
    # A simple shear deformation: x' = x + gamma*y, y'=y, z'=z.
    gamma = 0.05
    f = np.array([[1.0, gamma, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    voigt = green_lagrange_strain_voigt(f)
    tensor = green_lagrange_strain_tensor(f)
    assert voigt[3] == pytest.approx(2.0 * tensor[0, 1])
