"""Tests for the low-level 3D continuum math: shape functions, Jacobian, strain, mass.

Complements tests/test_tet4_element.py and tests/test_hex8_element.py, which
test the element classes; this file tests the underlying pure functions in
femtoolkit.continuum directly.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from femtoolkit.continuum.gauss import GAUSS_2X2X2_POINTS
from femtoolkit.continuum.jacobian import (
    jacobian_determinant_3d,
    jacobian_matrix_3d,
    physical_shape_function_derivatives_3d,
)
from femtoolkit.continuum.mass import (
    hex8_consistent_mass_matrix,
    lumped_mass_matrix,
    tetrahedron_consistent_mass_matrix,
)
from femtoolkit.continuum.shape_functions import (
    hex8_shape_function_derivatives,
    hex8_shape_functions,
    tet4_shape_function_derivatives,
    tet4_shape_functions,
)
from femtoolkit.continuum.strain import (
    hex8_strain_displacement_matrix,
    tet4_strain_displacement_matrix,
)
from femtoolkit.exceptions import DegenerateElementError, ValidationError

_UNIT_TET = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
_UNIT_CUBE = (
    (-1.0, -1.0, -1.0),
    (1.0, -1.0, -1.0),
    (1.0, 1.0, -1.0),
    (-1.0, 1.0, -1.0),
    (-1.0, -1.0, 1.0),
    (1.0, -1.0, 1.0),
    (1.0, 1.0, 1.0),
    (-1.0, 1.0, 1.0),
)


# --- TET4 shape functions -------------------------------------------------


@pytest.mark.parametrize(
    "xi, eta, zeta", [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.25, 0.25, 0.25)]
)
def test_tet4_shape_functions_partition_of_unity(xi: float, eta: float, zeta: float) -> None:
    assert sum(tet4_shape_functions(xi, eta, zeta)) == pytest.approx(1.0)


def test_tet4_shape_functions_kronecker_delta_property() -> None:
    node_natural_coords = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
    for i, (xi, eta, zeta) in enumerate(node_natural_coords):
        values = tet4_shape_functions(xi, eta, zeta)
        expected = [1.0 if j == i else 0.0 for j in range(4)]
        assert_allclose(values, expected, atol=1e-12)


def test_tet4_shape_function_derivatives_sum_to_zero() -> None:
    dn_dxi, dn_deta, dn_dzeta = tet4_shape_function_derivatives()
    assert sum(dn_dxi) == pytest.approx(0.0)
    assert sum(dn_deta) == pytest.approx(0.0)
    assert sum(dn_dzeta) == pytest.approx(0.0)


# --- HEX8 shape functions --------------------------------------------------


@pytest.mark.parametrize("xi, eta, zeta", [(0.0, 0.0, 0.0), (0.5, -0.5, 0.3), (-1.0, 1.0, -1.0)])
def test_hex8_shape_functions_partition_of_unity(xi: float, eta: float, zeta: float) -> None:
    assert sum(hex8_shape_functions(xi, eta, zeta)) == pytest.approx(1.0)


def test_hex8_shape_functions_kronecker_delta_property() -> None:
    for i, (xi, eta, zeta) in enumerate(_UNIT_CUBE):
        values = hex8_shape_functions(xi, eta, zeta)
        expected = [1.0 if j == i else 0.0 for j in range(8)]
        assert_allclose(values, expected, atol=1e-12)


def test_hex8_shape_function_derivatives_sum_to_zero() -> None:
    dn_dxi, dn_deta, dn_dzeta = hex8_shape_function_derivatives(0.3, -0.2, 0.6)
    assert sum(dn_dxi) == pytest.approx(0.0, abs=1e-12)
    assert sum(dn_deta) == pytest.approx(0.0, abs=1e-12)
    assert sum(dn_dzeta) == pytest.approx(0.0, abs=1e-12)


# --- 3D Jacobian ------------------------------------------------------------


def test_tet4_jacobian_determinant_gives_correct_volume() -> None:
    x, y, z = zip(*_UNIT_TET, strict=True)
    dn_dxi, dn_deta, dn_dzeta = tet4_shape_function_derivatives()
    jacobian = jacobian_matrix_3d(dn_dxi, dn_deta, dn_dzeta, x, y, z)
    volume = abs(jacobian_determinant_3d(jacobian)) / 6.0
    assert volume == pytest.approx(1.0 / 6.0)


def test_hex8_jacobian_determinant_gives_correct_volume_via_gauss_sum() -> None:
    x, y, z = zip(*_UNIT_CUBE, strict=True)
    total_volume = 0.0
    for point in GAUSS_2X2X2_POINTS:
        dn_dxi, dn_deta, dn_dzeta = hex8_shape_function_derivatives(point.xi, point.eta, point.zeta)
        _, _, _, det_j = physical_shape_function_derivatives_3d(dn_dxi, dn_deta, dn_dzeta, x, y, z)
        total_volume += point.weight * det_j
    assert total_volume == pytest.approx(8.0)  # (2x2x2 cube)


def test_degenerate_tet4_jacobian_raises_on_physical_derivatives() -> None:
    """Four coplanar points give a zero-volume (singular) Jacobian."""
    x = (0.0, 1.0, 2.0, 3.0)
    y = (0.0, 0.0, 0.0, 0.0)
    z = (0.0, 0.0, 0.0, 0.0)
    dn_dxi, dn_deta, dn_dzeta = tet4_shape_function_derivatives()
    with pytest.raises(DegenerateElementError):
        physical_shape_function_derivatives_3d(dn_dxi, dn_deta, dn_dzeta, x, y, z)


# --- 3D strain-displacement matrices ---------------------------------------


def test_tet4_strain_displacement_matrix_shape() -> None:
    x, y, z = zip(*_UNIT_TET, strict=True)
    dn_dxi, dn_deta, dn_dzeta = tet4_shape_function_derivatives()
    dn_dx, dn_dy, dn_dz, _ = physical_shape_function_derivatives_3d(
        dn_dxi, dn_deta, dn_dzeta, x, y, z
    )
    b_matrix = tet4_strain_displacement_matrix(dn_dx, dn_dy, dn_dz)
    assert b_matrix.shape == (6, 12)


def test_hex8_strain_displacement_matrix_shape() -> None:
    x, y, z = zip(*_UNIT_CUBE, strict=True)
    dn_dxi, dn_deta, dn_dzeta = hex8_shape_function_derivatives(0.0, 0.0, 0.0)
    dn_dx, dn_dy, dn_dz, _ = physical_shape_function_derivatives_3d(
        dn_dxi, dn_deta, dn_dzeta, x, y, z
    )
    b_matrix = hex8_strain_displacement_matrix(dn_dx, dn_dy, dn_dz)
    assert b_matrix.shape == (6, 24)


def test_solid_b_matrix_shear_row_couples_correct_columns() -> None:
    """Row 3 (gamma_xy) must read dNi/dy from the u_i column and dNi/dx from the v_i column."""
    dn_dx = [1.0, 2.0]
    dn_dy = [3.0, 4.0]
    dn_dz = [5.0, 6.0]
    b_matrix = tet4_strain_displacement_matrix(dn_dx, dn_dy, dn_dz)

    # Node 0's u column (col 0): eps_xx row gets dNi/dx, gamma_xy row gets dNi/dy,
    # gamma_xz row gets dNi/dz.
    assert b_matrix[0, 0] == pytest.approx(dn_dx[0])
    assert b_matrix[3, 0] == pytest.approx(dn_dy[0])
    assert b_matrix[5, 0] == pytest.approx(dn_dz[0])
    # Node 0's v column (col 1): eps_yy row gets dNi/dy, gamma_xy row gets dNi/dx,
    # gamma_yz row gets dNi/dz.
    assert b_matrix[1, 1] == pytest.approx(dn_dy[0])
    assert b_matrix[3, 1] == pytest.approx(dn_dx[0])
    assert b_matrix[4, 1] == pytest.approx(dn_dz[0])
    # Node 0's w column (col 2): eps_zz row gets dNi/dz, gamma_yz row gets dNi/dy,
    # gamma_xz row gets dNi/dx.
    assert b_matrix[2, 2] == pytest.approx(dn_dz[0])
    assert b_matrix[4, 2] == pytest.approx(dn_dy[0])
    assert b_matrix[5, 2] == pytest.approx(dn_dx[0])


# --- 3D mass matrices --------------------------------------------------


def test_tetrahedron_consistent_mass_conserves_total_mass() -> None:
    density, volume = 2700.0, 0.05
    mass = tetrahedron_consistent_mass_matrix(density=density, volume=volume)
    assert mass.shape == (12, 12)
    assert_allclose(mass, mass.T)
    assert mass.sum() / 3.0 == pytest.approx(density * volume)


def test_hex8_consistent_mass_conserves_total_mass() -> None:
    x, y, z = zip(*_UNIT_CUBE, strict=True)
    density = 2700.0
    mass = hex8_consistent_mass_matrix(x, y, z, density=density)
    assert mass.shape == (24, 24)
    assert_allclose(mass, mass.T)
    assert mass.sum() / 3.0 == pytest.approx(density * 8.0)


def test_tetrahedron_lumped_mass_is_diagonal_and_conserves_total() -> None:
    density, volume = 2700.0, 0.05
    consistent = tetrahedron_consistent_mass_matrix(density=density, volume=volume)
    lumped = lumped_mass_matrix(consistent)
    assert_allclose(lumped, np.diag(np.diag(lumped)))
    assert lumped.sum() / 3.0 == pytest.approx(density * volume)


@pytest.mark.parametrize("volume", [0.0, -1.0])
def test_tetrahedron_mass_rejects_non_positive_volume(volume: float) -> None:
    with pytest.raises(ValidationError):
        tetrahedron_consistent_mass_matrix(density=1000.0, volume=volume)
