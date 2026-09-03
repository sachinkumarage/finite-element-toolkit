"""Tests for the geometrically nonlinear TET4 dispatch (femtoolkit.analysis.geometric_nonlinear)."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from femtoolkit.analysis.geometric_nonlinear import (
    GEOMETRIC_NONLINEAR_CAPABLE_ELEMENT_TYPES,
    initial_element_state,
    tet4_geometric_internal_force_and_tangent,
    tet4_geometric_stiffness_split,
)
from femtoolkit.exceptions import InvalidDeformationGradientError
from femtoolkit.materials import LinearElastic3D, SaintVenantKirchhoff3D
from femtoolkit.mesh import Node, Tet4Element3D


@pytest.fixture
def tet() -> Tet4Element3D:
    nodes = (
        Node(id=1, x=0.0, y=0.0, z=0.0),
        Node(id=2, x=1.0, y=0.0, z=0.0),
        Node(id=3, x=0.0, y=1.0, z=0.0),
        Node(id=4, x=0.0, y=0.0, z=1.0),
    )
    placeholder = LinearElastic3D(youngs_modulus=200e9, poisson_ratio=0.3, density=7850.0)
    return Tet4Element3D(id=1, nodes=nodes, material=placeholder)


@pytest.fixture
def material() -> SaintVenantKirchhoff3D:
    return SaintVenantKirchhoff3D(youngs_modulus=200e9, poisson_ratio=0.3)


def test_tet4_is_geometric_nonlinear_capable(tet: Tet4Element3D) -> None:
    assert isinstance(tet, GEOMETRIC_NONLINEAR_CAPABLE_ELEMENT_TYPES)


def test_zero_displacement_gives_zero_force(
    tet: Tet4Element3D, material: SaintVenantKirchhoff3D
) -> None:
    committed = initial_element_state(tet, material)
    f_int, k_t, trial_state = tet4_geometric_internal_force_and_tangent(
        tet, material, np.zeros(12), committed
    )
    assert_allclose(f_int, np.zeros(12), atol=1e-8)
    assert_allclose(trial_state.states[0].strain, np.zeros(6), atol=1e-12)


def test_uniform_uniaxial_stretch_gives_expected_strain(
    tet: Tet4Element3D, material: SaintVenantKirchhoff3D
) -> None:
    committed = initial_element_state(tet, material)
    lam = 1.2
    ref_coords = np.array([[n.x, n.y, n.z] for n in tet.nodes])
    current = ref_coords.copy()
    current[:, 0] *= lam
    displacements = (current - ref_coords).flatten()

    _, _, trial_state = tet4_geometric_internal_force_and_tangent(
        tet, material, displacements, committed
    )
    expected_exx = 0.5 * (lam**2 - 1.0)
    assert trial_state.states[0].strain[0] == pytest.approx(expected_exx)
    assert_allclose(trial_state.states[0].strain[1:], 0.0, atol=1e-12)


@pytest.mark.parametrize("translation", [(0.2, -0.1, 0.3), (1.0, 1.0, 1.0), (-0.5, 0.4, 0.2)])
def test_rigid_translation_gives_zero_force(
    tet: Tet4Element3D, material: SaintVenantKirchhoff3D, translation: tuple[float, float, float]
) -> None:
    committed = initial_element_state(tet, material)
    displacements = np.tile(translation, 4)
    f_int, _, trial_state = tet4_geometric_internal_force_and_tangent(
        tet, material, displacements, committed
    )
    assert_allclose(f_int, np.zeros(12), atol=1e-6)
    assert_allclose(trial_state.states[0].strain, np.zeros(6), atol=1e-10)


@pytest.mark.parametrize("theta", [0.5, -1.0, np.pi / 3])
def test_rigid_rotation_gives_zero_force(
    tet: Tet4Element3D, material: SaintVenantKirchhoff3D, theta: float
) -> None:
    committed = initial_element_state(tet, material)
    rotation = np.array(
        [[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]]
    )
    ref_coords = np.array([[n.x, n.y, n.z] for n in tet.nodes])
    current = ref_coords @ rotation.T
    displacements = (current - ref_coords).flatten()

    f_int, _, trial_state = tet4_geometric_internal_force_and_tangent(
        tet, material, displacements, committed
    )
    assert_allclose(f_int, np.zeros(12), atol=1e-5)
    assert_allclose(trial_state.states[0].strain, np.zeros(6), atol=1e-8)


def test_tangent_symmetric(tet: Tet4Element3D, material: SaintVenantKirchhoff3D) -> None:
    committed = initial_element_state(tet, material)
    displacements = np.array([0, 0, 0, 0.05, 0.02, -0.01, 0.01, 0.06, 0.0, -0.02, 0.0, 0.05])
    _, k_t, _ = tet4_geometric_internal_force_and_tangent(tet, material, displacements, committed)
    assert_allclose(k_t, k_t.T, atol=1e-1)


def test_material_and_geometric_split_sums_to_total(
    tet: Tet4Element3D, material: SaintVenantKirchhoff3D
) -> None:
    committed = initial_element_state(tet, material)
    displacements = np.array([0, 0, 0, 0.05, 0.02, -0.01, 0.01, 0.06, 0.0, -0.02, 0.0, 0.05])
    _, k_t, _ = tet4_geometric_internal_force_and_tangent(tet, material, displacements, committed)
    k_material, k_geometric = tet4_geometric_stiffness_split(
        tet, material, displacements, committed
    )
    assert_allclose(k_material + k_geometric, k_t)


def test_reference_configuration_geometric_stiffness_is_zero(
    tet: Tet4Element3D, material: SaintVenantKirchhoff3D
) -> None:
    """At zero strain, S=0, so the initial-stress geometric stiffness must vanish."""
    committed = initial_element_state(tet, material)
    k_material, k_geometric = tet4_geometric_stiffness_split(
        tet, material, np.zeros(12), committed
    )
    assert_allclose(k_geometric, np.zeros((12, 12)), atol=1e-8)


def test_reference_configuration_material_stiffness_matches_linear_tet4(
    tet: Tet4Element3D, material: SaintVenantKirchhoff3D
) -> None:
    """At zero strain, K_material should match the existing small-strain TET4 stiffness."""
    from femtoolkit.continuum.constitutive import isotropic_3d_matrix

    committed = initial_element_state(tet, material)
    k_material, _ = tet4_geometric_stiffness_split(tet, material, np.zeros(12), committed)

    d_matrix = isotropic_3d_matrix(200e9, 0.3)
    expected = tet.volume * tet.b_matrix.T @ d_matrix @ tet.b_matrix
    # k_material comes from a numerically differentiated total tangent (central
    # difference, O(h^2) truncation error); rtol=1e-4 comfortably covers that.
    assert_allclose(k_material, expected, atol=1.0, rtol=1e-4)


def test_element_inversion_raises(tet: Tet4Element3D, material: SaintVenantKirchhoff3D) -> None:
    committed = initial_element_state(tet, material)
    # Collapse node 2 through node 1 and beyond, inverting the element.
    displacements = np.array([0, 0, 0, -5, 0, 0, 0, 0, 0, 0, 0, 0], dtype=float)
    with pytest.raises(InvalidDeformationGradientError):
        tet4_geometric_internal_force_and_tangent(tet, material, displacements, committed)
