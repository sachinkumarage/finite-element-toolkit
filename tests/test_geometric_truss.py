"""Tests for the large-displacement truss in femtoolkit.analysis.geometric_nonlinear."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from femtoolkit.analysis.geometric_nonlinear import (
    GEOMETRIC_NONLINEAR_CAPABLE_ELEMENT_TYPES,
    initial_element_state,
    truss_geometric_internal_force_and_tangent,
    truss_geometric_stiffness_split,
)
from femtoolkit.exceptions import ValidationError
from femtoolkit.materials import Material, SaintVenantKirchhoff1D
from femtoolkit.mesh import Node, TrussElement2D
from femtoolkit.sections import CrossSection


@pytest.fixture
def truss() -> TrussElement2D:
    node_1 = Node(id=1, x=0.0, y=0.0, z=0.0)
    node_2 = Node(id=2, x=3.0, y=4.0, z=0.0)  # reference length 5.0
    placeholder = Material(name="steel", density=7850.0, youngs_modulus=200e9, poissons_ratio=0.3)
    return TrussElement2D(
        id=1, nodes=(node_1, node_2), material=placeholder, cross_section=CrossSection(area=0.001)
    )


@pytest.fixture
def material() -> SaintVenantKirchhoff1D:
    return SaintVenantKirchhoff1D(youngs_modulus=200e9)


def test_truss_is_geometric_nonlinear_capable(truss: TrussElement2D) -> None:
    assert isinstance(truss, GEOMETRIC_NONLINEAR_CAPABLE_ELEMENT_TYPES)


def test_zero_displacement_gives_zero_force(
    truss: TrussElement2D, material: SaintVenantKirchhoff1D
) -> None:
    committed = initial_element_state(truss, material)
    f_int, k_t, trial_state = truss_geometric_internal_force_and_tangent(
        truss, material, [0, 0, 0, 0], committed
    )
    assert_allclose(f_int, np.zeros(4), atol=1e-10)
    assert trial_state.states[0].strain == pytest.approx(0.0)


def test_uniaxial_stretch_gives_expected_green_lagrange_strain(
    truss: TrussElement2D, material: SaintVenantKirchhoff1D
) -> None:
    committed = initial_element_state(truss, material)
    # Stretch along the member's own direction (0.6, 0.8) by 0.5 m.
    stretch_displacement = 0.5
    displacements = [0, 0, 0.6 * stretch_displacement, 0.8 * stretch_displacement]
    _, _, trial_state = truss_geometric_internal_force_and_tangent(
        truss, material, displacements, committed
    )

    new_length = truss.length + stretch_displacement
    expected_strain = (new_length**2 - truss.length**2) / (2 * truss.length**2)
    assert trial_state.states[0].strain == pytest.approx(expected_strain)


def test_current_direction_used_not_reference_direction(
    truss: TrussElement2D, material: SaintVenantKirchhoff1D
) -> None:
    """Internal force must act along the CURRENT (displaced), not reference, direction."""
    committed = initial_element_state(truss, material)
    # Move node 2 perpendicular to the reference axis, changing current direction.
    displacements = [0, 0, -0.8 * 0.4, 0.6 * 0.4]  # perpendicular unit vector times 0.4
    f_int, _, trial_state = truss_geometric_internal_force_and_tangent(
        truss, material, displacements, committed
    )
    # A pure perpendicular displacement stretches the member (length increases),
    # so force should be nonzero and not aligned with the original (0.6, 0.8) direction.
    assert not np.allclose(f_int, 0.0)


@pytest.mark.parametrize("translation", [(0.3, -0.2), (-1.0, 0.5), (2.2, 2.2)])
def test_rigid_translation_gives_zero_force(
    truss: TrussElement2D, material: SaintVenantKirchhoff1D, translation: tuple[float, float]
) -> None:
    committed = initial_element_state(truss, material)
    tx, ty = translation
    displacements = [tx, ty, tx, ty]
    f_int, _, trial_state = truss_geometric_internal_force_and_tangent(
        truss, material, displacements, committed
    )
    assert_allclose(f_int, np.zeros(4), atol=1e-8)
    assert trial_state.states[0].strain == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("theta", [0.3, 1.2, -0.7, np.pi / 2])
def test_rigid_rotation_gives_zero_force(
    truss: TrussElement2D, material: SaintVenantKirchhoff1D, theta: float
) -> None:
    committed = initial_element_state(truss, material)
    reference = np.array([[0.0, 0.0], [3.0, 4.0]])
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    current = reference @ rotation.T
    displacements = (current - reference).flatten()

    f_int, _, trial_state = truss_geometric_internal_force_and_tangent(
        truss, material, displacements, committed
    )
    # Force scale here is ~E*A/L0 ~ 4e7 N; 1e-5 N "zero" is ~12 orders of
    # magnitude smaller -- comfortably absorbs floating-point noise while
    # still being a meaningful objectivity check.
    assert_allclose(f_int, np.zeros(4), atol=1e-5)
    assert trial_state.states[0].strain == pytest.approx(0.0, abs=1e-10)


def test_tangent_stiffness_matches_numerical_differentiation(
    truss: TrussElement2D, material: SaintVenantKirchhoff1D
) -> None:
    committed = initial_element_state(truss, material)
    displacements = np.array([0.05, -0.02, -0.1, 0.15])

    def force(u: np.ndarray) -> np.ndarray:
        f, _, _ = truss_geometric_internal_force_and_tangent(truss, material, u, committed)
        return f

    _, k_closed_form, _ = truss_geometric_internal_force_and_tangent(
        truss, material, displacements, committed
    )

    step = 1e-6
    k_numeric = np.zeros((4, 4))
    for j in range(4):
        perturbation = np.zeros(4)
        perturbation[j] = step
        force_plus = force(displacements + perturbation)
        force_minus = force(displacements - perturbation)
        k_numeric[:, j] = (force_plus - force_minus) / (2 * step)

    assert_allclose(k_closed_form, k_numeric, atol=1e-2, rtol=1e-4)


def test_tangent_stiffness_symmetric(
    truss: TrussElement2D, material: SaintVenantKirchhoff1D
) -> None:
    committed = initial_element_state(truss, material)
    _, k_t, _ = truss_geometric_internal_force_and_tangent(
        truss, material, [0.1, 0.05, -0.05, 0.2], committed
    )
    assert_allclose(k_t, k_t.T)


def test_material_and_geometric_split_sums_to_total_tangent(
    truss: TrussElement2D, material: SaintVenantKirchhoff1D
) -> None:
    committed = initial_element_state(truss, material)
    displacements = [0.1, 0.05, -0.05, 0.2]
    _, k_t, _ = truss_geometric_internal_force_and_tangent(
        truss, material, displacements, committed
    )
    k_material, k_geometric = truss_geometric_stiffness_split(
        truss, material, displacements, committed
    )
    assert_allclose(k_material + k_geometric, k_t)


def test_reference_configuration_reduces_to_linear_truss_stiffness(
    truss: TrussElement2D, material: SaintVenantKirchhoff1D
) -> None:
    """At L=L0, T=0, K_material should equal the standard EA/L*(n outer n) formula."""
    from femtoolkit.analysis.stiffness import truss_element_stiffness_2d

    committed = initial_element_state(truss, material)
    k_material, k_geometric = truss_geometric_stiffness_split(
        truss, material, [0, 0, 0, 0], committed
    )

    cos_theta, sin_theta = truss.direction_cosines
    expected_linear = truss_element_stiffness_2d(
        youngs_modulus=200e9,
        area=0.001,
        length=truss.length,
        cos_theta=cos_theta,
        sin_theta=sin_theta,
    )
    assert_allclose(k_material, expected_linear, atol=1e-6)
    assert_allclose(k_geometric, np.zeros((4, 4)), atol=1e-10)


def test_material_shape_mismatch_raises() -> None:
    node_1 = Node(id=1, x=0.0, y=0.0, z=0.0)
    node_2 = Node(id=2, x=1.0, y=0.0, z=0.0)
    placeholder = Material(name="steel", density=7850.0, youngs_modulus=200e9, poissons_ratio=0.3)
    truss = TrussElement2D(
        id=1, nodes=(node_1, node_2), material=placeholder, cross_section=CrossSection(area=0.001)
    )
    from femtoolkit.materials import SaintVenantKirchhoff3D

    bad_material = SaintVenantKirchhoff3D(youngs_modulus=200e9, poisson_ratio=0.3)
    with pytest.raises(ValidationError):
        initial_element_state(truss, bad_material)
