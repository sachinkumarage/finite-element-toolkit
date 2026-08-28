"""Tests for natural_frequencies, natural_frequencies_of_system, and rigid-body mode handling."""

import math

import numpy as np
import pytest
from numpy.testing import assert_allclose

from femtoolkit.analysis import BoundaryCondition, TranslationDOF
from femtoolkit.analysis.dynamic_system import build_dynamic_system
from femtoolkit.analysis.modal import (
    compute_periods,
    effective_modal_mass,
    effective_modal_mass_ratio,
    influence_vector,
    mass_normalize_mode_shapes,
    modal_analysis,
    modal_analysis_of_system,
    modal_participation_factors,
    natural_frequencies,
    natural_frequencies_of_system,
)
from femtoolkit.exceptions import EigenvalueComputationError, ValidationError
from femtoolkit.geometry import Rectangle
from femtoolkit.materials import LinearElastic2D
from femtoolkit.mesh import create_quad_mesh

# --- Pure eigenvalue math: 2-DOF spring-mass, analytical golden-ratio reference ---


@pytest.fixture
def two_dof_system():
    """Two unit masses, springs k1=k2=1 (m1 to wall, m1 to m2).

    K = [[2,-1],[-1,1]], M = I. Analytical eigenvalues:
    lambda = (3 +/- sqrt(5)) / 2.
    """
    k = np.array([[2.0, -1.0], [-1.0, 1.0]])
    m = np.eye(2)
    return k, m


def test_two_dof_eigenvalues_match_analytical(two_dof_system) -> None:
    k, m = two_dof_system
    result = natural_frequencies(k, m)

    expected = np.array([(3 - math.sqrt(5)) / 2, (3 + math.sqrt(5)) / 2])
    assert_allclose(result.eigenvalues, expected, rtol=1e-9)


def test_two_dof_angular_frequencies_are_sqrt_eigenvalues(two_dof_system) -> None:
    k, m = two_dof_system
    result = natural_frequencies(k, m)
    assert_allclose(result.angular_frequencies, np.sqrt(result.eigenvalues))


def test_two_dof_frequencies_hz_conversion(two_dof_system) -> None:
    k, m = two_dof_system
    result = natural_frequencies(k, m)
    assert_allclose(result.frequencies, result.angular_frequencies / (2 * math.pi))


def test_two_dof_mode_shapes_satisfy_eigenvalue_equation(two_dof_system) -> None:
    k, m = two_dof_system
    result = natural_frequencies(k, m)

    for i in range(2):
        phi = result.mode_shapes[:, i]
        lam = result.eigenvalues[i]
        assert_allclose(k @ phi, lam * (m @ phi), atol=1e-9)


def test_two_dof_mode_shapes_normalized_to_max_abs_one(two_dof_system) -> None:
    k, m = two_dof_system
    result = natural_frequencies(k, m)

    for i in range(2):
        assert_allclose(np.max(np.abs(result.mode_shapes[:, i])), 1.0)


def test_two_dof_no_rigid_body_modes_when_supported(two_dof_system) -> None:
    k, m = two_dof_system
    result = natural_frequencies(k, m)
    assert not result.is_rigid_body_mode.any()


# --- Rigid-body modes ---


def test_free_free_system_flags_rigid_body_mode() -> None:
    k = np.array([[1.0, -1.0], [-1.0, 1.0]])
    m = np.eye(2)
    result = natural_frequencies(k, m)

    assert_allclose(result.eigenvalues[0], 0.0, atol=1e-9)
    assert result.is_rigid_body_mode[0]
    assert not result.is_rigid_body_mode[1]


def test_negative_eigenvalue_beyond_tolerance_raises() -> None:
    """A genuinely negative eigenvalue (non-positive-semi-definite K) is
    not a rigid-body mode -- it signals an invalid model.
    """
    k = np.array([[-1.0, 0.0], [0.0, 1.0]])
    m = np.eye(2)
    with pytest.raises(EigenvalueComputationError):
        natural_frequencies(k, m)


# --- num_modes truncation ---


def test_num_modes_truncates_to_lowest(two_dof_system) -> None:
    k, m = two_dof_system
    result = natural_frequencies(k, m, num_modes=1)

    assert result.eigenvalues.shape == (1,)
    assert result.mode_shapes.shape == (2, 1)
    assert_allclose(result.eigenvalues[0], (3 - math.sqrt(5)) / 2, rtol=1e-9)


def test_num_modes_rejects_out_of_range(two_dof_system) -> None:
    k, m = two_dof_system
    with pytest.raises(ValidationError):
        natural_frequencies(k, m, num_modes=5)


def test_num_modes_rejects_zero(two_dof_system) -> None:
    k, m = two_dof_system
    with pytest.raises(ValidationError):
        natural_frequencies(k, m, num_modes=0)


# --- Input validation ---


def test_rejects_non_square_stiffness() -> None:
    k = np.zeros((2, 3))
    m = np.eye(2)
    with pytest.raises(ValidationError):
        natural_frequencies(k, m)


def test_rejects_mismatched_shapes() -> None:
    k = np.eye(2)
    m = np.eye(3)
    with pytest.raises(ValidationError):
        natural_frequencies(k, m)


def test_rejects_negative_rigid_body_tolerance() -> None:
    k = np.eye(2)
    m = np.eye(2)
    with pytest.raises(ValidationError):
        natural_frequencies(k, m, rigid_body_tolerance=-1e-6)


# --- natural_frequencies_of_system (FEM, with boundary conditions) ---


@pytest.fixture
def cantilever_dynamic_system():
    material = LinearElastic2D(
        youngs_modulus=200e9, poisson_ratio=0.3, formulation="plane_stress", density=7850.0
    )
    domain = Rectangle(width=2.0, height=1.0)
    mesh = create_quad_mesh(width=2.0, height=1.0, nx=4, ny=2, material=material, thickness=0.01)

    bcs = []
    for node in mesh.nodes_on_boundary(domain.boundary("left")):
        bcs.append(BoundaryCondition(node.id, TranslationDOF.X, 0.0))
        bcs.append(BoundaryCondition(node.id, TranslationDOF.Y, 0.0))

    return build_dynamic_system(mesh, bcs)


def test_fem_natural_frequencies_are_finite_and_positive(cantilever_dynamic_system) -> None:
    result = natural_frequencies_of_system(cantilever_dynamic_system, num_modes=5)

    assert np.isfinite(result.frequencies).all()
    assert (result.frequencies > 0).all()


def test_fem_natural_frequencies_ascending(cantilever_dynamic_system) -> None:
    result = natural_frequencies_of_system(cantilever_dynamic_system, num_modes=5)
    assert np.all(np.diff(result.frequencies) >= 0)


def test_fem_no_rigid_body_modes_for_cantilever(cantilever_dynamic_system) -> None:
    """A fully fixed-left cantilever has no rigid-body freedom left."""
    result = natural_frequencies_of_system(cantilever_dynamic_system, num_modes=5)
    assert not result.is_rigid_body_mode.any()


def test_fem_mode_shapes_are_zero_at_constrained_dofs(cantilever_dynamic_system) -> None:
    result = natural_frequencies_of_system(cantilever_dynamic_system, num_modes=3)

    system = cantilever_dynamic_system
    for bc in system.boundary_conditions:
        index = system.dof_map.global_index(bc.node_id, bc.dof)
        assert_allclose(result.mode_shapes[index, :], np.zeros(3), atol=1e-12)


def test_fem_mode_shapes_full_dof_space_shape(cantilever_dynamic_system) -> None:
    result = natural_frequencies_of_system(cantilever_dynamic_system, num_modes=3)
    assert result.mode_shapes.shape == (cantilever_dynamic_system.dof_map.total_dofs, 3)


def test_natural_frequencies_of_system_all_constrained_raises() -> None:
    material = LinearElastic2D(
        youngs_modulus=200e9, poisson_ratio=0.3, formulation="plane_stress", density=7850.0
    )
    mesh = create_quad_mesh(width=1.0, height=1.0, nx=1, ny=1, material=material, thickness=0.01)
    bcs = [
        BoundaryCondition(node.id, dof, 0.0)
        for node in mesh.nodes
        for dof in (TranslationDOF.X, TranslationDOF.Y)
    ]
    system = build_dynamic_system(mesh, bcs)

    with pytest.raises(ValidationError):
        natural_frequencies_of_system(system)


# --- Version 12: mass normalization ---


def test_mass_normalize_mode_shapes_satisfies_phi_t_m_phi_equals_one(two_dof_system) -> None:
    k, m = two_dof_system
    result = natural_frequencies(k, m)
    normalized = mass_normalize_mode_shapes(result.mode_shapes, m)

    for i in range(2):
        phi = normalized[:, i]
        assert_allclose(phi @ m @ phi, 1.0, atol=1e-12)


def test_mass_normalization_is_invariant_to_input_scale(two_dof_system) -> None:
    """Mass-normalizing an arbitrarily rescaled mode shape gives the
    same result (up to sign) as normalizing the default-scaled one.
    """
    k, m = two_dof_system
    result = natural_frequencies(k, m)
    normalized_default = mass_normalize_mode_shapes(result.mode_shapes, m)

    rescaled = result.mode_shapes * np.array([3.0, -7.0])
    normalized_rescaled = mass_normalize_mode_shapes(rescaled, m)

    for i in range(2):
        # Same up to sign: compare absolute values, or align signs first.
        ratio = normalized_rescaled[:, i] / normalized_default[:, i]
        assert_allclose(np.abs(ratio), np.ones(2), atol=1e-9)


def test_mass_normalize_does_not_mutate_input(two_dof_system) -> None:
    k, m = two_dof_system
    result = natural_frequencies(k, m)
    original = result.mode_shapes.copy()
    mass_normalize_mode_shapes(result.mode_shapes, m)
    assert_allclose(result.mode_shapes, original)


# --- Version 12: modal orthogonality ---


def test_mass_and_stiffness_orthogonality_two_dof(two_dof_system) -> None:
    k, m = two_dof_system
    result = natural_frequencies(k, m)
    phi = mass_normalize_mode_shapes(result.mode_shapes, m)

    mass_modal = phi.T @ m @ phi
    stiffness_modal = phi.T @ k @ phi

    assert_allclose(mass_modal, np.eye(2), atol=1e-10)
    assert_allclose(stiffness_modal, np.diag(result.eigenvalues), atol=1e-8)


def test_mass_and_stiffness_orthogonality_fem(cantilever_dynamic_system) -> None:
    """Off-diagonal generalized mass/stiffness terms between distinct
    FEM modes are (numerically) zero -- the general orthogonality
    property, not specific to the 2-DOF analytical case.
    """
    system = cantilever_dynamic_system
    result = natural_frequencies_of_system(system, num_modes=5)
    phi = mass_normalize_mode_shapes(result.mode_shapes, system.mass)

    mass_modal = phi.T @ system.mass @ phi
    stiffness_modal = phi.T @ system.stiffness @ phi

    off_diagonal_mass = mass_modal - np.diag(np.diag(mass_modal))
    off_diagonal_stiffness = stiffness_modal - np.diag(np.diag(stiffness_modal))

    assert_allclose(off_diagonal_mass, np.zeros((5, 5)), atol=1e-8)
    assert_allclose(off_diagonal_stiffness, np.zeros((5, 5)), atol=1.0)
    assert_allclose(np.diag(mass_modal), np.ones(5), atol=1e-8)
    assert_allclose(np.diag(stiffness_modal), result.eigenvalues, rtol=1e-6)


# --- Version 12: periods ---


def test_compute_periods_matches_two_pi_over_omega() -> None:
    omega = np.array([1.0, 2.0, 10.0])
    is_rigid = np.array([False, False, False])
    periods = compute_periods(omega, is_rigid)
    assert_allclose(periods, 2.0 * math.pi / omega)


def test_compute_periods_rigid_body_mode_is_infinite() -> None:
    omega = np.array([0.0, 5.0])
    is_rigid = np.array([True, False])
    periods = compute_periods(omega, is_rigid)
    assert periods[0] == math.inf
    assert_allclose(periods[1], 2.0 * math.pi / 5.0)


# --- Version 12: influence vectors ---


def test_influence_vector_x_direction(cantilever_dynamic_system) -> None:
    system = cantilever_dynamic_system
    r = influence_vector(system.dof_map, "x")
    for node_id in system.dof_map.node_ids:
        assert r[system.dof_map.global_index(node_id, TranslationDOF.X)] == 1.0
        assert r[system.dof_map.global_index(node_id, TranslationDOF.Y)] == 0.0


def test_influence_vector_case_insensitive(cantilever_dynamic_system) -> None:
    system = cantilever_dynamic_system
    assert_allclose(influence_vector(system.dof_map, "X"), influence_vector(system.dof_map, "x"))


def test_influence_vector_rejects_invalid_direction(cantilever_dynamic_system) -> None:
    with pytest.raises(ValidationError):
        influence_vector(cantilever_dynamic_system.dof_map, "z")


# --- Version 12: modal participation factors and effective modal mass ---


def test_participation_factor_times_mode_shape_is_normalization_invariant(
    two_dof_system,
) -> None:
    """A bare Gamma_i is normalization-dependent (see the function's
    docstring), but the physical contribution Gamma_i * phi_i is not:
    computing it from the default-scaled mode shapes or from the
    mass-normalized ones must give the same vector.
    """
    k, m = two_dof_system
    result = natural_frequencies(k, m)
    r = np.array([1.0, 0.0])

    default_shapes = result.mode_shapes
    mass_normalized = mass_normalize_mode_shapes(default_shapes, m)

    gamma_default = modal_participation_factors(default_shapes, m, r)
    gamma_mass_normalized = modal_participation_factors(mass_normalized, m, r)

    for i in range(2):
        contribution_default = gamma_default[i] * default_shapes[:, i]
        contribution_mass_normalized = gamma_mass_normalized[i] * mass_normalized[:, i]
        assert_allclose(contribution_default, contribution_mass_normalized, rtol=1e-9)


def test_mass_normalized_participation_factor_matches_shortcut_formula(two_dof_system) -> None:
    """For already mass-normalized mode shapes, Gamma_i reduces exactly
    to phi_i^T * M * r (no division by generalized mass needed, since
    it is already 1).
    """
    k, m = two_dof_system
    result = natural_frequencies(k, m)
    r = np.array([1.0, 0.0])

    mass_normalized = mass_normalize_mode_shapes(result.mode_shapes, m)
    general = modal_participation_factors(mass_normalized, m, r)
    shortcut = np.array([mass_normalized[:, i] @ m @ r for i in range(2)])

    assert_allclose(general, shortcut, rtol=1e-9)


def test_effective_modal_mass_mass_normalized_shortcut(two_dof_system) -> None:
    k, m = two_dof_system
    result = natural_frequencies(k, m)
    r = np.array([1.0, 0.0])

    participation = modal_participation_factors(result.mode_shapes, m, r)
    eff_mass = effective_modal_mass(participation, result.mode_shapes, m)

    # For mass-normalized modes, M_eff,i = Gamma_i^2 exactly.
    mass_normalized = mass_normalize_mode_shapes(result.mode_shapes, m)
    participation_mn = modal_participation_factors(mass_normalized, m, r)
    assert_allclose(eff_mass, participation_mn**2, rtol=1e-9)


def test_effective_modal_mass_ratio_sums_to_one_over_all_modes(two_dof_system) -> None:
    k, m = two_dof_system
    r = np.array([1.0, 0.0])
    result = modal_analysis(k, m, direction=r)

    assert_allclose(result.cumulative_mass_ratio[-1], 1.0, rtol=1e-9)


def test_effective_modal_mass_ratio_rejects_zero_direction(two_dof_system) -> None:
    k, m = two_dof_system
    with pytest.raises(ValidationError):
        effective_modal_mass_ratio(np.array([1.0, 2.0]), m, np.zeros(2))


# --- Version 12: modal_analysis (raw arrays) ---


def test_modal_analysis_bundles_periods_and_mass_normalized_shapes(two_dof_system) -> None:
    k, m = two_dof_system
    result = modal_analysis(stiffness_matrix=k, mass_matrix=m, num_modes=2)

    assert result.periods.shape == (2,)
    assert result.mass_normalized_mode_shapes.shape == (2, 2)
    assert result.participation_factors is None
    assert result.effective_modal_mass is None


def test_modal_analysis_direction_wrong_length_raises(two_dof_system) -> None:
    k, m = two_dof_system
    with pytest.raises(ValidationError):
        modal_analysis(k, m, direction=np.array([1.0, 0.0, 0.0]))


def test_modal_result_properties_delegate_to_base(two_dof_system) -> None:
    k, m = two_dof_system
    base = natural_frequencies(k, m)
    result = modal_analysis(k, m)

    assert_allclose(result.eigenvalues, base.eigenvalues)
    assert_allclose(result.angular_frequencies, base.angular_frequencies)
    assert_allclose(result.frequencies, base.frequencies)
    assert_allclose(result.mode_shapes, base.mode_shapes)
    assert_allclose(result.is_rigid_body_mode, base.is_rigid_body_mode)


# --- Version 12: modal_analysis_of_system (FEM, with direction) ---


def test_modal_analysis_of_system_with_direction_populates_participation(
    cantilever_dynamic_system,
) -> None:
    result = modal_analysis_of_system(cantilever_dynamic_system, num_modes=5, direction="x")

    assert result.participation_factors is not None
    assert result.participation_factors.shape == (5,)
    assert result.effective_modal_mass.shape == (5,)
    assert result.effective_modal_mass_ratio.shape == (5,)
    assert result.cumulative_mass_ratio.shape == (5,)
    assert np.all(np.diff(result.cumulative_mass_ratio) >= -1e-12)  # non-decreasing


def test_modal_analysis_of_system_cumulative_mass_approaches_total_with_all_modes(
    cantilever_dynamic_system,
) -> None:
    system = cantilever_dynamic_system
    n_free = system.dof_map.total_dofs - len(system.boundary_conditions)

    result = modal_analysis_of_system(system, num_modes=n_free, direction="x")

    assert_allclose(result.cumulative_mass_ratio[-1], 1.0, rtol=1e-6)


def test_modal_analysis_of_system_without_direction_leaves_participation_none(
    cantilever_dynamic_system,
) -> None:
    result = modal_analysis_of_system(cantilever_dynamic_system, num_modes=3)
    assert result.participation_factors is None
    assert result.direction is None


def test_modal_analysis_of_system_accepts_raw_direction_array(cantilever_dynamic_system) -> None:
    system = cantilever_dynamic_system
    r = influence_vector(system.dof_map, "y")
    result = modal_analysis_of_system(system, num_modes=3, direction=r)
    assert result.participation_factors is not None


def test_modal_analysis_of_system_invalid_direction_string_raises(
    cantilever_dynamic_system,
) -> None:
    with pytest.raises(ValidationError):
        modal_analysis_of_system(cantilever_dynamic_system, num_modes=3, direction="z")


def test_modal_analysis_of_system_periods_finite_for_physical_modes(
    cantilever_dynamic_system,
) -> None:
    result = modal_analysis_of_system(cantilever_dynamic_system, num_modes=5)
    # No rigid-body modes expected for a fixed cantilever.
    assert np.all(np.isfinite(result.periods))
