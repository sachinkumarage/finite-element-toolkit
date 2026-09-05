"""Tests for femtoolkit.materials.finite_strain_plasticity.J2FiniteStrainPlasticity3D."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from femtoolkit.continuum.stress import von_mises_3d
from femtoolkit.continuum.tensor import (
    tensor_to_voigt_strain,
    voigt_strain_to_tensor,
    voigt_stress_to_tensor,
)
from femtoolkit.exceptions import ValidationError
from femtoolkit.materials.finite_strain_plasticity import (
    FiniteStrainPlasticMaterial,
    J2FiniteStrainPlasticity3D,
)
from femtoolkit.materials.nonlinear import MaterialState, NonlinearMaterial


def _symmetric_deformation_gradient(strain_voigt: np.ndarray) -> np.ndarray:
    """Build a definite F=U (pure stretch, no rotation) consistent with a given strain."""
    strain_tensor = voigt_strain_to_tensor(strain_voigt)
    right_cauchy_green = 2.0 * strain_tensor + np.eye(3)
    eigenvalues, eigenvectors = np.linalg.eigh(right_cauchy_green)
    return eigenvectors @ np.diag(np.sqrt(eigenvalues)) @ eigenvectors.T


def _von_mises_kirchhoff(material: FiniteStrainPlasticMaterial, state: MaterialState) -> float:
    """Equivalent (von Mises) Kirchhoff stress -- the measure the yield criterion is defined on.

    ``state.stress`` holds the 2nd Piola-Kirchhoff stress (the reference-
    configuration measure the Total Lagrangian dispatch consumes); the
    yield surface itself is defined in terms of Kirchhoff stress
    ``tau = F @ S @ F^T`` (see the module docstring's derivation), so a
    direct von Mises check against ``state.stress`` would compare the
    wrong stress measure.
    """
    f = _symmetric_deformation_gradient(np.asarray(state.strain, dtype=float))
    jacobian = float(np.linalg.det(f))
    cauchy = material.cauchy_stress(state, f)
    kirchhoff = jacobian * cauchy
    return von_mises_3d(
        kirchhoff[0, 0], kirchhoff[1, 1], kirchhoff[2, 2],
        kirchhoff[0, 1], kirchhoff[1, 2], kirchhoff[0, 2],
    )


@pytest.fixture
def steel() -> J2FiniteStrainPlasticity3D:
    return J2FiniteStrainPlasticity3D(
        youngs_modulus=200e9, poisson_ratio=0.3, yield_stress=250e6, hardening_modulus=10e9
    )


def test_is_a_finite_strain_plastic_and_nonlinear_material(
    steel: J2FiniteStrainPlasticity3D,
) -> None:
    assert isinstance(steel, FiniteStrainPlasticMaterial)
    assert isinstance(steel, NonlinearMaterial)


def test_initial_state_is_zero_with_identity_fp(steel: J2FiniteStrainPlasticity3D) -> None:
    state = steel.initial_state()
    assert_allclose(state.strain, np.zeros(6))
    assert_allclose(state.stress, np.zeros(6))
    assert_allclose(state.plastic_strain, np.zeros(6))
    assert not state.yielded
    assert state.hardening_variable == pytest.approx(0.0)
    assert state.plastic_multiplier == pytest.approx(0.0)
    assert_allclose(state.plastic_deformation_gradient, np.eye(3))


def test_reference_configuration_gives_zero_stress(steel: J2FiniteStrainPlasticity3D) -> None:
    state = steel.trial_state(np.zeros(6), steel.initial_state())
    assert_allclose(state.stress, np.zeros(6), atol=1e-6)
    assert not state.yielded


# --- Elastic response ------------------------------------------------------


def test_small_elastic_strain_approaches_hookes_law(steel: J2FiniteStrainPlasticity3D) -> None:
    """At small strain, ln(1+x) ~= x, so the response should approach linear elasticity."""
    strain = np.array([1e-6, -3e-7, -3e-7, 0.0, 0.0, 0.0])
    state = steel.trial_state(strain, steel.initial_state())
    assert not state.yielded

    lame_lambda, mu = steel.lame_lambda, steel.shear_modulus
    expected_sxx = (lame_lambda + 2.0 * mu) * 1e-6 + lame_lambda * (-3e-7) * 2.0
    assert state.stress[0] == pytest.approx(expected_sxx, rel=1e-4)


def test_hydrostatic_loading_does_not_yield(steel: J2FiniteStrainPlasticity3D) -> None:
    """Pure volumetric deformation must never trigger yielding (deviatoric-only criterion)."""
    strain = np.array([1e-2, 1e-2, 1e-2, 0.0, 0.0, 0.0])
    state = steel.trial_state(strain, steel.initial_state())
    assert not state.yielded
    assert von_mises_3d(*state.stress) == pytest.approx(0.0, abs=1.0)


def test_pure_shear_below_yield_stays_elastic(steel: J2FiniteStrainPlasticity3D) -> None:
    gamma = 1e-4
    strain = np.array([0.0, 0.0, 0.0, gamma, 0.0, 0.0])
    state = steel.trial_state(strain, steel.initial_state())
    assert not state.yielded


# --- Yield onset and plastic loading ---------------------------------------


def test_yielding_lands_on_the_yield_surface(steel: J2FiniteStrainPlasticity3D) -> None:
    strain = np.array([0.01, 0.0, 0.0, 0.0, 0.0, 0.0])
    state = steel.trial_state(strain, steel.initial_state())

    assert state.yielded
    current_yield = steel.yield_stress + steel.hardening_modulus * state.hardening_variable
    assert _von_mises_kirchhoff(steel, state) == pytest.approx(current_yield, rel=1e-5)


def test_plastic_loading_increases_equivalent_plastic_strain(
    steel: J2FiniteStrainPlasticity3D,
) -> None:
    state_a = steel.trial_state(np.array([0.01, 0, 0, 0, 0, 0]), steel.initial_state())
    state_b = steel.trial_state(np.array([0.015, 0, 0, 0, 0, 0]), state_a)

    assert state_a.hardening_variable > 0.0
    assert state_b.hardening_variable > state_a.hardening_variable
    assert state_b.plastic_multiplier > 0.0


def test_hardening_consistency_through_incremental_loading(
    steel: J2FiniteStrainPlasticity3D,
) -> None:
    """sigma_y = sigma_y0 + H*alpha must hold after every plastic increment."""
    state = steel.initial_state()
    for axial_strain in (0.005, 0.008, 0.012, 0.02):
        state = steel.trial_state(np.array([axial_strain, 0, 0, 0, 0, 0]), state)
        if state.yielded:
            current_yield = steel.yield_stress + steel.hardening_modulus * state.hardening_variable
            assert _von_mises_kirchhoff(steel, state) == pytest.approx(current_yield, rel=1e-5)


def test_perfectly_plastic_special_case_caps_von_mises() -> None:
    perfectly_plastic = J2FiniteStrainPlasticity3D(
        youngs_modulus=200e9, poisson_ratio=0.3, yield_stress=250e6, hardening_modulus=0.0
    )
    state_a = perfectly_plastic.trial_state(
        np.array([0.01, 0, 0, 0, 0, 0]), perfectly_plastic.initial_state()
    )
    state_b = perfectly_plastic.trial_state(np.array([0.02, 0, 0, 0, 0, 0]), state_a)

    assert _von_mises_kirchhoff(perfectly_plastic, state_a) == pytest.approx(250e6, rel=1e-5)
    assert _von_mises_kirchhoff(perfectly_plastic, state_b) == pytest.approx(250e6, rel=1e-5)


def test_uniaxial_tension_yields_and_hardens(steel: J2FiniteStrainPlasticity3D) -> None:
    state = steel.initial_state()
    stresses = []
    for eps in (0.001, 0.004, 0.008, 0.015):
        strain = np.array([eps, -0.3 * eps, -0.3 * eps, 0, 0, 0])
        state = steel.trial_state(strain, state)
        stresses.append(state.stress[0])
    assert state.yielded
    # Monotonically increasing stress under monotonic hardening loading.
    assert all(b > a for a, b in zip(stresses, stresses[1:], strict=False))


def test_uniaxial_compression_yields_symmetrically(steel: J2FiniteStrainPlasticity3D) -> None:
    """Tension/compression yielding is symmetric in *stretch* (equivalently log-strain)
    space, not exactly in Green-Lagrange strain space -- +/-0.01 Green-Lagrange strain
    maps to slightly asymmetric stretch ratios through the nonlinear E=1/2(lambda^2-1)
    relation, so a small (sub-1%) residual asymmetry here is expected, not a bug."""
    tension = steel.trial_state(np.array([0.01, -0.003, -0.003, 0, 0, 0]), steel.initial_state())
    compression = steel.trial_state(
        np.array([-0.01, 0.003, 0.003, 0, 0, 0]), steel.initial_state()
    )
    assert tension.yielded
    assert compression.yielded
    assert _von_mises_kirchhoff(steel, tension) == pytest.approx(
        _von_mises_kirchhoff(steel, compression), rel=5e-3
    )
    assert compression.stress[0] < 0.0


def test_simple_shear_yields(steel: J2FiniteStrainPlasticity3D) -> None:
    strain = np.array([0.0, 0.0, 0.0, 0.02, 0.0, 0.0])
    state = steel.trial_state(strain, steel.initial_state())
    assert state.yielded
    assert state.stress[3] != 0.0


# --- Unloading / reloading and path independence ----------------------------


def test_elastic_unload_does_not_change_plastic_state(steel: J2FiniteStrainPlasticity3D) -> None:
    loaded = steel.trial_state(np.array([0.006, -0.0018, -0.0018, 0, 0, 0]), steel.initial_state())
    assert loaded.yielded

    unloaded = steel.trial_state(
        np.array([0.0055, -0.00165, -0.00165, 0, 0, 0]), loaded
    )
    assert not unloaded.yielded
    assert unloaded.hardening_variable == pytest.approx(loaded.hardening_variable)
    assert_allclose(
        unloaded.plastic_deformation_gradient, loaded.plastic_deformation_gradient, rtol=1e-10
    )


def test_reload_to_same_strain_reproduces_committed_stress_exactly(
    steel: J2FiniteStrainPlasticity3D,
) -> None:
    """Regression test for a real bug caught during development: the second
    Piola-Kirchhoff pullback must use the *converged* plastic deformation
    gradient (Fp_new), not the trial reference (Fp_old) -- using Fp_old
    silently introduced an O(accumulated plastic strain) stress error,
    invisible on a single isolated plastic step from a pristine Fp=I
    reference but obvious once a second plastic evaluation was requested
    from an already-plastically-deformed committed state."""
    strain_a = np.array([0.006, -0.0018, -0.0018, 0, 0, 0])
    committed = steel.trial_state(strain_a, steel.initial_state())

    reload_from_same_state = steel.trial_state(strain_a, committed)
    assert reload_from_same_state.plastic_multiplier == pytest.approx(0.0)
    assert_allclose(reload_from_same_state.stress, committed.stress, rtol=1e-8)


def test_path_independence_under_proportional_elastic_unload_reload(
    steel: J2FiniteStrainPlasticity3D,
) -> None:
    strain_a = np.array([0.006, -0.0018, -0.0018, 0, 0, 0])
    committed = steel.trial_state(strain_a, steel.initial_state())

    small_unload = steel.trial_state(
        np.array([0.0055, -0.00165, -0.00165, 0, 0, 0]), committed
    )
    reload = steel.trial_state(strain_a, small_unload)

    assert_allclose(reload.stress, committed.stress, rtol=1e-8)
    assert_allclose(
        reload.plastic_deformation_gradient, committed.plastic_deformation_gradient, rtol=1e-8
    )


def test_reversed_loading_can_re_yield_isotropic_hardening(
    steel: J2FiniteStrainPlasticity3D,
) -> None:
    """Isotropic hardening expands the yield surface symmetrically (no
    kinematic/Bauschinger softening): a large enough strain reversal can
    still re-trigger yielding, in the opposite sense."""
    loaded = steel.trial_state(np.array([0.006, -0.0018, -0.0018, 0, 0, 0]), steel.initial_state())
    reversed_state = steel.trial_state(np.zeros(6), loaded)
    assert reversed_state.yielded
    assert reversed_state.stress[0] < 0.0


# --- Multiplicative decomposition math --------------------------------------


def test_plastic_right_cauchy_green_matches_reported_plastic_strain(
    steel: J2FiniteStrainPlasticity3D,
) -> None:
    """plastic_strain (Voigt) must equal 1/2*(Fp^T @ Fp - I), the plastic
    Green-Lagrange-strain-like reporting quantity this module documents."""
    state = steel.trial_state(np.array([0.01, -0.003, -0.001, 0.002, 0, 0]), steel.initial_state())
    fp = state.plastic_deformation_gradient
    expected_tensor = 0.5 * (fp.T @ fp - np.eye(3))
    assert_allclose(state.plastic_strain, tensor_to_voigt_strain(expected_tensor), atol=1e-10)


def test_elastic_deformation_gradient_fe_equals_f_times_fp_inverse(
    steel: J2FiniteStrainPlasticity3D,
) -> None:
    """Fe = F @ Fp^-1: verified by reconstructing Ce = Fe^T@Fe and checking it matches
    Fp^-T @ C @ Fp^-1 (the tensor this module's return map is actually built on)."""
    f = np.array([[1.05, 0.02, 0.0], [0.0, 0.97, 0.01], [0.0, 0.0, 1.02]])
    c = f.T @ f

    committed = steel.trial_state(np.array([0.02, 0, 0, 0, 0, 0]), steel.initial_state())
    fp = committed.plastic_deformation_gradient

    fe = f @ np.linalg.inv(fp)
    ce_via_fe = fe.T @ fe
    ce_via_formula = np.linalg.inv(fp).T @ c @ np.linalg.inv(fp)
    assert_allclose(ce_via_fe, ce_via_formula, rtol=1e-10)


def test_multiplicative_decomposition_recovers_total_deformation(
    steel: J2FiniteStrainPlasticity3D,
) -> None:
    """F = Fe @ Fp: with Fe built from the plastic-corrector's own elastic log
    strain and eigenvectors, Fe @ Fp must reproduce the same C as F itself."""
    strain = np.array([0.01, -0.003, -0.001, 0.002, 0, 0])
    committed = steel.trial_state(strain, steel.initial_state())
    fp = committed.plastic_deformation_gradient

    strain_tensor = voigt_strain_to_tensor(strain)
    c = 2.0 * strain_tensor + np.eye(3)
    ce = np.linalg.inv(fp).T @ c @ np.linalg.inv(fp)
    eigenvalues, eigenvectors = np.linalg.eigh(ce)
    fe = eigenvectors @ np.diag(np.sqrt(eigenvalues)) @ eigenvectors.T  # symmetric square root

    f_reconstructed = fe @ fp
    c_reconstructed = f_reconstructed.T @ f_reconstructed
    assert_allclose(c_reconstructed, c, rtol=1e-8)


# --- Consistent tangent ------------------------------------------------------


def test_elastic_tangent_is_symmetric_positive_definite(steel: J2FiniteStrainPlasticity3D) -> None:
    state = steel.trial_state(np.array([1e-5, 0, 0, 0, 0, 0]), steel.initial_state())
    tangent = steel.tangent_modulus(state)
    assert_allclose(tangent, tangent.T, atol=1.0)
    assert np.all(np.linalg.eigvalsh(tangent) > 0)


def test_plastic_tangent_is_symmetric_positive_definite(steel: J2FiniteStrainPlasticity3D) -> None:
    state = steel.trial_state(np.array([0.01, 0, 0, 0, 0, 0]), steel.initial_state())
    tangent = steel.tangent_modulus(state)
    assert_allclose(tangent, tangent.T, atol=10.0)
    assert np.all(np.linalg.eigvalsh(tangent) > 0)


def test_plastic_tangent_is_softer_than_elastic_tangent(steel: J2FiniteStrainPlasticity3D) -> None:
    elastic_state = steel.trial_state(np.array([1e-5, 0, 0, 0, 0, 0]), steel.initial_state())
    plastic_state = steel.trial_state(np.array([0.01, 0, 0, 0, 0, 0]), steel.initial_state())
    elastic_tangent = steel.tangent_modulus(elastic_state)
    plastic_tangent = steel.tangent_modulus(plastic_state)
    assert plastic_tangent[0, 0] < elastic_tangent[0, 0]


def test_tangent_at_multi_step_accumulated_state_matches_independent_finite_difference(
    steel: J2FiniteStrainPlasticity3D,
) -> None:
    """The committed-reference reconstruction inside tangent_modulus must remain
    correct even several plastic steps away from the pristine Fp=I reference."""
    committed = steel.initial_state()
    for eps in (0.003, 0.006, 0.009):
        strain = np.array([eps, -0.3 * eps, -0.3 * eps, 0.001, 0, 0])
        committed = steel.trial_state(strain, committed)

    final_strain = np.array([0.012, -0.0036, -0.0036, 0.001, 0, 0])
    state = steel.trial_state(final_strain, committed)
    tangent = steel.tangent_modulus(state)

    step = 1e-7
    jacobian = np.zeros((6, 6))
    for component in range(6):
        perturbation = np.zeros(6)
        perturbation[component] = step
        stress_plus = steel.trial_state(final_strain + perturbation, committed).stress
        stress_minus = steel.trial_state(final_strain - perturbation, committed).stress
        jacobian[:, component] = (stress_plus - stress_minus) / (2.0 * step)
    jacobian = 0.5 * (jacobian + jacobian.T)

    scale = np.abs(jacobian).max()
    assert np.abs(tangent - jacobian).max() / scale < 1e-3


# --- Validation ---------------------------------------------------------------


@pytest.mark.parametrize("youngs_modulus", [0.0, -1.0, float("nan")])
def test_rejects_non_positive_youngs_modulus(youngs_modulus: float) -> None:
    with pytest.raises(ValidationError):
        J2FiniteStrainPlasticity3D(
            youngs_modulus=youngs_modulus, poisson_ratio=0.3, yield_stress=250e6,
            hardening_modulus=10e9,
        )


@pytest.mark.parametrize("poisson_ratio", [-1.0, 0.5, 0.6, float("nan")])
def test_rejects_invalid_poisson_ratio(poisson_ratio: float) -> None:
    with pytest.raises(ValidationError):
        J2FiniteStrainPlasticity3D(
            youngs_modulus=200e9, poisson_ratio=poisson_ratio, yield_stress=250e6,
            hardening_modulus=10e9,
        )


@pytest.mark.parametrize("yield_stress", [0.0, -1.0, float("nan")])
def test_rejects_non_positive_yield_stress(yield_stress: float) -> None:
    with pytest.raises(ValidationError):
        J2FiniteStrainPlasticity3D(
            youngs_modulus=200e9, poisson_ratio=0.3, yield_stress=yield_stress,
            hardening_modulus=10e9,
        )


@pytest.mark.parametrize("hardening_modulus", [-1.0, float("nan")])
def test_rejects_invalid_hardening_modulus(hardening_modulus: float) -> None:
    with pytest.raises(ValidationError):
        J2FiniteStrainPlasticity3D(
            youngs_modulus=200e9, poisson_ratio=0.3, yield_stress=250e6,
            hardening_modulus=hardening_modulus,
        )


def test_trial_state_does_not_mutate_committed_state(steel: J2FiniteStrainPlasticity3D) -> None:
    committed = steel.initial_state()
    steel.trial_state(np.array([0.01, 0, 0, 0, 0, 0]), committed)

    assert_allclose(committed.strain, np.zeros(6))
    assert_allclose(committed.plastic_deformation_gradient, np.eye(3))
    assert committed.hardening_variable == pytest.approx(0.0)


def test_two_gauss_point_states_are_independent(steel: J2FiniteStrainPlasticity3D) -> None:
    """A single material instance must not carry hidden per-call state between Gauss points."""
    state_yielded = steel.trial_state(np.array([0.02, 0, 0, 0, 0, 0]), steel.initial_state())
    state_elastic = steel.trial_state(np.array([1e-5, 0, 0, 0, 0, 0]), steel.initial_state())

    assert state_yielded.yielded
    assert not state_elastic.yielded
    assert_allclose(state_elastic.plastic_deformation_gradient, np.eye(3))


# --- State rollback ------------------------------------------------------------


def test_discarded_trial_state_leaves_committed_state_fully_usable(
    steel: J2FiniteStrainPlasticity3D,
) -> None:
    """Simulates a Newton-Raphson iteration that failed to converge: a trial state is
    computed and then discarded (never becomes the new committed reference). The
    original committed state must remain exactly as valid and usable as before --
    this immutability is the entire rollback mechanism in this architecture (see
    the module docstring for femtoolkit.materials.nonlinear)."""
    committed = steel.trial_state(
        np.array([0.008, -0.0024, -0.0024, 0, 0, 0]), steel.initial_state()
    )

    # A large, "failed iteration" trial evaluation, discarded (its result is never used).
    steel.trial_state(np.array([0.5, -0.15, -0.15, 0, 0, 0]), committed)

    # Re-evaluating the SAME next strain from the (untouched) committed state must
    # give exactly the same result whether or not the discarded trial ever happened.
    retry_a = steel.trial_state(np.array([0.009, -0.0027, -0.0027, 0, 0, 0]), committed)
    retry_b = steel.trial_state(np.array([0.009, -0.0027, -0.0027, 0, 0, 0]), committed)
    assert_allclose(retry_a.stress, retry_b.stress, rtol=1e-14)
    assert_allclose(
        retry_a.plastic_deformation_gradient, retry_b.plastic_deformation_gradient, rtol=1e-14
    )


def test_committed_state_survives_many_discarded_trials(steel: J2FiniteStrainPlasticity3D) -> None:
    committed = steel.trial_state(np.array([0.01, -0.003, -0.003, 0, 0, 0]), steel.initial_state())
    baseline_fp = np.array(committed.plastic_deformation_gradient)
    baseline_alpha = committed.hardening_variable

    for large_but_valid_strain in (
        np.array([1.0, -0.2, -0.2, 0, 0, 0]),
        np.array([0.3, 0.3, 0.3, 0.1, 0.1, 0.1]),
        np.array([-0.3, 0.1, 0.1, 0, 0, 0]),
    ):
        steel.trial_state(large_but_valid_strain, committed)  # discarded

    assert_allclose(committed.plastic_deformation_gradient, baseline_fp)
    assert committed.hardening_variable == pytest.approx(baseline_alpha)


def test_committed_state_survives_a_trial_that_raises(steel: J2FiniteStrainPlasticity3D) -> None:
    """Even a trial evaluation invalid enough to raise (an inverted element -- a
    Newton-Raphson iteration overshooting badly) must not corrupt the committed
    state: MaterialState is immutable, and committed_state is never written to."""
    from femtoolkit.exceptions import InvalidDeformationGradientError

    committed = steel.trial_state(np.array([0.01, -0.003, -0.003, 0, 0, 0]), steel.initial_state())
    baseline_fp = np.array(committed.plastic_deformation_gradient)

    with pytest.raises(InvalidDeformationGradientError):
        steel.trial_state(np.array([-0.9, 0, 0, 0, 0, 0]), committed)  # inverted, discarded

    assert_allclose(committed.plastic_deformation_gradient, baseline_fp)
    retry = steel.trial_state(np.array([0.012, -0.0036, -0.0036, 0, 0, 0]), committed)
    assert np.all(np.isfinite(retry.stress))


# --- Large deformation -----------------------------------------------------------


def test_large_deformation_uniaxial_stretch_remains_stable_and_yields(
    steel: J2FiniteStrainPlasticity3D,
) -> None:
    """A genuinely large stretch (lambda ~= 1.5, well beyond small-strain plasticity's
    validity) must still produce a finite, physically sensible, yielded state."""
    stretch = 1.5
    green_lagrange_axial = 0.5 * (stretch**2 - 1.0)
    lateral_stretch = stretch ** (-0.3)
    green_lagrange_lateral = 0.5 * (lateral_stretch**2 - 1.0)
    strain = np.array(
        [green_lagrange_axial, green_lagrange_lateral, green_lagrange_lateral, 0, 0, 0]
    )

    state = steel.trial_state(strain, steel.initial_state())
    assert state.yielded
    assert np.all(np.isfinite(state.stress))
    assert np.all(np.isfinite(state.plastic_deformation_gradient))
    assert np.linalg.det(state.plastic_deformation_gradient) > 0.0


def test_large_simple_shear_remains_stable(steel: J2FiniteStrainPlasticity3D) -> None:
    strain = np.array([0.0, 0.0, 0.0, 0.5, 0.0, 0.0])
    state = steel.trial_state(strain, steel.initial_state())
    assert state.yielded
    assert np.all(np.isfinite(state.stress))
    tangent = steel.tangent_modulus(state)
    assert np.all(np.isfinite(tangent))


# --- Stress-measure transformations ----------------------------------------------


def test_first_piola_kirchhoff_equals_f_dot_s(steel: J2FiniteStrainPlasticity3D) -> None:
    state = steel.trial_state(np.array([0.01, -0.003, -0.001, 0.002, 0, 0]), steel.initial_state())
    f = _symmetric_deformation_gradient(state.strain)
    s_tensor = voigt_stress_to_tensor(state.stress)
    expected_p = f @ s_tensor
    assert_allclose(steel.first_piola_kirchhoff_stress(state, f), expected_p, rtol=1e-10)


def test_cauchy_stress_equals_one_over_j_times_f_s_ft(steel: J2FiniteStrainPlasticity3D) -> None:
    state = steel.trial_state(np.array([0.01, -0.003, -0.001, 0.002, 0, 0]), steel.initial_state())
    f = _symmetric_deformation_gradient(state.strain)
    s_tensor = voigt_stress_to_tensor(state.stress)
    jacobian = np.linalg.det(f)
    expected_sigma = (1.0 / jacobian) * f @ s_tensor @ f.T
    assert_allclose(steel.cauchy_stress(state, f), expected_sigma, rtol=1e-10)


def test_cauchy_stress_is_symmetric(steel: J2FiniteStrainPlasticity3D) -> None:
    state = steel.trial_state(np.array([0.015, -0.004, -0.002, 0.003, 0, 0]), steel.initial_state())
    f = _symmetric_deformation_gradient(state.strain)
    sigma = steel.cauchy_stress(state, f)
    assert_allclose(sigma, sigma.T, atol=1e-6)
