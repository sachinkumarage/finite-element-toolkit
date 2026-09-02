"""Tests for femtoolkit.materials.j2_plasticity.J2Plasticity3D."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from femtoolkit.continuum.stress import von_mises_3d
from femtoolkit.continuum.tensor import voigt_stress_to_tensor
from femtoolkit.exceptions import InvalidMaterialStateError, ValidationError
from femtoolkit.materials.j2_plasticity import J2Plasticity3D
from femtoolkit.materials.nonlinear import MaterialState


@pytest.fixture
def steel() -> J2Plasticity3D:
    return J2Plasticity3D(
        youngs_modulus=200e9, poisson_ratio=0.3, yield_stress=250e6, hardening_modulus=10e9
    )


def test_initial_state_is_zero(steel: J2Plasticity3D) -> None:
    state = steel.initial_state()
    assert_allclose(state.strain, np.zeros(6))
    assert_allclose(state.stress, np.zeros(6))
    assert_allclose(state.plastic_strain, np.zeros(6))
    assert not state.yielded
    assert state.hardening_variable == pytest.approx(0.0)
    assert state.plastic_multiplier == pytest.approx(0.0)


def test_elastic_loading_matches_hookes_law(steel: J2Plasticity3D) -> None:
    strain = np.array([1e-4, -2e-5, 3e-5, 5e-5, -1e-5, 2e-5])
    state = steel.trial_state(strain, steel.initial_state())

    assert not state.yielded
    assert_allclose(state.stress, steel.constitutive_matrix @ strain, rtol=1e-10)
    assert_allclose(state.plastic_strain, np.zeros(6))


def test_hydrostatic_loading_does_not_increase_von_mises(steel: J2Plasticity3D) -> None:
    """Large hydrostatic strain must never trigger yielding (module docstring's physical claim)."""
    strain = np.array([1e-2, 1e-2, 1e-2, 0.0, 0.0, 0.0])
    state = steel.trial_state(strain, steel.initial_state())

    assert not state.yielded
    assert von_mises_3d(*state.stress) == pytest.approx(0.0, abs=1.0)


def test_pure_shear_gives_expected_von_mises(steel: J2Plasticity3D) -> None:
    tau = 100e6  # below yield
    gamma = tau / steel.shear_modulus
    strain = np.array([0.0, 0.0, 0.0, gamma, 0.0, 0.0])
    state = steel.trial_state(strain, steel.initial_state())

    assert not state.yielded
    assert von_mises_3d(*state.stress) == pytest.approx(np.sqrt(3.0) * tau, rel=1e-6)


def test_uniaxial_loading_von_mises_equals_stress_magnitude(steel: J2Plasticity3D) -> None:
    axial_strain = 1e-4
    lateral_strain = -steel.poisson_ratio * axial_strain
    strain = np.array([axial_strain, lateral_strain, lateral_strain, 0.0, 0.0, 0.0])
    state = steel.trial_state(strain, steel.initial_state())

    assert von_mises_3d(*state.stress) == pytest.approx(abs(state.stress[0]), rel=1e-8)


def test_yielding_lands_exactly_on_the_yield_surface(steel: J2Plasticity3D) -> None:
    strain = np.array([0.01, 0.0, 0.0, 0.0, 0.0, 0.0])
    state = steel.trial_state(strain, steel.initial_state())

    assert state.yielded
    current_yield = steel.yield_stress + steel.hardening_modulus * state.hardening_variable
    von_mises = von_mises_3d(*state.stress)
    assert von_mises == pytest.approx(current_yield, rel=1e-6)


def test_plastic_loading_increases_equivalent_plastic_strain(steel: J2Plasticity3D) -> None:
    state_a = steel.trial_state(np.array([0.01, 0, 0, 0, 0, 0]), steel.initial_state())
    state_b = steel.trial_state(np.array([0.015, 0, 0, 0, 0, 0]), state_a)

    assert state_a.hardening_variable > 0.0
    assert state_b.hardening_variable > state_a.hardening_variable


def test_hardening_consistency_through_incremental_loading(steel: J2Plasticity3D) -> None:
    """sigma_y = sigma_y0 + H*alpha must hold after every plastic increment."""
    state = steel.initial_state()
    for axial_strain in (0.005, 0.008, 0.012, 0.02):
        state = steel.trial_state(np.array([axial_strain, 0, 0, 0, 0, 0]), state)
        if state.yielded:
            current_yield = steel.yield_stress + steel.hardening_modulus * state.hardening_variable
            assert von_mises_3d(*state.stress) == pytest.approx(current_yield, rel=1e-6)


def test_perfectly_plastic_special_case_caps_von_mises(steel: J2Plasticity3D) -> None:
    perfectly_plastic = J2Plasticity3D(
        youngs_modulus=200e9, poisson_ratio=0.3, yield_stress=250e6, hardening_modulus=0.0
    )
    state_a = perfectly_plastic.trial_state(
        np.array([0.01, 0, 0, 0, 0, 0]), perfectly_plastic.initial_state()
    )
    state_b = perfectly_plastic.trial_state(np.array([0.02, 0, 0, 0, 0, 0]), state_a)

    assert von_mises_3d(*state_a.stress) == pytest.approx(250e6, rel=1e-6)
    assert von_mises_3d(*state_b.stress) == pytest.approx(250e6, rel=1e-6)


def test_trial_state_does_not_mutate_committed_state(steel: J2Plasticity3D) -> None:
    committed = steel.initial_state()
    steel.trial_state(np.array([0.01, 0, 0, 0, 0, 0]), committed)

    assert_allclose(committed.strain, np.zeros(6))
    assert_allclose(committed.plastic_strain, np.zeros(6))
    assert committed.hardening_variable == pytest.approx(0.0)


def test_two_gauss_point_states_are_independent(steel: J2Plasticity3D) -> None:
    """A single material instance must not carry hidden per-call state between Gauss points."""
    state_yielded = steel.trial_state(np.array([0.02, 0, 0, 0, 0, 0]), steel.initial_state())
    state_elastic = steel.trial_state(np.array([1e-5, 0, 0, 0, 0, 0]), steel.initial_state())

    assert state_yielded.yielded
    assert not state_elastic.yielded
    assert_allclose(state_elastic.plastic_strain, np.zeros(6))


# --- Tangent modulus ---------------------------------------------------


def test_elastic_tangent_equals_constitutive_matrix(steel: J2Plasticity3D) -> None:
    state = steel.trial_state(np.array([1e-5, 0, 0, 0, 0, 0]), steel.initial_state())
    tangent = steel.tangent_modulus(state)
    assert_allclose(tangent, steel.constitutive_matrix)


def test_plastic_tangent_is_symmetric_and_positive_definite(steel: J2Plasticity3D) -> None:
    state = steel.trial_state(np.array([0.01, 0, 0, 0, 0, 0]), steel.initial_state())
    tangent = steel.tangent_modulus(state)

    assert_allclose(tangent, tangent.T, atol=1.0)
    eigenvalues = np.linalg.eigvalsh(tangent)
    assert np.all(eigenvalues > 0)


def test_plastic_tangent_is_softer_than_elastic(steel: J2Plasticity3D) -> None:
    state = steel.trial_state(np.array([0.01, 0, 0, 0, 0, 0]), steel.initial_state())
    tangent = steel.tangent_modulus(state)
    # The axial (11) entry of the plastic tangent must be smaller than the elastic one.
    assert tangent[0, 0] < steel.constitutive_matrix[0, 0]


def test_return_mapping_radial_return_theorem_direction_preserved(steel: J2Plasticity3D) -> None:
    """The deviatoric stress direction must be identical before and after the plastic correction."""
    strain = np.array([0.01, -0.003, -0.001, 0.002, 0.0, 0.0])
    trial_stress = steel.constitutive_matrix @ strain
    trial_tensor = voigt_stress_to_tensor(trial_stress)
    from femtoolkit.continuum.tensor import deviatoric_stress

    trial_dev = deviatoric_stress(trial_tensor)
    trial_direction = trial_dev / np.linalg.norm(trial_dev)

    state = steel.trial_state(strain, steel.initial_state())
    assert state.yielded
    final_dev = deviatoric_stress(voigt_stress_to_tensor(state.stress))
    final_direction = final_dev / np.linalg.norm(final_dev)

    assert_allclose(final_direction, trial_direction, atol=1e-8)


# --- Validation ----------------------------------------------------------


def test_rejects_2d_committed_state(steel: J2Plasticity3D) -> None:
    bad_state = MaterialState(
        strain=np.zeros(3), stress=np.zeros(3), plastic_strain=np.zeros(3), yielded=False
    )
    with pytest.raises(InvalidMaterialStateError):
        steel.trial_state(np.zeros(6), bad_state)


@pytest.mark.parametrize("hardening_modulus", [-1.0, float("nan")])
def test_rejects_invalid_hardening_modulus(hardening_modulus: float) -> None:
    with pytest.raises(ValidationError):
        J2Plasticity3D(
            youngs_modulus=200e9,
            poisson_ratio=0.3,
            yield_stress=250e6,
            hardening_modulus=hardening_modulus,
        )


@pytest.mark.parametrize("yield_stress", [0.0, -1.0])
def test_rejects_invalid_yield_stress(yield_stress: float) -> None:
    with pytest.raises(ValidationError):
        J2Plasticity3D(
            youngs_modulus=200e9,
            poisson_ratio=0.3,
            yield_stress=yield_stress,
            hardening_modulus=1e9,
        )
