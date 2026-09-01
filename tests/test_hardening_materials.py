"""Tests for Version 14 hardening plasticity materials."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from femtoolkit.exceptions import (
    ConstitutiveUpdateError,
    InvalidMaterialStateError,
    UnsupportedLoadingPathError,
    ValidationError,
)
from femtoolkit.materials import (
    BilinearIsotropicHardeningMaterial1D,
    BilinearKinematicHardeningMaterial1D,
    DecoupledIsotropicHardeningAdapter2D,
    ElasticPerfectlyPlasticMaterial1D,
    MaterialState,
    MultilinearIsotropicHardeningMaterial1D,
)

YOUNGS_MODULUS = 200e9
YIELD_STRESS = 250e6
HARDENING_MODULUS = 20e9


# --- MaterialState.elastic_strain (shared, Version 14 addition) ---


def test_material_state_elastic_strain_scalar() -> None:
    state = MaterialState(strain=0.005, stress=1.0, plastic_strain=0.002, yielded=True)

    assert state.elastic_strain == pytest.approx(0.003)


def test_material_state_elastic_strain_array() -> None:
    state = MaterialState(
        strain=np.array([0.005, 0.001, 0.0]),
        stress=np.zeros(3),
        plastic_strain=np.array([0.002, 0.0, 0.0]),
        yielded=np.array([True, False, False]),
    )

    assert_allclose(state.elastic_strain, np.array([0.003, 0.001, 0.0]))


def test_material_state_hardening_variable_and_back_stress_default_to_zero() -> None:
    state = MaterialState.zero(0.0)

    assert state.hardening_variable == 0.0
    assert state.back_stress == 0.0


# --- BilinearIsotropicHardeningMaterial1D ---


@pytest.fixture
def isotropic_material() -> BilinearIsotropicHardeningMaterial1D:
    return BilinearIsotropicHardeningMaterial1D(
        youngs_modulus=YOUNGS_MODULUS,
        yield_stress=YIELD_STRESS,
        hardening_modulus=HARDENING_MODULUS,
    )


def test_isotropic_elastic_region(isotropic_material: BilinearIsotropicHardeningMaterial1D) -> None:
    strain = YIELD_STRESS / YOUNGS_MODULUS * 0.5
    state = isotropic_material.trial_state(strain, isotropic_material.initial_state())

    assert state.stress == pytest.approx(YOUNGS_MODULUS * strain)
    assert state.yielded is False
    assert state.plastic_strain == pytest.approx(0.0)
    assert state.hardening_variable == pytest.approx(0.0)
    assert isotropic_material.tangent_modulus(state) == YOUNGS_MODULUS


def test_isotropic_first_yield(isotropic_material: BilinearIsotropicHardeningMaterial1D) -> None:
    yield_strain = YIELD_STRESS / YOUNGS_MODULUS
    state = isotropic_material.trial_state(yield_strain, isotropic_material.initial_state())

    assert state.stress == pytest.approx(YIELD_STRESS)
    assert state.yielded is False  # exactly at the boundary: not yet past it.


def test_isotropic_plastic_correction_and_hardening(
    isotropic_material: BilinearIsotropicHardeningMaterial1D,
) -> None:
    yield_strain = YIELD_STRESS / YOUNGS_MODULUS
    state = isotropic_material.trial_state(2.0 * yield_strain, isotropic_material.initial_state())

    # Closed-form 1D return map: sigma_trial = E*2*eps_y = 2*sigma_y0.
    trial_stress = 2.0 * YIELD_STRESS
    expected_delta_gamma = (trial_stress - YIELD_STRESS) / (YOUNGS_MODULUS + HARDENING_MODULUS)
    expected_stress = trial_stress - YOUNGS_MODULUS * expected_delta_gamma
    assert state.stress == pytest.approx(expected_stress)
    assert state.yielded is True
    assert state.plastic_strain == pytest.approx(expected_delta_gamma)
    assert state.hardening_variable == pytest.approx(expected_delta_gamma)


def test_isotropic_yield_surface_expands_with_continued_loading(
    isotropic_material: BilinearIsotropicHardeningMaterial1D,
) -> None:
    yield_strain = YIELD_STRESS / YOUNGS_MODULUS
    state = isotropic_material.initial_state()
    stresses = []
    for multiple in (2.0, 3.0, 4.0):
        state = isotropic_material.trial_state(multiple * yield_strain, state)
        stresses.append(state.stress)

    # Continued monotonic loading keeps increasing stress (hardening),
    # unlike perfectly-plastic behavior where it would plateau.
    assert stresses[0] < stresses[1] < stresses[2]


def test_isotropic_tangent_modulus_reduced_when_yielded(
    isotropic_material: BilinearIsotropicHardeningMaterial1D,
) -> None:
    state = isotropic_material.trial_state(0.01, isotropic_material.initial_state())

    tangent = isotropic_material.tangent_modulus(state)
    expected = YOUNGS_MODULUS * HARDENING_MODULUS / (YOUNGS_MODULUS + HARDENING_MODULUS)
    assert tangent == pytest.approx(expected)
    assert 0.0 < tangent < YOUNGS_MODULUS


def test_isotropic_trial_state_does_not_mutate_committed_state(
    isotropic_material: BilinearIsotropicHardeningMaterial1D,
) -> None:
    committed = isotropic_material.initial_state()
    isotropic_material.trial_state(0.01, committed)

    assert committed.strain == 0.0
    assert committed.plastic_strain == 0.0
    assert committed.hardening_variable == 0.0
    assert committed.yielded is False


def test_isotropic_matches_perfectly_plastic_when_hardening_modulus_is_zero() -> None:
    """H = 0 must reproduce ElasticPerfectlyPlasticMaterial1D's stress-strain response."""
    hardening_material = BilinearIsotropicHardeningMaterial1D(
        youngs_modulus=YOUNGS_MODULUS, yield_stress=YIELD_STRESS, hardening_modulus=0.0
    )
    perfectly_plastic = ElasticPerfectlyPlasticMaterial1D(
        youngs_modulus=YOUNGS_MODULUS, yield_stress=YIELD_STRESS
    )

    yield_strain = YIELD_STRESS / YOUNGS_MODULUS
    hardening_state = hardening_material.initial_state()
    plastic_state = perfectly_plastic.initial_state()

    strains = (
        0.5 * yield_strain,
        yield_strain,
        2 * yield_strain,
        5 * yield_strain,
        -3 * yield_strain,
    )
    for strain in strains:
        hardening_state = hardening_material.trial_state(strain, hardening_state)
        plastic_state = perfectly_plastic.trial_state(strain, plastic_state)
        assert hardening_state.stress == pytest.approx(plastic_state.stress, rel=1e-9)
        assert hardening_state.plastic_strain == pytest.approx(
            plastic_state.plastic_strain, rel=1e-9
        )


def test_isotropic_rejects_invalid_parameters() -> None:
    with pytest.raises(ValidationError):
        BilinearIsotropicHardeningMaterial1D(
            youngs_modulus=0.0, yield_stress=YIELD_STRESS, hardening_modulus=HARDENING_MODULUS
        )
    with pytest.raises(ValidationError):
        BilinearIsotropicHardeningMaterial1D(
            youngs_modulus=YOUNGS_MODULUS, yield_stress=-1.0, hardening_modulus=HARDENING_MODULUS
        )
    with pytest.raises(ValidationError):
        BilinearIsotropicHardeningMaterial1D(
            youngs_modulus=YOUNGS_MODULUS, yield_stress=YIELD_STRESS, hardening_modulus=-1.0
        )


def test_isotropic_rejects_non_finite_strain(
    isotropic_material: BilinearIsotropicHardeningMaterial1D,
) -> None:
    with pytest.raises(ConstitutiveUpdateError):
        isotropic_material.trial_state(float("nan"), isotropic_material.initial_state())


def test_isotropic_rejects_non_scalar_committed_state(
    isotropic_material: BilinearIsotropicHardeningMaterial1D,
) -> None:
    vector_state = MaterialState.zero(np.zeros(3))
    with pytest.raises(InvalidMaterialStateError):
        isotropic_material.trial_state(0.001, vector_state)


# --- BilinearKinematicHardeningMaterial1D ---


@pytest.fixture
def kinematic_material() -> BilinearKinematicHardeningMaterial1D:
    return BilinearKinematicHardeningMaterial1D(
        youngs_modulus=YOUNGS_MODULUS,
        yield_stress=YIELD_STRESS,
        hardening_modulus=HARDENING_MODULUS,
    )


def test_kinematic_initial_yield(kinematic_material: BilinearKinematicHardeningMaterial1D) -> None:
    yield_strain = YIELD_STRESS / YOUNGS_MODULUS
    state = kinematic_material.trial_state(1.5 * yield_strain, kinematic_material.initial_state())

    assert state.yielded is True
    assert state.back_stress != 0.0


def test_kinematic_yield_surface_size_stays_fixed(
    kinematic_material: BilinearKinematicHardeningMaterial1D,
) -> None:
    """Unlike isotropic hardening, the (sigma - X) yield width stays sigma_y0 forever."""
    state = kinematic_material.trial_state(0.01, kinematic_material.initial_state())

    assert abs(state.stress - state.back_stress) == pytest.approx(YIELD_STRESS, rel=1e-9)


def test_kinematic_unloading_is_elastic(
    kinematic_material: BilinearKinematicHardeningMaterial1D,
) -> None:
    yield_strain = YIELD_STRESS / YOUNGS_MODULUS
    yielded_state = kinematic_material.trial_state(
        3.0 * yield_strain, kinematic_material.initial_state()
    )

    unloaded_strain = yielded_state.strain - 0.1 * yield_strain
    unloaded_state = kinematic_material.trial_state(unloaded_strain, yielded_state)

    expected_stress = YOUNGS_MODULUS * (unloaded_strain - yielded_state.plastic_strain)
    assert unloaded_state.stress == pytest.approx(expected_stress)
    assert unloaded_state.yielded is False
    assert unloaded_state.plastic_strain == pytest.approx(yielded_state.plastic_strain)
    assert unloaded_state.back_stress == pytest.approx(yielded_state.back_stress)


def test_kinematic_back_stress_evolution(
    kinematic_material: BilinearKinematicHardeningMaterial1D,
) -> None:
    yield_strain = YIELD_STRESS / YOUNGS_MODULUS
    state = kinematic_material.trial_state(
        2.0 * yield_strain, kinematic_material.initial_state()
    )

    # X = H * plastic_strain (linear back-stress evolution) for monotonic loading from zero.
    assert state.back_stress == pytest.approx(HARDENING_MODULUS * state.plastic_strain)


def test_kinematic_bauschinger_effect(
    kinematic_material: BilinearKinematicHardeningMaterial1D,
) -> None:
    """After tensile yielding, the reverse (compressive) yield stress must be
    reached earlier in magnitude than the virgin yield stress -- the
    Bauschinger effect.
    """
    yield_strain = YIELD_STRESS / YOUNGS_MODULUS
    yielded_state = kinematic_material.trial_state(
        3.0 * yield_strain, kinematic_material.initial_state()
    )

    # The reverse-yield stress is (X - sigma_y0); its magnitude must be smaller
    # than the virgin yield stress because X > 0 after tensile plastic flow.
    reverse_yield_stress = yielded_state.back_stress - YIELD_STRESS
    assert abs(reverse_yield_stress) < YIELD_STRESS


def test_kinematic_trial_state_does_not_mutate_committed_state(
    kinematic_material: BilinearKinematicHardeningMaterial1D,
) -> None:
    committed = kinematic_material.initial_state()
    kinematic_material.trial_state(0.01, committed)

    assert committed.strain == 0.0
    assert committed.plastic_strain == 0.0
    assert committed.back_stress == 0.0


def test_kinematic_matches_perfectly_plastic_when_hardening_modulus_is_zero() -> None:
    hardening_material = BilinearKinematicHardeningMaterial1D(
        youngs_modulus=YOUNGS_MODULUS, yield_stress=YIELD_STRESS, hardening_modulus=0.0
    )
    perfectly_plastic = ElasticPerfectlyPlasticMaterial1D(
        youngs_modulus=YOUNGS_MODULUS, yield_stress=YIELD_STRESS
    )

    yield_strain = YIELD_STRESS / YOUNGS_MODULUS
    hardening_state = hardening_material.initial_state()
    plastic_state = perfectly_plastic.initial_state()

    for strain in (0.5 * yield_strain, yield_strain, 2 * yield_strain):
        hardening_state = hardening_material.trial_state(strain, hardening_state)
        plastic_state = perfectly_plastic.trial_state(strain, plastic_state)
        assert hardening_state.stress == pytest.approx(plastic_state.stress, rel=1e-9)


def test_kinematic_rejects_invalid_parameters() -> None:
    with pytest.raises(ValidationError):
        BilinearKinematicHardeningMaterial1D(
            youngs_modulus=YOUNGS_MODULUS, yield_stress=YIELD_STRESS, hardening_modulus=-1.0
        )


# --- MultilinearIsotropicHardeningMaterial1D ---


@pytest.fixture
def multilinear_material() -> MultilinearIsotropicHardeningMaterial1D:
    return MultilinearIsotropicHardeningMaterial1D(
        strain_points=(0.0, 0.001, 0.005, 0.02),
        stress_points=(0.0, 200e6, 250e6, 300e6),
    )


def test_multilinear_derives_youngs_modulus_from_first_segment(
    multilinear_material: MultilinearIsotropicHardeningMaterial1D,
) -> None:
    assert multilinear_material.youngs_modulus == pytest.approx(200e9)


def test_multilinear_elastic_segment_interpolation(
    multilinear_material: MultilinearIsotropicHardeningMaterial1D,
) -> None:
    state = multilinear_material.trial_state(0.0005, multilinear_material.initial_state())

    assert state.stress == pytest.approx(100e6)
    assert state.yielded is False


def test_multilinear_segment_transition(
    multilinear_material: MultilinearIsotropicHardeningMaterial1D,
) -> None:
    state = multilinear_material.trial_state(0.003, multilinear_material.initial_state())

    # Segment 2 (0.001 -> 0.005): slope = (250-200)e6 / (0.005-0.001) = 12.5e9
    expected_stress = 200e6 + 12.5e9 * (0.003 - 0.001)
    assert state.stress == pytest.approx(expected_stress)
    assert state.yielded is True
    assert multilinear_material.tangent_modulus(state) == pytest.approx(12.5e9)


def test_multilinear_final_hardening_region(
    multilinear_material: MultilinearIsotropicHardeningMaterial1D,
) -> None:
    state = multilinear_material.trial_state(0.01, multilinear_material.initial_state())

    # Segment 3 (0.005 -> 0.02): slope = (300-250)e6 / (0.02-0.005) = 10/3 GPa
    expected_slope = (300e6 - 250e6) / (0.02 - 0.005)
    expected_stress = 250e6 + expected_slope * (0.01 - 0.005)
    assert state.stress == pytest.approx(expected_stress)
    assert multilinear_material.tangent_modulus(state) == pytest.approx(expected_slope)


def test_multilinear_extrapolates_beyond_final_point(
    multilinear_material: MultilinearIsotropicHardeningMaterial1D,
) -> None:
    state = multilinear_material.trial_state(0.025, multilinear_material.initial_state())

    expected_slope = (300e6 - 250e6) / (0.02 - 0.005)
    expected_stress = 300e6 + expected_slope * (0.025 - 0.02)
    assert state.stress == pytest.approx(expected_stress)


def test_multilinear_symmetric_in_compression(
    multilinear_material: MultilinearIsotropicHardeningMaterial1D,
) -> None:
    state = multilinear_material.trial_state(-0.003, multilinear_material.initial_state())

    expected_stress = -(200e6 + 12.5e9 * (0.003 - 0.001))
    assert state.stress == pytest.approx(expected_stress)


def test_multilinear_rejects_unloading() -> None:
    material = MultilinearIsotropicHardeningMaterial1D(
        strain_points=(0.0, 0.001, 0.005, 0.02),
        stress_points=(0.0, 200e6, 250e6, 300e6),
    )
    state = material.trial_state(0.01, material.initial_state())

    with pytest.raises(UnsupportedLoadingPathError):
        material.trial_state(0.005, state)


def test_multilinear_rejects_direction_reversal() -> None:
    material = MultilinearIsotropicHardeningMaterial1D(
        strain_points=(0.0, 0.001, 0.005, 0.02),
        stress_points=(0.0, 200e6, 250e6, 300e6),
    )
    state = material.trial_state(0.01, material.initial_state())

    with pytest.raises(UnsupportedLoadingPathError):
        material.trial_state(-0.001, state)


def test_multilinear_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValidationError):
        MultilinearIsotropicHardeningMaterial1D(strain_points=(0.0, 0.001), stress_points=(0.0,))


def test_multilinear_rejects_too_few_points() -> None:
    with pytest.raises(ValidationError):
        MultilinearIsotropicHardeningMaterial1D(strain_points=(0.0,), stress_points=(0.0,))


def test_multilinear_rejects_nonzero_first_point() -> None:
    with pytest.raises(ValidationError):
        MultilinearIsotropicHardeningMaterial1D(
            strain_points=(0.0001, 0.001), stress_points=(0.0, 200e6)
        )


def test_multilinear_rejects_non_increasing_strain() -> None:
    with pytest.raises(ValidationError):
        MultilinearIsotropicHardeningMaterial1D(
            strain_points=(0.0, 0.001, 0.001), stress_points=(0.0, 200e6, 250e6)
        )


def test_multilinear_rejects_decreasing_stress() -> None:
    with pytest.raises(ValidationError):
        MultilinearIsotropicHardeningMaterial1D(
            strain_points=(0.0, 0.001, 0.005), stress_points=(0.0, 200e6, 150e6)
        )


def test_multilinear_rejects_non_finite_values() -> None:
    with pytest.raises(ValidationError):
        MultilinearIsotropicHardeningMaterial1D(
            strain_points=(0.0, float("inf")), stress_points=(0.0, 200e6)
        )


# --- DecoupledIsotropicHardeningAdapter2D ---


@pytest.fixture
def decoupled_material() -> DecoupledIsotropicHardeningAdapter2D:
    return DecoupledIsotropicHardeningAdapter2D(
        youngs_modulus=YOUNGS_MODULUS,
        yield_stress=YIELD_STRESS,
        hardening_modulus=HARDENING_MODULUS,
    )


def test_decoupled_initial_state_is_length_three(
    decoupled_material: DecoupledIsotropicHardeningAdapter2D,
) -> None:
    state = decoupled_material.initial_state()

    assert np.asarray(state.strain).shape == (3,)
    assert np.asarray(state.plastic_strain).shape == (3,)


def test_decoupled_components_yield_independently(
    decoupled_material: DecoupledIsotropicHardeningAdapter2D,
) -> None:
    yield_strain = YIELD_STRESS / YOUNGS_MODULUS
    strain = np.array([3.0 * yield_strain, 0.1 * yield_strain, 0.0])
    state = decoupled_material.trial_state(strain, decoupled_material.initial_state())

    assert_allclose(state.yielded, np.array([True, False, False]))
    assert state.stress[0] < YOUNGS_MODULUS * strain[0]  # returned to the yield surface
    assert state.stress[1] == pytest.approx(YOUNGS_MODULUS * strain[1])  # still elastic


def test_decoupled_tangent_is_diagonal_and_component_specific(
    decoupled_material: DecoupledIsotropicHardeningAdapter2D,
) -> None:
    yield_strain = YIELD_STRESS / YOUNGS_MODULUS
    strain = np.array([3.0 * yield_strain, 0.1 * yield_strain, 0.0])
    state = decoupled_material.trial_state(strain, decoupled_material.initial_state())

    tangent = decoupled_material.tangent_modulus(state)
    assert tangent.shape == (3, 3)
    assert_allclose(tangent, np.diag(np.diag(tangent)))  # purely diagonal, no coupling
    expected_plastic_tangent = (
        YOUNGS_MODULUS * HARDENING_MODULUS / (YOUNGS_MODULUS + HARDENING_MODULUS)
    )
    assert tangent[0, 0] == pytest.approx(expected_plastic_tangent)
    assert tangent[1, 1] == pytest.approx(YOUNGS_MODULUS)
    assert tangent[2, 2] == pytest.approx(YOUNGS_MODULUS)


def test_decoupled_trial_state_does_not_mutate_committed_state(
    decoupled_material: DecoupledIsotropicHardeningAdapter2D,
) -> None:
    committed = decoupled_material.initial_state()
    decoupled_material.trial_state(np.array([0.01, 0.0, 0.0]), committed)

    assert_allclose(committed.plastic_strain, np.zeros(3))


def test_decoupled_rejects_scalar_committed_state(
    decoupled_material: DecoupledIsotropicHardeningAdapter2D,
) -> None:
    with pytest.raises(InvalidMaterialStateError):
        decoupled_material.trial_state(np.array([0.01, 0.0, 0.0]), MaterialState.zero(0.0))


def test_decoupled_rejects_non_finite_strain(
    decoupled_material: DecoupledIsotropicHardeningAdapter2D,
) -> None:
    with pytest.raises(ConstitutiveUpdateError):
        decoupled_material.trial_state(
            np.array([float("nan"), 0.0, 0.0]), decoupled_material.initial_state()
        )
