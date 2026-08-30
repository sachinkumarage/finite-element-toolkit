"""Validation: elastic-perfectly-plastic 1D material vs. its closed-form
bilinear response (Section 24).

Parameters match the spec's example: ``E = 200 GPa``, ``sigma_y = 250 MPa``.
"""

import numpy as np
import pytest

from femtoolkit.materials import ElasticPerfectlyPlasticMaterial1D

YOUNGS_MODULUS = 200e9
YIELD_STRESS = 250e6
YIELD_STRAIN = YIELD_STRESS / YOUNGS_MODULUS


@pytest.fixture
def material() -> ElasticPerfectlyPlasticMaterial1D:
    return ElasticPerfectlyPlasticMaterial1D(
        youngs_modulus=YOUNGS_MODULUS, yield_stress=YIELD_STRESS
    )


def test_elastic_region_follows_hookes_law(material: ElasticPerfectlyPlasticMaterial1D) -> None:
    strains = np.linspace(0.0, YIELD_STRAIN, 10, endpoint=False)
    committed = material.initial_state()

    for strain in strains:
        state = material.trial_state(float(strain), committed)
        assert state.stress == pytest.approx(YOUNGS_MODULUS * strain)
        assert state.yielded is False


def test_yield_point_matches_specified_yield_stress(
    material: ElasticPerfectlyPlasticMaterial1D,
) -> None:
    state = material.trial_state(YIELD_STRAIN, material.initial_state())

    assert state.stress == pytest.approx(YIELD_STRESS, rel=1e-9)


def test_plastic_region_stress_plateaus_at_yield_stress(
    material: ElasticPerfectlyPlasticMaterial1D,
) -> None:
    committed = material.initial_state()

    for multiple in (1.5, 2.0, 5.0, 10.0):
        state = material.trial_state(multiple * YIELD_STRAIN, committed)
        assert state.stress == pytest.approx(YIELD_STRESS, rel=1e-9)
        assert state.yielded is True


def test_unloading_from_plastic_state_follows_elastic_slope(
    material: ElasticPerfectlyPlasticMaterial1D,
) -> None:
    """After yielding, an incremental strain decrease must trace a line of
    slope E back from the plastic point -- not immediately drop to zero or
    re-yield in compression -- until it crosses back through zero stress
    at the accumulated plastic strain.
    """
    yielded_state = material.trial_state(4.0 * YIELD_STRAIN, material.initial_state())
    plastic_strain = yielded_state.plastic_strain

    unload_strains = np.linspace(yielded_state.strain, plastic_strain, 10)
    for strain in unload_strains:
        state = material.trial_state(float(strain), yielded_state)
        expected_stress = YOUNGS_MODULUS * (strain - plastic_strain)
        assert state.stress == pytest.approx(expected_stress, abs=1.0)
        assert abs(state.stress) <= YIELD_STRESS + 1e-3


def test_reloading_back_to_yield_does_not_double_yield(
    material: ElasticPerfectlyPlasticMaterial1D,
) -> None:
    """Unload to zero stress then reload back to the original plastic
    strain: stress must return to exactly the yield stress again, not
    exceed it.
    """
    yielded_state = material.trial_state(4.0 * YIELD_STRAIN, material.initial_state())
    zero_stress_state = material.trial_state(yielded_state.plastic_strain, yielded_state)
    assert zero_stress_state.stress == pytest.approx(0.0, abs=1e-6)

    reloaded_state = material.trial_state(yielded_state.strain, zero_stress_state)
    assert reloaded_state.stress == pytest.approx(YIELD_STRESS, rel=1e-9)


def test_model_limitation_no_hardening_stress_never_exceeds_yield(
    material: ElasticPerfectlyPlasticMaterial1D,
) -> None:
    """Documents the Version 13 model limitation: however far strain is
    pushed past yield, stress never exceeds sigma_y -- there is no
    hardening branch (reserved for a future version).
    """
    committed = material.initial_state()
    state = material.trial_state(1000.0 * YIELD_STRAIN, committed)

    assert abs(state.stress) == pytest.approx(YIELD_STRESS, rel=1e-9)
