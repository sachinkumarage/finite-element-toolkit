"""Tests for the nonlinear material interface, MaterialState, and the two concrete materials."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from femtoolkit.exceptions import ValidationError
from femtoolkit.materials import (
    ElasticMaterialAdapter,
    ElasticPerfectlyPlasticMaterial1D,
    LinearElastic2D,
    Material,
    MaterialState,
)

YOUNGS_MODULUS = 200e9
YIELD_STRESS = 250e6


# --- MaterialState ---


def test_material_state_zero_scalar() -> None:
    state = MaterialState.zero(0.0)

    assert state.strain == 0.0
    assert state.stress == 0.0
    assert state.plastic_strain == 0.0
    assert state.yielded is False


def test_material_state_zero_array() -> None:
    state = MaterialState.zero(np.zeros(3))

    assert_allclose(state.strain, np.zeros(3))
    assert_allclose(state.stress, np.zeros(3))
    assert_allclose(state.plastic_strain, np.zeros(3))


def test_material_state_is_frozen() -> None:
    state = MaterialState.zero(0.0)

    with pytest.raises(Exception):  # noqa: B017, PT011 - dataclasses.FrozenInstanceError
        state.strain = 1.0


# --- ElasticMaterialAdapter ---


def test_elastic_adapter_from_material_matches_scalar_modulus() -> None:
    steel = Material(
        name="Steel", density=7850.0, youngs_modulus=YOUNGS_MODULUS, poissons_ratio=0.3
    )
    adapter = ElasticMaterialAdapter.from_material(steel)

    state = adapter.trial_state(0.001, adapter.initial_state())

    assert state.stress == pytest.approx(YOUNGS_MODULUS * 0.001)
    assert state.yielded is False
    assert adapter.tangent_modulus(state) == YOUNGS_MODULUS


def test_elastic_adapter_from_linear_elastic_2d_matches_constitutive_matrix() -> None:
    material_2d = LinearElastic2D(
        youngs_modulus=210e9, poisson_ratio=0.3, formulation="plane_stress"
    )
    adapter = ElasticMaterialAdapter.from_linear_elastic_2d(material_2d)
    strain = np.array([0.001, -0.0002, 0.0005])

    state = adapter.trial_state(strain, adapter.initial_state())

    assert_allclose(state.stress, material_2d.constitutive_matrix @ strain)
    assert_allclose(adapter.tangent_modulus(state), material_2d.constitutive_matrix)


def test_elastic_adapter_never_yields() -> None:
    adapter = ElasticMaterialAdapter(modulus=YOUNGS_MODULUS)

    state = adapter.trial_state(1.0, adapter.initial_state())

    assert state.yielded is False


def test_elastic_adapter_rejects_non_positive_scalar_modulus() -> None:
    with pytest.raises(ValidationError):
        ElasticMaterialAdapter(modulus=0.0)
    with pytest.raises(ValidationError):
        ElasticMaterialAdapter(modulus=-1.0)


def test_elastic_adapter_rejects_wrong_shaped_matrix_modulus() -> None:
    with pytest.raises(ValidationError):
        ElasticMaterialAdapter(modulus=np.eye(2))


# --- ElasticPerfectlyPlasticMaterial1D ---


@pytest.fixture
def bilinear_material() -> ElasticPerfectlyPlasticMaterial1D:
    return ElasticPerfectlyPlasticMaterial1D(
        youngs_modulus=YOUNGS_MODULUS, yield_stress=YIELD_STRESS
    )


def test_bilinear_material_elastic_region(
    bilinear_material: ElasticPerfectlyPlasticMaterial1D,
) -> None:
    strain = YIELD_STRESS / YOUNGS_MODULUS * 0.5
    state = bilinear_material.trial_state(strain, bilinear_material.initial_state())

    assert state.stress == pytest.approx(YOUNGS_MODULUS * strain)
    assert state.yielded is False
    assert state.plastic_strain == pytest.approx(0.0)
    assert bilinear_material.tangent_modulus(state) == YOUNGS_MODULUS


def test_bilinear_material_yields_exactly_at_yield_stress(
    bilinear_material: ElasticPerfectlyPlasticMaterial1D,
) -> None:
    yield_strain = YIELD_STRESS / YOUNGS_MODULUS
    state = bilinear_material.trial_state(yield_strain, bilinear_material.initial_state())

    assert state.stress == pytest.approx(YIELD_STRESS)
    # exactly at the boundary: |trial_stress| <= yield_stress holds, so not yet flagged as yielded.
    assert state.yielded is False


def test_bilinear_material_plastic_region_stress_stays_at_yield(
    bilinear_material: ElasticPerfectlyPlasticMaterial1D,
) -> None:
    yield_strain = YIELD_STRESS / YOUNGS_MODULUS
    state = bilinear_material.trial_state(5.0 * yield_strain, bilinear_material.initial_state())

    assert state.stress == pytest.approx(YIELD_STRESS)
    assert state.yielded is True
    assert state.plastic_strain > 0.0


def test_bilinear_material_plastic_strain_accumulates_correctly(
    bilinear_material: ElasticPerfectlyPlasticMaterial1D,
) -> None:
    yield_strain = YIELD_STRESS / YOUNGS_MODULUS
    total_strain = 3.0 * yield_strain
    state = bilinear_material.trial_state(total_strain, bilinear_material.initial_state())

    expected_plastic_strain = total_strain - YIELD_STRESS / YOUNGS_MODULUS
    assert state.plastic_strain == pytest.approx(expected_plastic_strain)


def test_bilinear_material_compression_is_symmetric(
    bilinear_material: ElasticPerfectlyPlasticMaterial1D,
) -> None:
    yield_strain = YIELD_STRESS / YOUNGS_MODULUS
    state = bilinear_material.trial_state(-5.0 * yield_strain, bilinear_material.initial_state())

    assert state.stress == pytest.approx(-YIELD_STRESS)
    assert state.yielded is True
    assert state.plastic_strain < 0.0


def test_bilinear_material_tangent_modulus_reduced_when_yielded(
    bilinear_material: ElasticPerfectlyPlasticMaterial1D,
) -> None:
    yield_strain = YIELD_STRESS / YOUNGS_MODULUS
    state = bilinear_material.trial_state(2.0 * yield_strain, bilinear_material.initial_state())

    tangent = bilinear_material.tangent_modulus(state)
    assert 0.0 < tangent < YOUNGS_MODULUS * 1e-3


def test_bilinear_material_unloading_from_plastic_state_is_elastic(
    bilinear_material: ElasticPerfectlyPlasticMaterial1D,
) -> None:
    """Once yielded, a strain decrease is measured elastically relative to
    the accumulated plastic strain -- not immediately re-yielding in
    compression -- matching a real elastic-perfectly-plastic unload/reload
    within the (shifted) elastic range.
    """
    yield_strain = YIELD_STRESS / YOUNGS_MODULUS
    yielded_state = bilinear_material.trial_state(
        3.0 * yield_strain, bilinear_material.initial_state()
    )
    plastic_strain = yielded_state.plastic_strain

    # Unload by a small elastic amount from the yielded strain.
    unloaded_strain = yielded_state.strain - 0.1 * yield_strain
    unloaded_state = bilinear_material.trial_state(unloaded_strain, yielded_state)

    expected_stress = YOUNGS_MODULUS * (unloaded_strain - plastic_strain)
    assert unloaded_state.stress == pytest.approx(expected_stress)
    assert unloaded_state.yielded is False
    assert unloaded_state.plastic_strain == pytest.approx(plastic_strain)


def test_bilinear_material_trial_state_does_not_mutate_committed_state(
    bilinear_material: ElasticPerfectlyPlasticMaterial1D,
) -> None:
    committed = bilinear_material.initial_state()
    bilinear_material.trial_state(10.0 * YIELD_STRESS / YOUNGS_MODULUS, committed)

    assert committed.strain == 0.0
    assert committed.stress == 0.0
    assert committed.plastic_strain == 0.0
    assert committed.yielded is False


def test_bilinear_material_rejects_non_positive_parameters() -> None:
    with pytest.raises(ValidationError):
        ElasticPerfectlyPlasticMaterial1D(youngs_modulus=0.0, yield_stress=YIELD_STRESS)
    with pytest.raises(ValidationError):
        ElasticPerfectlyPlasticMaterial1D(youngs_modulus=YOUNGS_MODULUS, yield_stress=-1.0)
