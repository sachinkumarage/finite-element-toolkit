"""Validation: analytical (numerical-by-design) tangent vs. finite-difference (spec section 10).

:meth:`~femtoolkit.materials.finite_strain_plasticity.FiniteStrainPlasticMaterial.tangent_modulus`
is deliberately a central-difference tangent (see that module's docstring
for why a hand-derived closed form was judged too risky). This file is
the independent verification the spec explicitly requires: comparing the
reported tangent against a *separately implemented* finite-difference
tangent (not sharing code with the material's own internal
differencing), at deformation states ranging from a pristine reference
through several plastic load increments, and reporting the relative
error.
"""

import numpy as np
import pytest

from femtoolkit.materials import J2FiniteStrainPlasticity3D


@pytest.fixture
def steel() -> J2FiniteStrainPlasticity3D:
    return J2FiniteStrainPlasticity3D(
        youngs_modulus=200e9, poisson_ratio=0.3, yield_stress=250e6, hardening_modulus=10e9
    )


def _independent_finite_difference_tangent(material, strain, committed_state, step=1e-7):
    """A separately-implemented central-difference dS/dE, sharing no code with the material."""
    tangent = np.zeros((6, 6))
    for component in range(6):
        perturbation = np.zeros(6)
        perturbation[component] = step
        stress_plus = material.trial_state(strain + perturbation, committed_state).stress
        stress_minus = material.trial_state(strain - perturbation, committed_state).stress
        tangent[:, component] = (stress_plus - stress_minus) / (2.0 * step)
    return 0.5 * (tangent + tangent.T)


@pytest.mark.parametrize(
    "strain",
    [
        np.array([1e-5, 0, 0, 0, 0, 0]),
        np.array([0.01, 0, 0, 0, 0, 0]),
        np.array([0.02, -0.006, -0.002, 0.003, 0, 0]),
    ],
)
def test_tangent_matches_independent_finite_difference_from_pristine_reference(
    steel: J2FiniteStrainPlasticity3D, strain: np.ndarray
) -> None:
    committed = steel.initial_state()
    state = steel.trial_state(strain, committed)
    tangent = steel.tangent_modulus(state)

    reference = _independent_finite_difference_tangent(steel, strain, committed)
    scale = np.abs(reference).max()
    relative_error = np.abs(tangent - reference).max() / scale
    assert relative_error < 1e-3, f"relative error {relative_error} too large"


def test_tangent_matches_independent_finite_difference_after_multiple_plastic_steps(
    steel: J2FiniteStrainPlasticity3D,
) -> None:
    """The committed-reference reconstruction inside tangent_modulus must remain
    accurate several plastic load increments away from the pristine Fp=I state."""
    committed = steel.initial_state()
    for eps in (0.002, 0.005, 0.008):
        strain = np.array([eps, -0.3 * eps, -0.3 * eps, 0.0015, 0, 0])
        committed = steel.trial_state(strain, committed)

    final_strain = np.array([0.012, -0.0036, -0.0036, 0.0015, 0, 0])
    state = steel.trial_state(final_strain, committed)
    tangent = steel.tangent_modulus(state)

    reference = _independent_finite_difference_tangent(steel, final_strain, committed)
    scale = np.abs(reference).max()
    relative_error = np.abs(tangent - reference).max() / scale
    assert relative_error < 1e-3, f"relative error {relative_error} too large"


def test_tangent_is_symmetric_at_every_tested_state(steel: J2FiniteStrainPlasticity3D) -> None:
    committed = steel.initial_state()
    for eps in (0.001, 0.004, 0.009, 0.014):
        strain = np.array([eps, -0.3 * eps, -0.3 * eps, 0, 0, 0])
        state = steel.trial_state(strain, committed)
        tangent = steel.tangent_modulus(state)
        assert np.abs(tangent - tangent.T).max() / np.abs(tangent).max() < 1e-8
        committed = state
