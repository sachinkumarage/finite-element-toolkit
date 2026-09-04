"""Validation: material tangent vs. finite-difference stress derivative (spec section 15).

Independent of the finite-difference *implementation* used internally by
femtoolkit.materials.hyperelastic (which this test does not import), this
file re-derives the tangent from stress at nearby strain states using its
own, separately-coded central difference, and checks it against
NeoHookean3D's tangent (which chains an analytical stress with the base
class's numerical tangent) and MooneyRivlin3D's tangent (fully numerical)
-- confirming ``stress(F + dF)`` matches the tangent's linear prediction,
as required by spec section 15.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from femtoolkit.continuum.tensor import tensor_to_voigt_strain
from femtoolkit.materials import MooneyRivlin3D, NeoHookean3D

_MATERIALS = [
    NeoHookean3D(youngs_modulus=5.0e6, poisson_ratio=0.45),
    MooneyRivlin3D(c10=0.4e6, c01=0.1e6, bulk_modulus=200e6),
]
_MATERIAL_IDS = ["neo_hookean", "mooney_rivlin"]

# Physically realistic deformation states (moderate stretch, near-isochoric):
# equilibrium loading would produce states like these, not the extreme,
# near-incompressible-under-huge-bulk-modulus states documented as
# tangent-unstable in hyperelastic.py's module docstring.
_STATES = [
    np.diag([1.2, 1.2 ** (-0.5), 1.2 ** (-0.5)]),
    np.array([[1.05, 0.1, 0.0], [0.0, 0.97, 0.0], [0.0, 0.0, 0.99]]),
    np.array([[0.9, 0.0, 0.0], [0.05, 1.08, 0.02], [0.0, 0.0, 1.03]]),
]


def _independently_coded_finite_difference_tangent(material, f: np.ndarray) -> np.ndarray:
    """A separately-implemented central-difference dS/dE, deliberately not sharing
    any code with femtoolkit.materials.hyperelastic._tangent_from_strain_voigt."""
    step = 1e-6
    c = f.T @ f
    strain_voigt = tensor_to_voigt_strain(0.5 * (c - np.eye(3)))
    tangent = np.zeros((6, 6))
    for component in range(6):
        perturbation = np.zeros(6)
        perturbation[component] = step
        stress_plus = material._stress_from_strain_voigt(strain_voigt + perturbation)
        stress_minus = material._stress_from_strain_voigt(strain_voigt - perturbation)
        tangent[:, component] = (stress_plus - stress_minus) / (2.0 * step)
    return tangent


@pytest.mark.parametrize("material", _MATERIALS, ids=_MATERIAL_IDS)
@pytest.mark.parametrize("f", _STATES)
def test_tangent_matches_independent_finite_difference(material, f: np.ndarray) -> None:
    reported_tangent = material.material_tangent(f)
    reference_tangent = _independently_coded_finite_difference_tangent(material, f)
    assert_allclose(reported_tangent, reference_tangent, rtol=2e-2, atol=1e3)


@pytest.mark.parametrize("material", _MATERIALS, ids=_MATERIAL_IDS)
@pytest.mark.parametrize("f", _STATES)
def test_stress_at_perturbed_strain_matches_tangent_linear_prediction(
    material, f: np.ndarray
) -> None:
    """stress(E + dE) approx stress(E) + tangent @ dE, for a small dE (spec section 15)."""
    c = f.T @ f
    strain_voigt = tensor_to_voigt_strain(0.5 * (c - np.eye(3)))
    tangent = material.material_tangent(f)
    stress_at_e = material._stress_from_strain_voigt(strain_voigt)

    rng = np.random.default_rng(42)
    delta_e = rng.normal(scale=1e-5, size=6)

    predicted_stress = stress_at_e + tangent @ delta_e
    actual_stress = material._stress_from_strain_voigt(strain_voigt + delta_e)

    assert_allclose(actual_stress, predicted_stress, rtol=1e-2, atol=50.0)


# Pure isochoric uniaxial-with-Poisson-contraction states only (J == 1 exactly, no
# shear): these are the near-equilibrium states an actual Newton-Raphson solve
# would reach for a near-incompressible material, and where a positive-definite
# tangent is physically expected. States that combine shear with even a small
# volume change (see _STATES above) can show genuine, non-numerical negative
# tangent eigenvalues for MooneyRivlin3D under a stiff bulk_modulus, as
# documented in femtoolkit.materials.hyperelastic's module docstring and
# tests/validation/test_nearly_incompressible.py -- deliberately not tested
# for positive-definiteness here.
_PD_STATES = [
    np.diag([1.1, 1.1 ** (-0.5), 1.1 ** (-0.5)]),
    np.diag([1.3, 1.3 ** (-0.5), 1.3 ** (-0.5)]),
    np.diag([0.85, 0.85 ** (-0.5), 0.85 ** (-0.5)]),
]


@pytest.mark.parametrize("material", _MATERIALS, ids=_MATERIAL_IDS)
@pytest.mark.parametrize("f", _PD_STATES)
def test_material_tangent_is_positive_definite_at_near_equilibrium_states(
    material, f: np.ndarray
) -> None:
    tangent = material.material_tangent(f)
    eigenvalues = np.linalg.eigvalsh(tangent)
    assert np.all(eigenvalues > 0), (
        f"Expected a positive-definite tangent at a near-equilibrium deformation "
        f"state; got eigenvalues {eigenvalues}"
    )
