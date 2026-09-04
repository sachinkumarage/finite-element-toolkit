"""Validation: W(F+dF) - W(F) vs. stress work S:dE (spec section 16).

The defining property of hyperelasticity is that stress is the *gradient*
of the strain-energy density: to first order, a small change in strain
energy must equal the stress work done over the corresponding small
strain increment, ``dW = S : dE``. This is checked directly here (not
merely as a byproduct of the stress-vs-tangent tests) at several
deformation states, for both Version 17 materials, using a small but
finite perturbation and first-order (O(h)) agreement -- distinct from
tests/validation/test_hyperelastic_tangent_validation.py, which checks
the *tangent* (second derivative of energy) against the finite-difference
of *stress*, not energy directly.
"""

import numpy as np
import pytest

from femtoolkit.continuum.tensor import tensor_to_voigt_strain, voigt_stress_to_tensor
from femtoolkit.materials import MooneyRivlin3D, NeoHookean3D

_MATERIALS = [
    NeoHookean3D(youngs_modulus=5.0e6, poisson_ratio=0.45),
    MooneyRivlin3D(c10=0.4e6, c01=0.1e6, bulk_modulus=200e6),
]
_MATERIAL_IDS = ["neo_hookean", "mooney_rivlin"]

_STATES = [
    np.eye(3),
    np.diag([1.2, 1.2 ** (-0.5), 1.2 ** (-0.5)]),
    np.array([[1.05, 0.1, 0.0], [0.0, 0.97, 0.0], [0.0, 0.0, 0.99]]),
    np.array([[0.9, 0.0, 0.0], [0.05, 1.08, 0.02], [0.0, 0.0, 1.03]]),
]


@pytest.mark.parametrize("material", _MATERIALS, ids=_MATERIAL_IDS)
@pytest.mark.parametrize("f", _STATES)
def test_energy_increment_matches_stress_work(material, f: np.ndarray) -> None:
    c = f.T @ f
    e_tensor = 0.5 * (c - np.eye(3))

    delta_e = np.zeros((3, 3))
    delta_e[0, 0] = 5e-7
    delta_e[1, 1] = -3e-7
    delta_e[0, 1] = delta_e[1, 0] = 2e-7

    energy_before = material._energy_from_right_cauchy_green(c)
    c_after = c + 2.0 * delta_e
    energy_after = material._energy_from_right_cauchy_green(c_after)

    stress_voigt = material._stress_from_strain_voigt(tensor_to_voigt_strain(e_tensor))
    stress_tensor = voigt_stress_to_tensor(stress_voigt)
    stress_work = float(np.tensordot(stress_tensor, delta_e))

    energy_increment = energy_after - energy_before
    assert energy_increment == pytest.approx(stress_work, rel=2e-2, abs=1e-4)


@pytest.mark.parametrize("material", _MATERIALS, ids=_MATERIAL_IDS)
def test_energy_is_nonnegative_for_all_tested_states(material) -> None:
    """Hyperelastic strain energy is bounded below by zero (minimized at F=I, W=0)."""
    for f in _STATES:
        assert material.strain_energy_density(f) >= -1e-6
