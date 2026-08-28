"""Validation Case: single-degree-of-freedom harmonic (frequency) response.

For ``m*u'' + c*u' + k*u = F0*sin(omega*t)``, the exact steady-state
response amplitude is:

.. code-block:: text

    |U/F0| = 1 / sqrt((k - m*omega^2)^2 + (c*omega)^2)

This is the textbook formula every general-purpose harmonic/frequency
response tool must reduce to for the simplest possible system --
checked here directly against
:func:`~femtoolkit.analysis.harmonic.harmonic_response` at frequencies
below, at, and above resonance (complementing the qualitative resonance
behavior covered in ``test_resonance.py``).
"""

import math

import numpy as np
import pytest
from numpy.testing import assert_allclose

from femtoolkit.analysis import BoundaryCondition, NodalLoad, TranslationDOF
from femtoolkit.analysis.dof import DOFMap
from femtoolkit.analysis.dynamic_system import DynamicSystem
from femtoolkit.analysis.harmonic import harmonic_response

MASS = 3.0
DAMPING = 8.0
STIFFNESS = 1200.0
FORCE_AMPLITUDE = 25.0


@pytest.fixture
def system() -> DynamicSystem:
    dof_map = DOFMap(node_ids=[1, 2], dofs_per_node=1)
    mass = np.array([[1.0, 0.0], [0.0, MASS]])
    damping = np.array([[0.0, 0.0], [0.0, DAMPING]])
    stiffness = np.array([[1.0, 0.0], [0.0, STIFFNESS]])
    bcs = [BoundaryCondition(1, TranslationDOF.X, 0.0)]
    return DynamicSystem(
        dof_map=dof_map, mass=mass, damping=damping, stiffness=stiffness, boundary_conditions=bcs
    )


def _analytical_amplitude(omega: float) -> float:
    return FORCE_AMPLITUDE / math.sqrt((STIFFNESS - MASS * omega**2) ** 2 + (DAMPING * omega) ** 2)


@pytest.mark.parametrize("omega_fraction", [0.1, 0.5, 0.9, 1.0, 1.1, 2.0, 10.0])
def test_amplitude_matches_analytical_formula_across_range(system, omega_fraction) -> None:
    omega_n = math.sqrt(STIFFNESS / MASS)
    omega = omega_n * omega_fraction

    result = harmonic_response(system, omega, [NodalLoad(2, TranslationDOF.X, FORCE_AMPLITUDE)])

    assert_allclose(
        result.amplitude_at(2, TranslationDOF.X), _analytical_amplitude(omega), rtol=1e-9
    )


def test_phase_matches_analytical_arctangent_formula(system) -> None:
    """phase = -atan2(c*omega, k - m*omega^2), the standard closed-form
    phase-lag expression for a damped SDOF oscillator.
    """
    omega_n = math.sqrt(STIFFNESS / MASS)
    omega = omega_n * 1.3

    result = harmonic_response(system, omega, [NodalLoad(2, TranslationDOF.X, FORCE_AMPLITUDE)])

    expected_phase = -math.atan2(DAMPING * omega, STIFFNESS - MASS * omega**2)
    assert_allclose(result.phase_at(2, TranslationDOF.X), expected_phase, rtol=1e-9)


def test_static_limit_matches_f0_over_k(system) -> None:
    result = harmonic_response(system, 0.0, [NodalLoad(2, TranslationDOF.X, FORCE_AMPLITUDE)])
    static_amplitude = FORCE_AMPLITUDE / STIFFNESS
    assert_allclose(result.amplitude_at(2, TranslationDOF.X), static_amplitude, rtol=1e-9)
