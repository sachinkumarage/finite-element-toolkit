"""Validation: resonance behavior of a damped single-degree-of-freedom oscillator.

For ``m*u'' + c*u' + k*u = F0*sin(omega*t)``, the steady-state response
amplitude,

.. code-block:: text

    |U/F0| = 1 / sqrt((k - m*omega^2)^2 + (c*omega)^2)

grows sharply as the excitation frequency ``omega`` approaches the
undamped natural frequency ``omega_n = sqrt(k/m)``, bounded only by
damping: lighter damping gives a taller, narrower peak; heavier damping
flattens it. This module verifies that qualitative (and, at exact
resonance, exactly quantitative) behavior directly against
:func:`~femtoolkit.analysis.harmonic.frequency_response`.
"""

import math

import numpy as np
import pytest
from numpy.testing import assert_allclose

from femtoolkit.analysis import BoundaryCondition, NodalLoad, TranslationDOF
from femtoolkit.analysis.dof import DOFMap
from femtoolkit.analysis.dynamic_system import DynamicSystem
from femtoolkit.analysis.harmonic import frequency_response, harmonic_response

MASS = 1.0
STIFFNESS = 400.0  # omega_n = 20 rad/s
FORCE_AMPLITUDE = 10.0


def _sdof_system(damping_coefficient: float) -> DynamicSystem:
    dof_map = DOFMap(node_ids=[1, 2], dofs_per_node=1)
    mass = np.array([[1.0, 0.0], [0.0, MASS]])
    damping = np.array([[0.0, 0.0], [0.0, damping_coefficient]])
    stiffness = np.array([[1.0, 0.0], [0.0, STIFFNESS]])
    bcs = [BoundaryCondition(1, TranslationDOF.X, 0.0)]
    return DynamicSystem(
        dof_map=dof_map, mass=mass, damping=damping, stiffness=stiffness, boundary_conditions=bcs
    )


def test_exact_resonance_amplitude_matches_analytical_formula() -> None:
    """At exact resonance (omega = omega_n), the analytical formula
    reduces to |U/F0| = 1/(c*omega_n) -- the (k - m*omega^2) term
    vanishes identically.
    """
    damping_coefficient = 5.0
    system = _sdof_system(damping_coefficient)
    omega_n = math.sqrt(STIFFNESS / MASS)

    result = harmonic_response(system, omega_n, [NodalLoad(2, TranslationDOF.X, FORCE_AMPLITUDE)])

    expected = FORCE_AMPLITUDE / (damping_coefficient * omega_n)
    assert_allclose(result.amplitude_at(2, TranslationDOF.X), expected, rtol=1e-9)


def test_lighter_damping_gives_a_taller_resonance_peak() -> None:
    omega_n_hz = math.sqrt(STIFFNESS / MASS) / (2 * math.pi)
    freqs = np.linspace(omega_n_hz * 0.5, omega_n_hz * 1.5, 500)
    loads = [NodalLoad(2, TranslationDOF.X, FORCE_AMPLITUDE)]

    light = frequency_response(_sdof_system(2.0), freqs, loads)
    heavy = frequency_response(_sdof_system(20.0), freqs, loads)

    peak_light = light.amplitude_at(2, TranslationDOF.X).max()
    peak_heavy = heavy.amplitude_at(2, TranslationDOF.X).max()
    assert peak_light > peak_heavy


def test_response_far_from_resonance_is_insensitive_to_damping() -> None:
    """Well below resonance, the response approaches the static
    displacement F0/k regardless of damping (the c*omega term is
    negligible there).
    """
    omega_n_hz = math.sqrt(STIFFNESS / MASS) / (2 * math.pi)
    low_freq = np.array([omega_n_hz * 0.02])
    loads = [NodalLoad(2, TranslationDOF.X, FORCE_AMPLITUDE)]

    light = frequency_response(_sdof_system(2.0), low_freq, loads)
    heavy = frequency_response(_sdof_system(20.0), low_freq, loads)

    static_amplitude = FORCE_AMPLITUDE / STIFFNESS
    assert_allclose(light.amplitude_at(2, TranslationDOF.X)[0], static_amplitude, rtol=0.01)
    assert_allclose(heavy.amplitude_at(2, TranslationDOF.X)[0], static_amplitude, rtol=0.05)


def test_undamped_amplitude_diverges_approaching_resonance() -> None:
    """With damping fixed at (near) zero, amplitude must grow without
    bound as the excitation frequency approaches resonance -- verified
    by checking the amplitude keeps increasing as omega approaches
    omega_n from below, across several closer and closer points.
    """
    tiny_damping = 1e-6
    system = _sdof_system(tiny_damping)
    omega_n = math.sqrt(STIFFNESS / MASS)
    loads = [NodalLoad(2, TranslationDOF.X, FORCE_AMPLITUDE)]

    offsets = [0.5, 0.1, 0.01, 0.001]
    amplitudes = []
    for offset in offsets:
        result = harmonic_response(system, omega_n * (1.0 - offset), loads)
        amplitudes.append(result.amplitude_at(2, TranslationDOF.X))

    assert np.all(np.diff(amplitudes) > 0)  # monotonically increasing toward resonance


@pytest.mark.parametrize("damping_coefficient", [1.0, 5.0, 20.0])
def test_peak_amplitude_location_matches_analytical_damped_resonance(damping_coefficient) -> None:
    """The frequency of maximum amplitude for a damped SDOF oscillator is
    ``omega_peak = sqrt(omega_n^2 - c^2/(2*m^2))`` (from differentiating
    the amplitude formula and setting the derivative to zero) --
    slightly below the undamped natural frequency for nonzero damping.
    """
    omega_n = math.sqrt(STIFFNESS / MASS)
    expected_peak = math.sqrt(max(omega_n**2 - damping_coefficient**2 / (2 * MASS**2), 0.0))

    system = _sdof_system(damping_coefficient)
    freqs_hz = np.linspace(0.01, omega_n / (2 * math.pi) * 2, 4000)
    result = frequency_response(system, freqs_hz, [NodalLoad(2, TranslationDOF.X, FORCE_AMPLITUDE)])

    peak_omega = result.angular_frequencies[np.argmax(result.amplitude_at(2, TranslationDOF.X))]
    assert_allclose(peak_omega, expected_peak, rtol=0.01)
