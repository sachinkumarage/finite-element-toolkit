"""Tests for harmonic_response and frequency_response."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from femtoolkit.analysis import BoundaryCondition, NodalLoad, TranslationDOF
from femtoolkit.analysis.dof import DOFMap
from femtoolkit.analysis.dynamic_system import DynamicSystem
from femtoolkit.analysis.harmonic import frequency_response, harmonic_response
from femtoolkit.exceptions import SingularSystemError, ValidationError

MASS = 2.0
DAMPING = 5.0
STIFFNESS = 800.0
FORCE_AMPLITUDE = 10.0


@pytest.fixture
def sdof_system() -> DynamicSystem:
    """A 2-node, 1-DOF-per-node system: node 1 fixed, node 2 the oscillating mass."""
    dof_map = DOFMap(node_ids=[1, 2], dofs_per_node=1)
    mass = np.array([[1.0, 0.0], [0.0, MASS]])
    damping = np.array([[0.0, 0.0], [0.0, DAMPING]])
    stiffness = np.array([[1.0, 0.0], [0.0, STIFFNESS]])
    bcs = [BoundaryCondition(1, TranslationDOF.X, 0.0)]
    return DynamicSystem(
        dof_map=dof_map, mass=mass, damping=damping, stiffness=stiffness, boundary_conditions=bcs
    )


@pytest.fixture
def sdof_loads() -> list[NodalLoad]:
    return [NodalLoad(2, TranslationDOF.X, FORCE_AMPLITUDE)]


def _analytical_amplitude(omega: np.ndarray) -> np.ndarray:
    return FORCE_AMPLITUDE / np.sqrt((STIFFNESS - MASS * omega**2) ** 2 + (DAMPING * omega) ** 2)


# --- harmonic_response ---


def test_harmonic_response_matches_analytical_sdof_formula(sdof_system, sdof_loads) -> None:
    omega = 5.0
    result = harmonic_response(sdof_system, omega, sdof_loads)

    expected_amplitude = _analytical_amplitude(np.array([omega]))[0]
    assert_allclose(result.amplitude_at(2, TranslationDOF.X), expected_amplitude, rtol=1e-9)


def test_harmonic_response_below_resonance(sdof_system, sdof_loads) -> None:
    omega_n = np.sqrt(STIFFNESS / MASS)
    result = harmonic_response(sdof_system, omega_n * 0.1, sdof_loads)
    # Well below resonance: response is nearly the static displacement F0/k.
    static_amplitude = FORCE_AMPLITUDE / STIFFNESS
    assert_allclose(result.amplitude_at(2, TranslationDOF.X), static_amplitude, rtol=0.05)


def test_harmonic_response_near_resonance_amplifies(sdof_system, sdof_loads) -> None:
    omega_n = np.sqrt(STIFFNESS / MASS)
    static_amplitude = FORCE_AMPLITUDE / STIFFNESS
    result = harmonic_response(sdof_system, omega_n, sdof_loads)
    assert result.amplitude_at(2, TranslationDOF.X) > 3 * static_amplitude


def test_harmonic_response_above_resonance_attenuates(sdof_system, sdof_loads) -> None:
    omega_n = np.sqrt(STIFFNESS / MASS)
    static_amplitude = FORCE_AMPLITUDE / STIFFNESS
    result = harmonic_response(sdof_system, omega_n * 10, sdof_loads)
    assert result.amplitude_at(2, TranslationDOF.X) < 0.1 * static_amplitude


def test_harmonic_response_zero_frequency_matches_static_solution(
    sdof_system, sdof_loads
) -> None:
    result = harmonic_response(sdof_system, 0.0, sdof_loads)
    static_amplitude = FORCE_AMPLITUDE / STIFFNESS
    assert_allclose(result.amplitude_at(2, TranslationDOF.X), static_amplitude, rtol=1e-9)
    assert_allclose(result.phase_at(2, TranslationDOF.X), 0.0, atol=1e-9)


def test_harmonic_response_phase_near_zero_below_resonance(sdof_system, sdof_loads) -> None:
    omega_n = np.sqrt(STIFFNESS / MASS)
    result = harmonic_response(sdof_system, omega_n * 0.1, sdof_loads)
    assert abs(result.phase_at(2, TranslationDOF.X)) < 0.2  # radians, close to in-phase


def test_harmonic_response_phase_near_pi_above_resonance(sdof_system, sdof_loads) -> None:
    omega_n = np.sqrt(STIFFNESS / MASS)
    result = harmonic_response(sdof_system, omega_n * 20, sdof_loads)
    assert abs(abs(result.phase_at(2, TranslationDOF.X)) - np.pi) < 0.2


def test_harmonic_response_phase_degrees_matches_radians(sdof_system, sdof_loads) -> None:
    result = harmonic_response(sdof_system, 5.0, sdof_loads)
    assert_allclose(result.phase_degrees(), np.degrees(result.phase))


def test_harmonic_response_constrained_dof_is_zero(sdof_system, sdof_loads) -> None:
    result = harmonic_response(sdof_system, 5.0, sdof_loads)
    assert result.displacement_at(1, TranslationDOF.X) == 0.0


def test_harmonic_response_rejects_negative_frequency(sdof_system, sdof_loads) -> None:
    with pytest.raises(ValidationError):
        harmonic_response(sdof_system, -1.0, sdof_loads)


def test_harmonic_response_singular_at_exact_undamped_resonance() -> None:
    """An undamped SDOF driven exactly at its natural frequency has an
    infinite theoretical response -- the dynamic stiffness matrix
    (a 1x1 in the free DOF here) is exactly singular.
    """
    dof_map = DOFMap(node_ids=[1, 2], dofs_per_node=1)
    mass = np.array([[1.0, 0.0], [0.0, 1.0]])
    damping = np.zeros((2, 2))
    stiffness = np.array([[1.0, 0.0], [0.0, 100.0]])
    bcs = [BoundaryCondition(1, TranslationDOF.X, 0.0)]
    system = DynamicSystem(
        dof_map=dof_map, mass=mass, damping=damping, stiffness=stiffness, boundary_conditions=bcs
    )
    loads = [NodalLoad(2, TranslationDOF.X, 1.0)]
    omega_n = 10.0  # sqrt(100/1)

    with pytest.raises(SingularSystemError):
        harmonic_response(system, omega_n, loads)


# --- frequency_response ---


def test_frequency_response_correct_point_count(sdof_system, sdof_loads) -> None:
    freqs = np.linspace(0.1, 20.0, 50)
    result = frequency_response(sdof_system, freqs, sdof_loads)
    assert result.frequencies.shape == (50,)
    assert result.amplitude.shape == (50, 2)
    assert result.phase.shape == (50, 2)


def test_frequency_response_matches_analytical_across_sweep(sdof_system, sdof_loads) -> None:
    freqs_hz = np.linspace(0.1, 10.0, 200)
    result = frequency_response(sdof_system, freqs_hz, sdof_loads)

    expected = _analytical_amplitude(result.angular_frequencies)
    assert_allclose(result.amplitude_at(2, TranslationDOF.X), expected, rtol=1e-9)


def test_frequency_response_all_finite(sdof_system, sdof_loads) -> None:
    freqs_hz = np.linspace(0.1, 100.0, 500)
    result = frequency_response(sdof_system, freqs_hz, sdof_loads)
    assert np.isfinite(result.amplitude).all()
    assert np.isfinite(result.phase).all()


def test_frequency_response_maximum_near_expected_resonance(sdof_system, sdof_loads) -> None:
    freqs_hz = np.linspace(0.1, 20.0, 2000)
    result = frequency_response(sdof_system, freqs_hz, sdof_loads)

    omega_n = np.sqrt(STIFFNESS / MASS)
    peak_omega = result.angular_frequencies[np.argmax(result.amplitude_at(2, TranslationDOF.X))]
    assert_allclose(peak_omega, omega_n, rtol=0.05)


def test_frequency_response_phase_increases_monotonically_through_resonance(
    sdof_system, sdof_loads
) -> None:
    freqs_hz = np.linspace(0.01, 20.0, 500)
    result = frequency_response(sdof_system, freqs_hz, sdof_loads)
    phase = result.phase_at(2, TranslationDOF.X)
    # Phase lag should become monotonically more negative as frequency increases.
    assert np.all(np.diff(phase) <= 1e-9)


def test_frequency_response_rejects_empty_frequencies(sdof_system, sdof_loads) -> None:
    with pytest.raises(ValidationError):
        frequency_response(sdof_system, np.array([]), sdof_loads)


def test_frequency_response_rejects_negative_frequency(sdof_system, sdof_loads) -> None:
    with pytest.raises(ValidationError):
        frequency_response(sdof_system, np.array([1.0, -1.0]), sdof_loads)


def test_frequency_response_degrees_helper(sdof_system, sdof_loads) -> None:
    freqs_hz = np.linspace(0.1, 10.0, 20)
    result = frequency_response(sdof_system, freqs_hz, sdof_loads)
    expected = np.degrees(result.phase_at(2, TranslationDOF.X))
    assert_allclose(result.phase_degrees_at(2, TranslationDOF.X), expected)
