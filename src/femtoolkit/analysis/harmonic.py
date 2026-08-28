"""Steady-state harmonic response and frequency-response (frequency-sweep) analysis.

For a harmonic force ``F(t) = F0 * sin(omega*t)`` acting on a linear
system, the long-term (steady-state, transient decay ignored) response
is itself harmonic at the same frequency, ``u(t) = Re(U * exp(i*omega*t))``,
where ``U`` is a **complex displacement amplitude** satisfying the
frequency-domain equation:

.. code-block:: text

    [ -omega^2 * M + i*omega*C + K ] U = F

This reduces solving for the steady-state response to a single
*complex* linear solve at each frequency of interest -- no time
stepping needed, unlike :mod:`femtoolkit.analysis.newmark`. The complex
number ``U`` at each DOF encodes both **amplitude** (``|U|``, how far
that DOF moves) and **phase** (``atan2(Im(U), Re(U))``, how far behind
the applied force's own phase the response lags -- for an undamped
system below resonance the response is in phase (``0`` rad) with the
force; damping and/or driving above resonance shift it, approaching
``pi`` rad -- 180 degrees out of phase -- well above resonance).

:func:`harmonic_response` solves at one frequency;
:func:`frequency_response` sweeps a whole array of frequencies (a
**frequency-response function**), which is where **resonance** becomes
visible: the response amplitude grows sharply as the excitation
frequency approaches a natural frequency, bounded only by damping (see
``tests/validation/test_resonance.py``).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from femtoolkit.analysis.dof import DOFMap
from femtoolkit.analysis.dynamic_system import DynamicSystem, free_and_constrained_indices
from femtoolkit.analysis.loads import NodalLoad
from femtoolkit.analysis.system import build_force_vector
from femtoolkit.exceptions import SingularSystemError, ValidationError


@dataclass(frozen=True)
class HarmonicResult:
    """The steady-state harmonic response of a system at one excitation frequency.

    Attributes:
        dof_map: DOF map defining the global DOF numbering.
        angular_frequency: The excitation angular frequency, in rad/s.
        displacement: Complex displacement amplitude ``U``, shape
            ``(dof_map.total_dofs,)``. Zero at every constrained DOF
            (harmonic support motion is out of scope -- see the module
            docstring).
        amplitude: ``abs(displacement)``, the real displacement
            magnitude at each DOF.
        phase: ``angle(displacement)``, in radians, in
            ``(-pi, pi]``, the phase lag of each DOF's response
            relative to the applied force.
    """

    dof_map: DOFMap
    angular_frequency: float
    displacement: np.ndarray
    amplitude: np.ndarray
    phase: np.ndarray

    def phase_degrees(self) -> np.ndarray:
        """``phase``, converted to degrees, as a convenience (radians remain primary)."""
        return np.degrees(self.phase)

    def displacement_at(self, node_id: int, dof: int) -> complex:
        """Return the complex displacement amplitude at one node/DOF."""
        return complex(self.displacement[self.dof_map.global_index(node_id, dof)])

    def amplitude_at(self, node_id: int, dof: int) -> float:
        """Return the real displacement amplitude at one node/DOF."""
        return float(self.amplitude[self.dof_map.global_index(node_id, dof)])

    def phase_at(self, node_id: int, dof: int) -> float:
        """Return the phase lag, in radians, at one node/DOF."""
        return float(self.phase[self.dof_map.global_index(node_id, dof)])


def harmonic_response(
    system: DynamicSystem, angular_frequency: float, loads: Sequence[NodalLoad]
) -> HarmonicResult:
    """Solve ``[-omega^2*M + i*omega*C + K] U = F`` for the steady-state response.

    Args:
        system: The dynamic system to analyze.
        angular_frequency: Excitation angular frequency ``omega``, in
            rad/s. Must be finite and non-negative (``0`` is the static
            limit, where the equation reduces to the ordinary static
            ``K U = F``).
        loads: The force amplitude pattern ``F0`` (an ordinary
            :class:`~femtoolkit.analysis.loads.NodalLoad` list, reused
            unchanged from Version 2 -- the harmonic ``sin(omega*t)``
            time dependence is factored out of the frequency-domain
            equation entirely, so only the spatial amplitude pattern is
            needed here).

    Returns:
        A :class:`HarmonicResult`.

    Raises:
        ValidationError: If ``angular_frequency`` is negative or not finite.
        SingularSystemError: If the dynamic stiffness matrix
            ``-omega^2*M + i*omega*C + K`` is singular (e.g. undamped
            and driven exactly at a natural frequency -- true resonance,
            an infinite theoretical response).
    """
    if not math.isfinite(angular_frequency) or angular_frequency < 0:
        raise ValidationError(
            f"angular_frequency must be non-negative, got {angular_frequency}."
        )

    dof_map = system.dof_map
    total_dofs = dof_map.total_dofs
    free, _, _ = free_and_constrained_indices(system)

    mass_ff = system.mass[np.ix_(free, free)]
    damping_ff = system.damping[np.ix_(free, free)]
    stiffness_ff = system.stiffness[np.ix_(free, free)]

    dynamic_stiffness = (
        -(angular_frequency**2) * mass_ff + 1j * angular_frequency * damping_ff + stiffness_ff
    )

    force = build_force_vector(dof_map, loads)
    force_free = force[free]

    try:
        displacement_free = np.linalg.solve(dynamic_stiffness, force_free.astype(complex))
    except np.linalg.LinAlgError as error:
        raise SingularSystemError(
            "The dynamic stiffness matrix [-omega^2*M + i*omega*C + K] is singular; "
            "this is true resonance (an undamped system driven exactly at a natural "
            "frequency) or an insufficiently constrained system."
        ) from error

    displacement = np.zeros(total_dofs, dtype=complex)
    displacement[free] = displacement_free

    return HarmonicResult(
        dof_map=dof_map,
        angular_frequency=angular_frequency,
        displacement=displacement,
        amplitude=np.abs(displacement),
        phase=np.angle(displacement),
    )


@dataclass(frozen=True)
class FrequencyResponseResult:
    """A swept frequency-response function: :class:`HarmonicResult` at many frequencies.

    Attributes:
        dof_map: DOF map defining the global DOF numbering.
        frequencies: Ordinary excitation frequencies, in Hz, shape ``(n_freq,)``.
        angular_frequencies: ``2*pi*frequencies``, in rad/s, shape ``(n_freq,)``.
        displacement: Complex displacement amplitude at every frequency
            and DOF, shape ``(n_freq, dof_map.total_dofs)``.
        amplitude: ``abs(displacement)``, same shape.
        phase: ``angle(displacement)``, in radians, same shape.
    """

    dof_map: DOFMap
    frequencies: np.ndarray
    angular_frequencies: np.ndarray
    displacement: np.ndarray
    amplitude: np.ndarray
    phase: np.ndarray

    def displacement_at(self, node_id: int, dof: int) -> np.ndarray:
        """Return the complex displacement amplitude time series at one node/DOF."""
        return self.displacement[:, self.dof_map.global_index(node_id, dof)]

    def amplitude_at(self, node_id: int, dof: int) -> np.ndarray:
        """Return the displacement amplitude across the swept frequencies, at one node/DOF."""
        return self.amplitude[:, self.dof_map.global_index(node_id, dof)]

    def phase_at(self, node_id: int, dof: int) -> np.ndarray:
        """Return the phase lag (radians) across the swept frequencies, at one node/DOF."""
        return self.phase[:, self.dof_map.global_index(node_id, dof)]

    def phase_degrees_at(self, node_id: int, dof: int) -> np.ndarray:
        """Return the phase lag in degrees across the swept frequencies, at one node/DOF."""
        return np.degrees(self.phase_at(node_id, dof))


def frequency_response(
    system: DynamicSystem, frequencies: np.ndarray, loads: Sequence[NodalLoad]
) -> FrequencyResponseResult:
    """Sweep :func:`harmonic_response` over an array of excitation frequencies.

    Args:
        system: The dynamic system to analyze.
        frequencies: Ordinary excitation frequencies, in Hz (not
            angular). Every entry must be finite and non-negative.
        loads: The force amplitude pattern ``F0`` (see :func:`harmonic_response`).

    Returns:
        A :class:`FrequencyResponseResult` covering every requested
        frequency and every DOF.

    Raises:
        ValidationError: If ``frequencies`` is empty, or any entry is
            negative or not finite.
        SingularSystemError: See :func:`harmonic_response` (raised for
            the specific frequency that hits exact resonance).
    """
    frequencies = np.asarray(frequencies, dtype=float)
    if frequencies.size == 0:
        raise ValidationError("frequencies must not be empty.")
    if not np.all(np.isfinite(frequencies)) or np.any(frequencies < 0):
        raise ValidationError("Every entry in frequencies must be finite and non-negative.")

    angular_frequencies = 2.0 * math.pi * frequencies
    total_dofs = system.dof_map.total_dofs
    displacement = np.zeros((frequencies.size, total_dofs), dtype=complex)

    for i, omega in enumerate(angular_frequencies):
        result = harmonic_response(system, float(omega), loads)
        displacement[i] = result.displacement

    return FrequencyResponseResult(
        dof_map=system.dof_map,
        frequencies=frequencies,
        angular_frequencies=angular_frequencies,
        displacement=displacement,
        amplitude=np.abs(displacement),
        phase=np.angle(displacement),
    )
