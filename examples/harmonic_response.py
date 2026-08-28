"""Example: harmonic loading and a frequency sweep, Version 12.

Demonstrates :func:`~femtoolkit.analysis.harmonic.harmonic_response` and
:func:`~femtoolkit.analysis.harmonic.frequency_response` on a
cantilevered Q4 plate under a harmonic tip load, sweeping the excitation
frequency across the plate's first natural frequency to show the
resonance peak in displacement amplitude and the corresponding phase
shift from in-phase to out-of-phase response.
"""

import numpy as np

from femtoolkit.analysis import BoundaryCondition, NodalLoad, RayleighDamping, TranslationDOF
from femtoolkit.analysis.dynamic_system import build_dynamic_system
from femtoolkit.analysis.harmonic import frequency_response, harmonic_response
from femtoolkit.analysis.modal import natural_frequencies_of_system
from femtoolkit.geometry import Rectangle
from femtoolkit.materials import LinearElastic2D
from femtoolkit.mesh import create_quad_mesh

YOUNGS_MODULUS = 200e9  # Pa
POISSON_RATIO = 0.3
DENSITY = 7850.0  # kg/m^3
THICKNESS = 0.01  # m
WIDTH = 0.5  # m
HEIGHT = 0.1  # m
NX = 10
NY = 3
FORCE_AMPLITUDE = 10.0  # N


def main() -> None:
    """Sweep excitation frequency across a cantilevered plate's fundamental frequency."""
    material = LinearElastic2D(
        youngs_modulus=YOUNGS_MODULUS,
        poisson_ratio=POISSON_RATIO,
        formulation="plane_stress",
        density=DENSITY,
    )
    domain = Rectangle(width=WIDTH, height=HEIGHT)
    mesh = create_quad_mesh(
        width=WIDTH, height=HEIGHT, nx=NX, ny=NY, material=material, thickness=THICKNESS
    )
    tip_node = max((n for n in mesh.nodes if n.x == WIDTH), key=lambda n: n.y)

    boundary_conditions = []
    for node in mesh.nodes_on_boundary(domain.boundary("left")):
        boundary_conditions.append(BoundaryCondition(node.id, TranslationDOF.X, 0.0))
        boundary_conditions.append(BoundaryCondition(node.id, TranslationDOF.Y, 0.0))

    damping = RayleighDamping(alpha=15.0, beta=0.0001)
    system = build_dynamic_system(mesh, boundary_conditions, damping=damping)

    modal_result = natural_frequencies_of_system(system, num_modes=1)
    fundamental_hz = modal_result.frequencies[0]

    loads = [NodalLoad(tip_node.id, TranslationDOF.Y, FORCE_AMPLITUDE)]

    single = harmonic_response(system, 2 * np.pi * fundamental_hz, loads)

    sweep_freqs = np.linspace(0.2 * fundamental_hz, 2.0 * fundamental_hz, 300)
    sweep = frequency_response(system, sweep_freqs, loads)

    print_summary(tip_node.id, fundamental_hz, single, sweep)


def print_summary(tip_node_id: int, fundamental_hz: float, single, sweep) -> None:
    """Print the single-frequency result and a summary of the swept response."""
    print("Finite Element Toolkit")
    print("Version 12 -- Harmonic Response and Frequency Sweep")
    print("=" * 60)

    print(f"\nFundamental natural frequency: {fundamental_hz:.3f} Hz")

    print("\nHarmonic response driven exactly at the fundamental frequency:")
    amplitude = single.amplitude_at(tip_node_id, TranslationDOF.Y)
    phase_deg = np.degrees(single.phase_at(tip_node_id, TranslationDOF.Y))
    print(f"    Amplitude: {amplitude:.6e} m")
    print(f"    Phase:     {phase_deg:.2f} deg")

    amplitude_sweep = sweep.amplitude_at(tip_node_id, TranslationDOF.Y)
    phase_sweep = sweep.phase_degrees_at(tip_node_id, TranslationDOF.Y)
    peak_index = int(np.argmax(amplitude_sweep))

    print("\nFrequency sweep (every 30th point):")
    print(f"    {'f (Hz)':>10}{'amplitude (m)':>18}{'phase (deg)':>14}")
    for i in range(0, len(sweep.frequencies), 30):
        freq, amp, phase = sweep.frequencies[i], amplitude_sweep[i], phase_sweep[i]
        print(f"    {freq:>10.2f}{amp:>18.6e}{phase:>14.2f}")

    print(f"\nPeak amplitude at f = {sweep.frequencies[peak_index]:.3f} Hz")
    print(f"    (expected near the fundamental frequency, {fundamental_hz:.3f} Hz)")
    print(f"Phase at lowest swept frequency:  {phase_sweep[0]:.2f} deg (near 0, in phase)")
    print(f"Phase at highest swept frequency: {phase_sweep[-1]:.2f} deg (approaching -180)")


if __name__ == "__main__":
    main()
