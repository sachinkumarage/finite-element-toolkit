"""Example: single-degree-of-freedom (SDOF) free vibration, Version 11.

Demonstrates the Version 11 dynamic mathematics on the simplest possible
system -- a single mass on a single spring, ``m u'' + k u = 0`` -- with
no finite element mesh involved at all:

* :func:`~femtoolkit.analysis.modal.natural_frequencies` solves the
  1x1 generalized eigenvalue problem for the natural frequency.
* :func:`~femtoolkit.analysis.newmark.newmark_step` integrates free
  vibration from an initial displacement, compared directly against the
  exact analytical solution, ``u(t) = u0 * cos(omega_n * t)``.
"""

import math

import numpy as np

from femtoolkit.analysis.modal import natural_frequencies
from femtoolkit.analysis.newmark import newmark_step

MASS = 2.0  # kg
STIFFNESS = 800.0  # N/m
INITIAL_DISPLACEMENT = 0.05  # m
TIME_STEP = 0.001  # s
TOTAL_TIME = 2.0  # s


def main() -> None:
    """Solve for the SDOF natural frequency, then integrate free vibration."""
    mass_matrix = np.array([[MASS]])
    stiffness_matrix = np.array([[STIFFNESS]])
    damping_matrix = np.array([[0.0]])

    modal_result = natural_frequencies(stiffness_matrix, mass_matrix)
    omega_n = modal_result.angular_frequencies[0]
    frequency_hz = modal_result.frequencies[0]

    print("Finite Element Toolkit")
    print("Version 11 -- Single-DOF Free Vibration")
    print("=" * 40)
    print(f"\nMass:      {MASS} kg")
    print(f"Stiffness: {STIFFNESS} N/m")
    print(f"\nNatural angular frequency: omega_n = {omega_n:.6f} rad/s")
    print(f"Natural frequency:         f = {frequency_hz:.6f} Hz")

    analytical_omega_n = math.sqrt(STIFFNESS / MASS)
    print(f"Analytical omega_n = sqrt(k/m): {analytical_omega_n:.6f} rad/s")

    u = np.array([INITIAL_DISPLACEMENT])
    v = np.array([0.0])
    a = np.linalg.solve(mass_matrix, np.array([0.0]) - stiffness_matrix @ u)

    n_steps = round(TOTAL_TIME / TIME_STEP)
    times = np.zeros(n_steps + 1)
    displacement = np.zeros(n_steps + 1)
    displacement[0] = u[0]

    for step in range(n_steps):
        u, v, a = newmark_step(
            mass_matrix, damping_matrix, stiffness_matrix, u, v, a, np.array([0.0]), TIME_STEP
        )
        times[step + 1] = (step + 1) * TIME_STEP
        displacement[step + 1] = u[0]

    analytical = INITIAL_DISPLACEMENT * np.cos(analytical_omega_n * times)
    max_error = np.abs(displacement - analytical).max()

    print(f"\nSimulated {TOTAL_TIME} s of free vibration in {n_steps} Newmark-beta steps.")
    print(f"Max abs error vs. analytical u(t) = u0*cos(omega_n*t): {max_error:.3e} m")
    print(f"Max |displacement| over the simulation: {np.abs(displacement).max():.6f} m")
    print(f"    (undamped: should stay very close to u0 = {INITIAL_DISPLACEMENT} m)")


if __name__ == "__main__":
    main()
