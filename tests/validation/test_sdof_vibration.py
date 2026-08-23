"""Validation Case: single-degree-of-freedom oscillator, ``m u'' + c u' + k u = F(t)``.

The classical SDOF oscillator has closed-form solutions this module
checks the Version 11 dynamic machinery against directly, independent
of any finite element mesh:

* **Natural frequency**: ``omega_n = sqrt(k/m)``, checked against
  :func:`~femtoolkit.analysis.modal.natural_frequencies` on a 1x1 system.
* **Undamped free vibration**: ``u(t) = u0 * cos(omega_n * t)``, checked
  against :func:`~femtoolkit.analysis.newmark.newmark_step`.
* **Damped free vibration**: for an underdamped system (damping ratio
  ``zeta < 1``), ``u(t) = exp(-zeta*omega_n*t) * (cos(omega_d*t) +
  (zeta*omega_n/omega_d)*sin(omega_d*t))``, where
  ``omega_d = omega_n * sqrt(1 - zeta^2)``.
* **Static-limit test**: under a constant force, the time-averaged
  dynamic response converges to ``F0 / k``, the ordinary static
  displacement.
"""

import math

import numpy as np
from numpy.testing import assert_allclose

from femtoolkit.analysis.modal import natural_frequencies
from femtoolkit.analysis.newmark import newmark_step

MASS = 2.0
STIFFNESS = 800.0


def test_sdof_natural_frequency_matches_analytical() -> None:
    m = np.array([[MASS]])
    k = np.array([[STIFFNESS]])

    result = natural_frequencies(k, m)

    expected_omega = math.sqrt(STIFFNESS / MASS)
    assert_allclose(result.angular_frequencies[0], expected_omega, rtol=1e-9)
    assert_allclose(result.frequencies[0], expected_omega / (2 * math.pi), rtol=1e-9)


def _integrate(mass, damping, stiffness, u0, dt, n_steps):
    m = np.array([[mass]])
    c = np.array([[damping]])
    k = np.array([[stiffness]])
    u = np.array([u0])
    v = np.array([0.0])
    a = np.linalg.solve(m, np.array([0.0]) - c @ v - k @ u)

    times = [0.0]
    disp = [u[0]]
    for step in range(n_steps):
        u, v, a = newmark_step(m, c, k, u, v, a, np.array([0.0]), dt)
        times.append((step + 1) * dt)
        disp.append(u[0])
    return np.array(times), np.array(disp)


def test_undamped_free_vibration_matches_cosine_solution() -> None:
    omega_n = math.sqrt(STIFFNESS / MASS)
    dt = 0.0005
    period = 2 * math.pi / omega_n
    n_steps = round(3 * period / dt)  # three full periods

    times, disp = _integrate(MASS, 0.0, STIFFNESS, u0=0.05, dt=dt, n_steps=n_steps)

    analytical = 0.05 * np.cos(omega_n * times)
    assert_allclose(disp, analytical, atol=1e-4)


def test_damped_free_vibration_matches_analytical_envelope() -> None:
    damping = 20.0  # underdamped: zeta = 20 / (2*sqrt(2*800)) ~= 0.25
    omega_n = math.sqrt(STIFFNESS / MASS)
    zeta = damping / (2 * math.sqrt(MASS * STIFFNESS))
    assert zeta < 1.0, "test setup must be underdamped"
    omega_d = omega_n * math.sqrt(1 - zeta**2)

    dt = 0.0005
    n_steps = round(2.0 / dt)
    times, disp = _integrate(MASS, damping, STIFFNESS, u0=0.05, dt=dt, n_steps=n_steps)

    analytical = 0.05 * np.exp(-zeta * omega_n * times) * (
        np.cos(omega_d * times) + (zeta * omega_n / omega_d) * np.sin(omega_d * times)
    )
    assert_allclose(disp, analytical, atol=2e-4)


def test_damped_vibration_decays_over_time() -> None:
    damping = 20.0
    dt = 0.0005
    n_steps = round(2.0 / dt)
    _, disp = _integrate(MASS, damping, STIFFNESS, u0=0.05, dt=dt, n_steps=n_steps)

    first_quarter = np.abs(disp[: n_steps // 4]).max()
    last_quarter = np.abs(disp[-n_steps // 4 :]).max()
    assert last_quarter < first_quarter


def test_static_limit_constant_force_converges_to_static_displacement() -> None:
    """A constant force applied to an initially at-rest oscillator
    oscillates around, and (with damping) settles to, the static
    displacement F0 / k.
    """
    force = 100.0
    damping = 40.0  # heavily damped so the response settles within the window
    dt = 0.0005
    n_steps = round(3.0 / dt)

    m = np.array([[MASS]])
    c = np.array([[damping]])
    k = np.array([[STIFFNESS]])
    u = np.array([0.0])
    v = np.array([0.0])
    a = np.linalg.solve(m, np.array([force]) - c @ v - k @ u)

    disp = [u[0]]
    for _ in range(n_steps):
        u, v, a = newmark_step(m, c, k, u, v, a, np.array([force]), dt)
        disp.append(u[0])

    static_displacement = force / STIFFNESS
    assert_allclose(disp[-1], static_displacement, rtol=1e-3)
