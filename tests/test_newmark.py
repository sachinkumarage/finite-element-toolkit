"""Tests for Newmark-beta time integration."""

import math

import numpy as np
import pytest
from numpy.testing import assert_allclose

from femtoolkit.analysis.newmark import (
    DEFAULT_BETA,
    DEFAULT_GAMMA,
    effective_force,
    effective_stiffness,
    newmark_step,
    update_velocity_acceleration,
)
from femtoolkit.exceptions import SingularSystemError, ValidationError


def test_default_parameters_are_average_acceleration_method() -> None:
    assert DEFAULT_BETA == 0.25
    assert DEFAULT_GAMMA == 0.5


def test_effective_stiffness_formula() -> None:
    m = np.array([[1.0]])
    c = np.array([[0.1]])
    k = np.array([[100.0]])
    dt = 0.01
    beta, gamma = 0.25, 0.5

    k_eff = effective_stiffness(m, c, k, dt, beta, gamma)

    expected = k + (gamma / (beta * dt)) * c + (1.0 / (beta * dt**2)) * m
    assert_allclose(k_eff, expected)


def test_effective_stiffness_rejects_non_positive_dt() -> None:
    m, c, k = np.eye(1), np.eye(1), np.eye(1)
    with pytest.raises(ValidationError):
        effective_stiffness(m, c, k, dt=0.0)


def test_effective_stiffness_rejects_non_positive_beta() -> None:
    m, c, k = np.eye(1), np.eye(1), np.eye(1)
    with pytest.raises(ValidationError):
        effective_stiffness(m, c, k, dt=0.01, beta=0.0)


def test_effective_stiffness_rejects_negative_gamma() -> None:
    m, c, k = np.eye(1), np.eye(1), np.eye(1)
    with pytest.raises(ValidationError):
        effective_stiffness(m, c, k, dt=0.01, gamma=-0.1)


def test_update_velocity_acceleration_matches_newmark_formulas() -> None:
    dt, beta, gamma = 0.01, 0.25, 0.5
    u_n = np.array([0.0])
    v_n = np.array([1.0])
    a_n = np.array([0.0])
    u_next = np.array([0.01])

    v_next, a_next = update_velocity_acceleration(u_n, v_n, a_n, u_next, dt, beta, gamma)

    expected_a_next = (
        (1.0 / (beta * dt**2)) * (u_next - u_n)
        - (1.0 / (beta * dt)) * v_n
        - (1.0 / (2 * beta) - 1.0) * a_n
    )
    expected_v_next = v_n + dt * ((1 - gamma) * a_n + gamma * expected_a_next)
    assert_allclose(a_next, expected_a_next)
    assert_allclose(v_next, expected_v_next)


def test_newmark_step_singular_effective_stiffness_raises() -> None:
    m = np.zeros((1, 1))
    c = np.zeros((1, 1))
    k = np.zeros((1, 1))
    zero = np.array([0.0])
    with pytest.raises(SingularSystemError):
        newmark_step(m, c, k, zero, zero, zero, zero, dt=0.01)


# --- SDOF validation against the analytical undamped solution ---


def _undamped_sdof_history(mass, omega, u0, dt, n_steps):
    m = np.array([[mass]])
    c = np.array([[0.0]])
    k = np.array([[mass * omega**2]])
    u = np.array([u0])
    v = np.array([0.0])
    a = np.linalg.solve(m, np.array([0.0]) - k @ u)

    times = [0.0]
    disp = [u[0]]
    for step in range(n_steps):
        u, v, a = newmark_step(m, c, k, u, v, a, np.array([0.0]), dt)
        times.append((step + 1) * dt)
        disp.append(u[0])
    return np.array(times), np.array(disp)


def test_undamped_sdof_matches_analytical_cosine() -> None:
    omega = 2 * math.pi  # 1 Hz
    dt = 0.001
    times, disp = _undamped_sdof_history(mass=1.0, omega=omega, u0=1.0, dt=dt, n_steps=2000)

    analytical = np.cos(omega * times)
    assert_allclose(disp, analytical, atol=1e-3)


def test_undamped_sdof_amplitude_does_not_decay() -> None:
    """Average-acceleration Newmark is unconditionally stable and
    introduces no artificial (numerical) damping for an undamped system.
    """
    omega = 2 * math.pi
    dt = 0.001
    _, disp = _undamped_sdof_history(mass=1.0, omega=omega, u0=1.0, dt=dt, n_steps=4000)

    assert np.abs(disp).max() <= 1.0 + 1e-6
    assert np.abs(disp).max() >= 1.0 - 1e-3


def test_undamped_sdof_period_matches_natural_frequency() -> None:
    """After exactly one period, displacement should return to u0."""
    omega = 2 * math.pi
    dt = 0.0005
    period = 2 * math.pi / omega
    n_steps = round(period / dt)
    times, disp = _undamped_sdof_history(mass=1.0, omega=omega, u0=1.0, dt=dt, n_steps=n_steps)

    assert_allclose(disp[-1], 1.0, atol=1e-3)


def test_damped_sdof_matches_analytical_solution() -> None:
    """m u'' + c u' + k u = 0, underdamped, compared against the closed-form solution."""
    mass, stiffness, damping_coeff = 1.0, 100.0, 2.0
    omega_n = math.sqrt(stiffness / mass)
    zeta = damping_coeff / (2 * math.sqrt(mass * stiffness))
    omega_d = omega_n * math.sqrt(1 - zeta**2)

    m = np.array([[mass]])
    c = np.array([[damping_coeff]])
    k = np.array([[stiffness]])
    u = np.array([1.0])
    v = np.array([0.0])
    a = np.linalg.solve(m, np.array([0.0]) - c @ v - k @ u)

    dt = 0.001
    n_steps = 3000
    times = [0.0]
    disp = [u[0]]
    for step in range(n_steps):
        u, v, a = newmark_step(m, c, k, u, v, a, np.array([0.0]), dt)
        times.append((step + 1) * dt)
        disp.append(u[0])

    times = np.array(times)
    disp = np.array(disp)
    analytical = np.exp(-zeta * omega_n * times) * (
        np.cos(omega_d * times) + (zeta * omega_n / omega_d) * np.sin(omega_d * times)
    )
    assert_allclose(disp, analytical, atol=2e-3)


def test_effective_force_matches_manual_formula() -> None:
    m = np.array([[1.0]])
    c = np.array([[0.1]])
    dt, beta, gamma = 0.01, 0.25, 0.5
    u_n = np.array([0.5])
    v_n = np.array([0.2])
    a_n = np.array([0.1])
    force_next = np.array([10.0])

    f_eff = effective_force(m, c, force_next, u_n, v_n, a_n, dt, beta, gamma)

    mass_term = (1.0 / (beta * dt**2)) * u_n + (1.0 / (beta * dt)) * v_n + (
        1.0 / (2 * beta) - 1.0
    ) * a_n
    damping_term = (gamma / (beta * dt)) * u_n + (gamma / beta - 1.0) * v_n + dt * (
        gamma / (2 * beta) - 1.0
    ) * a_n
    expected = force_next + m @ mass_term + c @ damping_term
    assert_allclose(f_eff, expected)
