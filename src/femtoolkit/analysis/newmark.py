"""The Newmark-beta method: implicit time integration of ``M u'' + C u' + K u = F(t)``.

Given the state at time ``t_n`` (displacement ``u_n``, velocity ``v_n``,
acceleration ``a_n``) and the force at ``t_{n+1} = t_n + dt``, the
Newmark-beta method assumes the acceleration varies linearly over the
step (via the two free parameters ``beta`` and ``gamma``) and derives:

.. code-block:: text

    u_{n+1} = u_n + dt*v_n + dt^2 * [ (0.5-beta)*a_n + beta*a_{n+1} ]
    v_{n+1} = v_n + dt * [ (1-gamma)*a_n + gamma*a_{n+1} ]

Substituting these into ``M u''_{n+1} + C u'_{n+1} + K u_{n+1} = F_{n+1}``
and solving for ``u_{n+1}`` first (the only truly unknown quantity) gives
an **effective static problem** at every time step:

.. code-block:: text

    K_eff u_{n+1} = F_eff

    K_eff = K + (gamma/(beta*dt)) * C + (1/(beta*dt^2)) * M

    F_eff = F_{n+1}
          + M * [ (1/(beta*dt^2))*u_n + (1/(beta*dt))*v_n + (1/(2*beta) - 1)*a_n ]
          + C * [ (gamma/(beta*dt))*u_n + (gamma/beta - 1)*v_n
                  + dt*(gamma/(2*beta) - 1)*a_n ]

``u_{n+1}`` is recovered from this ordinary linear solve, then
``a_{n+1}`` and ``v_{n+1}`` follow directly from the two update formulas
above, rearranged for the new unknowns:

.. code-block:: text

    a_{n+1} = (1/(beta*dt^2))*(u_{n+1}-u_n) - (1/(beta*dt))*v_n - (1/(2*beta)-1)*a_n
    v_{n+1} = v_n + dt*[ (1-gamma)*a_n + gamma*a_{n+1} ]

The standard parameters ``beta = 1/4, gamma = 1/2`` (the **average
acceleration method**, used because it is unconditionally stable for a
linear system regardless of time step size -- unlike, e.g., the
central-difference method) are this module's default and the only
combination Version 11 exposes; see
:mod:`femtoolkit.analysis.dynamic_analysis` for the module that drives
this one step at a time across a full simulation, applying boundary
conditions via the same free/constrained DOF partition used by the
static solver.
"""

from __future__ import annotations

import math

import numpy as np

from femtoolkit.exceptions import SingularSystemError, ValidationError

DEFAULT_BETA: float = 0.25
"""Newmark beta parameter for the average-acceleration method."""

DEFAULT_GAMMA: float = 0.5
"""Newmark gamma parameter for the average-acceleration method."""


def _validate_newmark_parameters(dt: float, beta: float, gamma: float) -> None:
    if not math.isfinite(dt) or dt <= 0:
        raise ValidationError(f"Newmark time step dt must be positive, got {dt}.")
    if not math.isfinite(beta) or beta <= 0:
        raise ValidationError(f"Newmark beta must be positive, got {beta}.")
    if not math.isfinite(gamma) or gamma < 0:
        raise ValidationError(f"Newmark gamma must be non-negative, got {gamma}.")


def effective_stiffness(
    mass: np.ndarray,
    damping: np.ndarray,
    stiffness: np.ndarray,
    dt: float,
    beta: float = DEFAULT_BETA,
    gamma: float = DEFAULT_GAMMA,
) -> np.ndarray:
    """Compute the Newmark effective stiffness, ``K + (gamma/(beta*dt))*C + (1/(beta*dt^2))*M``.

    Constant across an entire simulation as long as ``mass``,
    ``damping``, ``stiffness``, ``dt``, ``beta``, and ``gamma`` do not
    change -- callers that run many time steps should compute this once
    and reuse it (see :class:`~femtoolkit.analysis.dynamic_analysis.DynamicAnalysis`),
    rather than recomputing it every step.

    Args:
        mass: Mass matrix ``M``.
        damping: Damping matrix ``C``, the same shape as ``mass``.
        stiffness: Stiffness matrix ``K``, the same shape as ``mass``.
        dt: Time step, in seconds. Must be positive.
        beta: Newmark beta parameter. Must be positive.
        gamma: Newmark gamma parameter. Must be non-negative.

    Returns:
        A NumPy array the same shape as ``mass``.

    Raises:
        ValidationError: If ``dt`` is not positive, ``beta`` is not
            positive, or ``gamma`` is negative.
    """
    _validate_newmark_parameters(dt, beta, gamma)
    return stiffness + (gamma / (beta * dt)) * damping + (1.0 / (beta * dt**2)) * mass


def effective_force(
    mass: np.ndarray,
    damping: np.ndarray,
    force_next: np.ndarray,
    u_n: np.ndarray,
    v_n: np.ndarray,
    a_n: np.ndarray,
    dt: float,
    beta: float = DEFAULT_BETA,
    gamma: float = DEFAULT_GAMMA,
) -> np.ndarray:
    """Compute the Newmark effective force vector at ``t_{n+1}``.

    Args:
        mass: Mass matrix ``M``.
        damping: Damping matrix ``C``.
        force_next: Applied force vector at ``t_{n+1}``, ``F_{n+1}``.
        u_n: Displacement at ``t_n``.
        v_n: Velocity at ``t_n``.
        a_n: Acceleration at ``t_n``.
        dt: Time step, in seconds. Must be positive.
        beta: Newmark beta parameter. Must be positive.
        gamma: Newmark gamma parameter. Must be non-negative.

    Returns:
        The effective force vector ``F_eff``, the same shape as ``force_next``.

    Raises:
        ValidationError: If ``dt`` is not positive, ``beta`` is not
            positive, or ``gamma`` is negative.
    """
    _validate_newmark_parameters(dt, beta, gamma)
    mass_term = (1.0 / (beta * dt**2)) * u_n + (1.0 / (beta * dt)) * v_n + (
        1.0 / (2.0 * beta) - 1.0
    ) * a_n
    damping_term = (gamma / (beta * dt)) * u_n + (gamma / beta - 1.0) * v_n + dt * (
        gamma / (2.0 * beta) - 1.0
    ) * a_n
    return force_next + mass @ mass_term + damping @ damping_term


def update_velocity_acceleration(
    u_n: np.ndarray,
    v_n: np.ndarray,
    a_n: np.ndarray,
    u_next: np.ndarray,
    dt: float,
    beta: float = DEFAULT_BETA,
    gamma: float = DEFAULT_GAMMA,
) -> tuple[np.ndarray, np.ndarray]:
    """Recover ``v_{n+1}``/``a_{n+1}`` once ``u_{n+1}`` is known.

    Args:
        u_n: Displacement at ``t_n``.
        v_n: Velocity at ``t_n``.
        a_n: Acceleration at ``t_n``.
        u_next: Displacement at ``t_{n+1}`` (from solving the effective
            static problem, see :func:`effective_stiffness`/:func:`effective_force`).
        dt: Time step, in seconds. Must be positive.
        beta: Newmark beta parameter. Must be positive.
        gamma: Newmark gamma parameter. Must be non-negative.

    Returns:
        ``(v_next, a_next)``.

    Raises:
        ValidationError: If ``dt`` is not positive, ``beta`` is not
            positive, or ``gamma`` is negative.
    """
    _validate_newmark_parameters(dt, beta, gamma)
    a_next = (
        (1.0 / (beta * dt**2)) * (u_next - u_n)
        - (1.0 / (beta * dt)) * v_n
        - (1.0 / (2.0 * beta) - 1.0) * a_n
    )
    v_next = v_n + dt * ((1.0 - gamma) * a_n + gamma * a_next)
    return v_next, a_next


def newmark_step(
    mass: np.ndarray,
    damping: np.ndarray,
    stiffness: np.ndarray,
    u_n: np.ndarray,
    v_n: np.ndarray,
    a_n: np.ndarray,
    force_next: np.ndarray,
    dt: float,
    beta: float = DEFAULT_BETA,
    gamma: float = DEFAULT_GAMMA,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Advance the dynamic system by one Newmark-beta time step.

    A self-contained convenience wrapper around
    :func:`effective_stiffness`, :func:`effective_force`, and
    :func:`update_velocity_acceleration` -- recomputes the effective
    stiffness matrix every call, so it is best suited to standalone use
    (e.g. a single-DOF oscillator) rather than a long multi-step
    simulation, where :class:`~femtoolkit.analysis.dynamic_analysis.DynamicAnalysis`
    computes it once and reuses it.

    Args:
        mass: Mass matrix ``M``.
        damping: Damping matrix ``C``.
        stiffness: Stiffness matrix ``K``.
        u_n: Displacement at ``t_n``.
        v_n: Velocity at ``t_n``.
        a_n: Acceleration at ``t_n``.
        force_next: Applied force vector at ``t_{n+1}``.
        dt: Time step, in seconds. Must be positive.
        beta: Newmark beta parameter. Defaults to ``0.25`` (average
            acceleration method).
        gamma: Newmark gamma parameter. Defaults to ``0.5``.

    Returns:
        ``(u_next, v_next, a_next)``.

    Raises:
        ValidationError: If ``dt`` is not positive, ``beta`` is not
            positive, or ``gamma`` is negative.
        SingularSystemError: If the effective stiffness matrix is singular.
    """
    k_eff = effective_stiffness(mass, damping, stiffness, dt, beta, gamma)
    f_eff = effective_force(mass, damping, force_next, u_n, v_n, a_n, dt, beta, gamma)

    try:
        u_next = np.linalg.solve(k_eff, f_eff)
    except np.linalg.LinAlgError as error:
        raise SingularSystemError(
            "The Newmark effective stiffness matrix is singular; check that the "
            "dynamic system is properly constrained."
        ) from error

    v_next, a_next = update_velocity_acceleration(u_n, v_n, a_n, u_next, dt, beta, gamma)
    return u_next, v_next, a_next
