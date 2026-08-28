"""Modal superposition: solving ``M u'' + C u' + K u = F(t)`` via decoupled modal coordinates.

For a linear system, the physical displacement can be expressed as a
combination of mode shapes weighted by time-varying **modal
coordinates** ``q(t)``:

.. code-block:: text

    u(t) = Phi * q(t)

where ``Phi`` is the (mass-normalized) mode shape matrix -- one column
per retained mode (see :func:`~femtoolkit.analysis.modal.modal_analysis_of_system`).
Substituting into the dynamic equation and left-multiplying by ``Phi^T``
gives:

.. code-block:: text

    Phi^T M Phi q'' + Phi^T C Phi q' + Phi^T K Phi q = Phi^T F(t)

For mass-normalized mode shapes, ``Phi^T M Phi = I`` and
``Phi^T K Phi = diag(omega_i^2)`` **exactly**, by modal orthogonality
(see :mod:`femtoolkit.analysis.modal`). ``Phi^T C Phi`` is diagonal too
whenever ``C`` is proportional to ``M``/``K`` (Rayleigh damping is,
exactly); this module uses modal damping ratios directly
(:class:`~femtoolkit.analysis.damping.ModalDamping`) rather than
building a general ``C`` and hoping it happens to be proportional, so
the modal equations decouple into ``n`` **independent single-DOF
oscillators**:

.. code-block:: text

    q_i'' + 2*zeta_i*omega_i*q_i' + omega_i^2*q_i = Phi_i^T F(t)   for each mode i

Each is integrated with the exact same Newmark-beta machinery used for
a direct time-history analysis (:mod:`femtoolkit.analysis.newmark`,
:class:`~femtoolkit.analysis.dynamic_analysis.DynamicAnalysis`) --
modal superposition changes *what* is being integrated (``n_modes``
decoupled scalar equations instead of the full coupled system), not
*how*. The physical response is recovered by superposing each mode's
contribution back: ``u(t) = sum_i(phi_i * q_i(t))``.

**Why this can be much cheaper than a direct solve.** A real structure's
dynamic response is usually dominated by its lowest few modes; keeping
only those (``modes`` far smaller than the total DOF count) gives an
*approximate* but often very accurate response at a fraction of the
cost of integrating the full system -- the classic motivation for modal
superposition in structural dynamics.

**Scope.** Rigid-body modes (undefined period, no meaningful damping
ratio) are excluded automatically. Modal superposition starts from rest
(zero initial modal displacement/velocity) and requires every boundary
condition in ``system`` to prescribe exactly zero displacement --
nonzero prescribed support motion cannot be represented purely through
mode shapes (which are zero at every constrained DOF by construction)
and is out of scope for this version, matching
:class:`~femtoolkit.analysis.dynamic_analysis.DynamicAnalysis`'s own
"time-varying prescribed displacement is out of scope" limitation.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from femtoolkit.analysis.damping import ModalDamping
from femtoolkit.analysis.dynamic_loads import TimeDependentNodalLoad
from femtoolkit.analysis.dynamic_system import DynamicSystem
from femtoolkit.analysis.modal import modal_analysis_of_system
from femtoolkit.analysis.newmark import (
    DEFAULT_BETA,
    DEFAULT_GAMMA,
    effective_force,
    effective_stiffness,
    update_velocity_acceleration,
)
from femtoolkit.analysis.system import build_force_vector
from femtoolkit.exceptions import SingularSystemError, ValidationError
from femtoolkit.results.dynamic_result import DynamicResult

_ZERO_BOUNDARY_VALUE_TOLERANCE = 1e-9


def _validate_zero_boundary_conditions(system: DynamicSystem) -> None:
    for bc in system.boundary_conditions:
        if abs(bc.value) > _ZERO_BOUNDARY_VALUE_TOLERANCE:
            raise ValidationError(
                "modal_superposition requires every boundary condition to prescribe "
                f"zero displacement (node_id={bc.node_id}, dof={bc.dof} has value "
                f"{bc.value}); nonzero prescribed support motion cannot be represented "
                "through mode shapes alone."
            )


def modal_superposition(
    system: DynamicSystem,
    modes: int,
    loads: Sequence[TimeDependentNodalLoad],
    time_step: float,
    total_time: float,
    damping: ModalDamping | None = None,
    beta: float = DEFAULT_BETA,
    gamma: float = DEFAULT_GAMMA,
) -> DynamicResult:
    """Solve a dynamic system's transient response by modal superposition.

    Args:
        system: The dynamic system to analyze. Every boundary condition
            must prescribe exactly zero displacement.
        modes: Number of lowest modes to request. Rigid-body modes among
            them are dropped automatically (see the module docstring),
            so fewer than ``modes`` modes may actually be integrated;
            a :class:`~femtoolkit.exceptions.ValidationError` is raised
            only if *none* of the requested modes are physical.
        loads: Time-dependent nodal loads making up ``F(t)``, reusing
            the exact same
            :class:`~femtoolkit.analysis.dynamic_loads.TimeDependentNodalLoad`
            abstraction as
            :class:`~femtoolkit.analysis.dynamic_analysis.DynamicAnalysis`.
        time_step: Fixed time step, in seconds. Must be positive.
        total_time: Total simulated duration, in seconds. Must be
            positive; the number of steps is ``round(total_time / time_step)``.
        damping: Modal damping ratios to apply, or ``None`` (the
            default) to derive each retained mode's damping coefficient
            directly from ``system.damping`` (``phi_i^T * C * phi_i``,
            exact when ``system.damping`` is proportional, e.g.
            :class:`~femtoolkit.analysis.damping.RayleighDamping`, and
            automatically zero for an undamped system).
        beta: Newmark beta parameter. Defaults to ``0.25`` (average
            acceleration method).
        gamma: Newmark gamma parameter. Defaults to ``0.5``.

    Returns:
        A :class:`~femtoolkit.results.dynamic_result.DynamicResult`, in
        the same full (unreduced) DOF space and with the same
        ``time``/``*_history`` shapes a direct
        :meth:`~femtoolkit.analysis.dynamic_analysis.DynamicAnalysis.solve`
        call would return, so both solution methods are directly
        comparable.

    Raises:
        ValidationError: If ``time_step``/``total_time`` are not
            positive, ``modes`` is not a positive integer, any boundary
            condition prescribes a nonzero value, or every requested
            mode is a rigid-body mode.
        EigenvalueComputationError: See
            :func:`~femtoolkit.analysis.modal.natural_frequencies`.
        SingularSystemError: If a modal Newmark effective stiffness
            "matrix" (scalar, per mode) is singular -- not expected for
            a physical mode with ``omega > 0``, but guarded against
            consistently with :class:`~femtoolkit.analysis.dynamic_analysis.DynamicAnalysis`.
    """
    if not isinstance(modes, int) or isinstance(modes, bool) or modes <= 0:
        raise ValidationError(f"modes must be a positive integer, got {modes!r}.")
    if not math.isfinite(time_step) or time_step <= 0:
        raise ValidationError(f"time_step must be positive, got {time_step}.")
    if not math.isfinite(total_time) or total_time <= 0:
        raise ValidationError(f"total_time must be positive, got {total_time}.")
    _validate_zero_boundary_conditions(system)

    modal_result = modal_analysis_of_system(system, num_modes=modes)
    physical_mask = ~modal_result.is_rigid_body_mode
    if not np.any(physical_mask):
        raise ValidationError(
            f"All {modes} requested mode(s) are rigid-body modes; modal superposition "
            "requires at least one physical vibration mode."
        )

    phi = modal_result.mass_normalized_mode_shapes[:, physical_mask]
    omega = modal_result.angular_frequencies[physical_mask]
    n_modes = phi.shape[1]

    if damping is not None:
        c_modal = damping.modal_damping_coefficients(omega)
    else:
        c_modal = np.array([float(phi[:, i] @ system.damping @ phi[:, i]) for i in range(n_modes)])

    mass_modal = np.eye(n_modes)
    damping_modal = np.diag(c_modal)
    stiffness_modal = np.diag(omega**2)

    dof_map = system.dof_map
    total_dofs = dof_map.total_dofs
    n_steps = round(total_time / time_step)
    time = np.array([step * time_step for step in range(n_steps + 1)])

    def force_at(t: float) -> np.ndarray:
        nodal_loads = [load.nodal_load_at(t) for load in loads]
        return build_force_vector(dof_map, nodal_loads)

    def modal_force_at(t: float) -> np.ndarray:
        return phi.T @ force_at(t)

    q = np.zeros(n_modes)
    q_dot = np.zeros(n_modes)
    rhs_0 = modal_force_at(0.0) - damping_modal @ q_dot - stiffness_modal @ q
    q_ddot = np.linalg.solve(mass_modal, rhs_0)

    displacement_history = np.zeros((n_steps + 1, total_dofs))
    velocity_history = np.zeros((n_steps + 1, total_dofs))
    acceleration_history = np.zeros((n_steps + 1, total_dofs))
    reaction_history = np.zeros((n_steps + 1, total_dofs))

    def record(
        step: int, q_vec: np.ndarray, q_dot_vec: np.ndarray, q_ddot_vec: np.ndarray, t: float
    ) -> None:
        u = phi @ q_vec
        v = phi @ q_dot_vec
        a = phi @ q_ddot_vec
        force_full = force_at(t)
        reaction = system.mass @ a + system.damping @ v + system.stiffness @ u - force_full
        displacement_history[step] = u
        velocity_history[step] = v
        acceleration_history[step] = a
        reaction_history[step] = reaction

    record(0, q, q_dot, q_ddot, 0.0)

    k_eff_modal = effective_stiffness(
        mass_modal, damping_modal, stiffness_modal, time_step, beta, gamma
    )

    for step in range(1, n_steps + 1):
        t_next = time[step]
        f_eff = effective_force(
            mass_modal,
            damping_modal,
            modal_force_at(t_next),
            q,
            q_dot,
            q_ddot,
            time_step,
            beta,
            gamma,
        )
        try:
            q_next = np.linalg.solve(k_eff_modal, f_eff)
        except np.linalg.LinAlgError as error:
            raise SingularSystemError(
                "The modal Newmark effective stiffness is singular; check that every "
                "retained mode has a positive natural frequency."
            ) from error

        q_dot_next, q_ddot_next = update_velocity_acceleration(
            q, q_dot, q_ddot, q_next, time_step, beta, gamma
        )
        record(step, q_next, q_dot_next, q_ddot_next, t_next)
        q, q_dot, q_ddot = q_next, q_dot_next, q_ddot_next

    return DynamicResult(
        dof_map=dof_map,
        time=time,
        displacement_history=displacement_history,
        velocity_history=velocity_history,
        acceleration_history=acceleration_history,
        reaction_history=reaction_history,
    )
