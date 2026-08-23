"""Time-dependent loads: ``F(t)`` applied to one degree of freedom.

A static :class:`~femtoolkit.analysis.loads.NodalLoad` (Version 2) has a
single, fixed ``value``. A dynamic analysis instead needs the applied
force at *every* time step of a simulation -- ``F(t)``, a function of
time, not a constant. :class:`TimeDependentNodalLoad` pairs a node/DOF
(exactly like :class:`~femtoolkit.analysis.loads.NodalLoad`) with a
:class:`TimeDependentLoad` (any object with a ``value_at(t)`` method),
and :meth:`TimeDependentNodalLoad.nodal_load_at` evaluates it into an
ordinary :class:`~femtoolkit.analysis.loads.NodalLoad` at one instant --
which is how :class:`~femtoolkit.analysis.dynamic_analysis.DynamicAnalysis`
reuses the existing :func:`~femtoolkit.analysis.system.build_force_vector`
at every time step instead of duplicating force-assembly logic.

Three load-time-history shapes are provided:

* :class:`ConstantLoad` -- ``F(t) = F0`` for all ``t``.
* :class:`StepLoad` -- ``F(t) = 0`` before ``step_time``, ``F0`` at and
  after it (a suddenly applied load).
* :class:`SinusoidalLoad` -- ``F(t) = F0 * sin(omega*t + phase)``, for
  harmonic/forced-vibration loading.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from femtoolkit.analysis.dof import validate_dof
from femtoolkit.analysis.loads import NodalLoad
from femtoolkit.exceptions import ValidationError


@runtime_checkable
class TimeDependentLoad(Protocol):
    """The minimal interface a time-dependent load history must expose."""

    def value_at(self, t: float) -> float:
        """Return the load's magnitude at time ``t`` (seconds), in newtons."""
        ...


@dataclass(frozen=True)
class ConstantLoad:
    """A time-independent load, ``F(t) = F0`` for all ``t``.

    Useful for verifying that a dynamic analysis converges to the
    corresponding static response under a constant force (see the
    Version 11 static-limit validation).

    Attributes:
        magnitude: The constant force value, in newtons.

    Example:
        >>> ConstantLoad(magnitude=1000.0).value_at(t=5.0)
        1000.0
    """

    magnitude: float

    def __post_init__(self) -> None:
        """Validate ``magnitude`` immediately after construction.

        Raises:
            ValidationError: If ``magnitude`` is not finite.
        """
        if not math.isfinite(self.magnitude):
            raise ValidationError(f"ConstantLoad magnitude must be finite, got {self.magnitude}.")

    def value_at(self, t: float) -> float:
        """Return ``magnitude``, independent of ``t``."""
        return self.magnitude


@dataclass(frozen=True)
class StepLoad:
    """A suddenly applied load: zero before ``step_time``, ``magnitude`` at and after it.

    Attributes:
        magnitude: The applied force value once the step occurs, in newtons.
        step_time: The time, in seconds, at which the load switches on.
            Defaults to ``0.0`` (applied from the very start of the
            simulation).

    Raises:
        ValidationError: If ``magnitude`` is not finite, or ``step_time``
            is not finite.

    Example:
        >>> load = StepLoad(magnitude=500.0, step_time=0.1)
        >>> load.value_at(0.05), load.value_at(0.1), load.value_at(0.2)
        (0.0, 500.0, 500.0)
    """

    magnitude: float
    step_time: float = 0.0

    def __post_init__(self) -> None:
        """Validate the step load immediately after construction.

        Raises:
            ValidationError: If ``magnitude`` or ``step_time`` is not finite.
        """
        if not math.isfinite(self.magnitude):
            raise ValidationError(f"StepLoad magnitude must be finite, got {self.magnitude}.")
        if not math.isfinite(self.step_time):
            raise ValidationError(f"StepLoad step_time must be finite, got {self.step_time}.")

    def value_at(self, t: float) -> float:
        """Return ``0.0`` before ``step_time``, ``magnitude`` at and after it."""
        return self.magnitude if t >= self.step_time else 0.0


@dataclass(frozen=True)
class SinusoidalLoad:
    """A harmonic load, ``F(t) = amplitude * sin(angular_frequency * t + phase)``.

    Attributes:
        amplitude: Peak force value, ``F0``, in newtons.
        angular_frequency: Angular (circular) forcing frequency,
            ``omega``, in rad/s. Must be non-negative.
        phase: Phase offset, in radians. Defaults to ``0.0``.

    Raises:
        ValidationError: If ``amplitude`` or ``phase`` is not finite, or
            ``angular_frequency`` is negative or not finite.

    Example:
        >>> load = SinusoidalLoad(amplitude=100.0, angular_frequency=10.0)
        >>> load.value_at(t=0.0)
        0.0
    """

    amplitude: float
    angular_frequency: float
    phase: float = 0.0

    def __post_init__(self) -> None:
        """Validate the sinusoidal load immediately after construction.

        Raises:
            ValidationError: If ``amplitude`` or ``phase`` is not finite,
                or ``angular_frequency`` is negative or not finite.
        """
        if not math.isfinite(self.amplitude):
            raise ValidationError(f"SinusoidalLoad amplitude must be finite, got {self.amplitude}.")
        if not math.isfinite(self.angular_frequency) or self.angular_frequency < 0:
            raise ValidationError(
                "SinusoidalLoad angular_frequency must be non-negative, got "
                f"{self.angular_frequency}."
            )
        if not math.isfinite(self.phase):
            raise ValidationError(f"SinusoidalLoad phase must be finite, got {self.phase}.")

    def value_at(self, t: float) -> float:
        """Return ``amplitude * sin(angular_frequency * t + phase)``."""
        return self.amplitude * math.sin(self.angular_frequency * t + self.phase)


@dataclass
class TimeDependentNodalLoad:
    """A time-dependent force applied to one degree of freedom of one node.

    Attributes:
        node_id: Positive integer ID of the loaded node.
        dof: DOF direction the load acts on, see
            :class:`~femtoolkit.analysis.dof.TranslationDOF`.
        load: The time-dependent load history, e.g. a :class:`ConstantLoad`,
            :class:`StepLoad`, or :class:`SinusoidalLoad`.

    Raises:
        ValidationError: If ``node_id`` is not a positive integer, or
            ``dof`` is not a valid DOF direction.

    Example:
        >>> tdl = TimeDependentNodalLoad(
        ...     node_id=3, dof=TranslationDOF.Y, load=SinusoidalLoad(1000.0, 50.0)
        ... )
        >>> tdl.nodal_load_at(t=0.01)
        NodalLoad(node_id=3, dof=1, value=841.47...)
    """

    node_id: int
    dof: int
    load: TimeDependentLoad

    def __post_init__(self) -> None:
        """Validate the time-dependent nodal load immediately after construction.

        Raises:
            ValidationError: If ``node_id`` is not a positive integer, or
                ``dof`` is not a valid DOF direction.
        """
        if not isinstance(self.node_id, int) or isinstance(self.node_id, bool) or self.node_id <= 0:
            raise ValidationError(
                f"TimeDependentNodalLoad node_id must be a positive integer, got "
                f"{self.node_id!r}."
            )
        self.dof = validate_dof(self.dof)

    def nodal_load_at(self, t: float) -> NodalLoad:
        """Evaluate this load at time ``t``, returning an ordinary :class:`NodalLoad`.

        Args:
            t: Time, in seconds.

        Returns:
            A :class:`~femtoolkit.analysis.loads.NodalLoad` with
            ``value = self.load.value_at(t)``.
        """
        return NodalLoad(self.node_id, self.dof, self.load.value_at(t))
