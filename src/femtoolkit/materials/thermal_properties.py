"""Temperature-dependent material properties (Version 19).

Real materials rarely have perfectly constant elastic constants: Young's
modulus, Poisson's ratio, and the thermal expansion coefficient all
generally soften or shift somewhat with temperature. This module gives
:class:`~femtoolkit.materials.thermoelastic.ThermoelasticMaterial3D` (and
any future material that needs it) a single, minimal way to express that,
without overcomplicating the common case where a property genuinely is
constant.

A property is represented as a :data:`ThermalProperty`: either a plain
``float`` (constant, independent of temperature) or any
``Callable[[float], float]`` (evaluated at the current temperature to get
the property's value). :class:`TemperatureDependentProperty` is one
concrete, ready-to-use callable: a **tabulated** property, linearly
interpolated between given ``(temperature, value)`` pairs -- but a caller
is equally free to pass their own arbitrary function (e.g. a closed-form
``lambda T: E0 * (1 - beta * (T - 293.15))``) instead, since both are
just plain Python callables from this module's point of view.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from femtoolkit.exceptions import ValidationError

ThermalProperty = float | Callable[[float], float]
"""A material property that is either a constant ``float`` or a
``Callable[[float], float]`` (such as :class:`TemperatureDependentProperty`,
or any user-defined function) mapping temperature (kelvin) to the
property's value at that temperature."""


def evaluate_thermal_property(value: ThermalProperty, temperature: float) -> float:
    """Evaluate a :data:`ThermalProperty` at a given temperature.

    Args:
        value: A constant ``float``, or a callable mapping temperature
            to the property's value.
        temperature: The temperature to evaluate at, in kelvin.

    Returns:
        The property's value at ``temperature``.
    """
    if callable(value):
        return float(value(temperature))
    return float(value)


@dataclass(frozen=True)
class TemperatureDependentProperty:
    """A material property given as a temperature-vs-value lookup table.

    Evaluated by linear interpolation between the tabulated points.
    Calling an instance outside the tabulated temperature range raises
    :class:`~femtoolkit.exceptions.ValidationError` rather than silently
    extrapolating -- a property extrapolated far past the range it was
    actually measured over is not a validated value, and silently
    producing one is exactly the kind of quiet, hard-to-notice error this
    project's "mathematical/constitutive correctness first" priority
    warns against.

    Attributes:
        temperatures: Strictly increasing temperatures, in kelvin, at
            which ``values`` are given. Must have at least two points.
        values: The property's value at each corresponding temperature
            in ``temperatures`` (same length).

    Raises:
        ValidationError: If ``temperatures`` and ``values`` have
            different lengths, ``temperatures`` has fewer than two
            points, any value is non-finite, or ``temperatures`` is not
            strictly increasing.

    Example:
        >>> youngs_modulus = TemperatureDependentProperty(
        ...     temperatures=(293.15, 373.15, 473.15),
        ...     values=(200e9, 195e9, 185e9),
        ... )
        >>> youngs_modulus(333.15)
        197500000000.0
    """

    temperatures: tuple[float, ...]
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        """Validate the lookup table immediately after construction.

        Raises:
            ValidationError: If ``temperatures``/``values`` are
                mismatched in length, too short, contain a non-finite
                entry, or ``temperatures`` is not strictly increasing.
        """
        if len(self.temperatures) != len(self.values):
            raise ValidationError(
                "TemperatureDependentProperty temperatures and values must have the same "
                f"length, got {len(self.temperatures)} and {len(self.values)}."
            )
        if len(self.temperatures) < 2:
            raise ValidationError(
                "TemperatureDependentProperty requires at least two tabulated points, got "
                f"{len(self.temperatures)}."
            )
        if not all(math.isfinite(t) for t in self.temperatures) or not all(
            math.isfinite(v) for v in self.values
        ):
            raise ValidationError(
                "TemperatureDependentProperty temperatures and values must all be finite."
            )
        if any(
            self.temperatures[i] >= self.temperatures[i + 1]
            for i in range(len(self.temperatures) - 1)
        ):
            raise ValidationError(
                "TemperatureDependentProperty temperatures must be strictly increasing, got "
                f"{self.temperatures!r}."
            )

    def __call__(self, temperature: float) -> float:
        """Return the linearly-interpolated property value at ``temperature``.

        Args:
            temperature: The temperature to evaluate at, in kelvin.

        Returns:
            The interpolated property value.

        Raises:
            ValidationError: If ``temperature`` is not finite, or lies
                outside ``[temperatures[0], temperatures[-1]]``.
        """
        if not math.isfinite(temperature):
            raise ValidationError(f"Temperature must be finite, got {temperature}.")
        if temperature < self.temperatures[0] or temperature > self.temperatures[-1]:
            raise ValidationError(
                f"Temperature {temperature} is outside the tabulated range "
                f"[{self.temperatures[0]}, {self.temperatures[-1]}]."
            )
        return float(np.interp(temperature, self.temperatures, self.values))
