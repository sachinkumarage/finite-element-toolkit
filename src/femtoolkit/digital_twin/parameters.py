"""Model parameters eligible for digital twin calibration (Version 38).

A :class:`ModelParameter` is the engineering-facing description of one
calibratable quantity (Young's modulus, density, thickness, ...): a current
(baseline) value, bounds, units, and whether it should actually be adjusted
during model updating. Converting an *updatable* parameter into a
:class:`~femtoolkit.optimization.variables.DesignVariable`
(:meth:`ModelParameter.to_design_variable`) reuses the existing Version 32
optimization machinery directly for the search itself -- this module defines
no search algorithm of its own.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from femtoolkit.exceptions import ValidationError
from femtoolkit.optimization.variables import DesignVariable, DesignVariableType


@dataclass
class ModelParameter:
    """One model parameter that may require calibration against measured data.

    Attributes:
        name: A unique, human-readable identifier (e.g. ``"youngs_modulus"``).
        current_value: The parameter's current (baseline) value, in physical units.
        lower_bound: The inclusive lower bound a calibrated value may take.
        upper_bound: The inclusive upper bound a calibrated value may take.
        units: A units string for display (e.g. ``"Pa"``).
        updatable: Whether model updating is allowed to adjust this parameter.
            A non-updatable parameter is reported alongside the updatable ones
            but is never changed by :func:`~femtoolkit.digital_twin.updating.run_model_update`.
        path: The dotted override path this parameter controls on a
            :class:`~femtoolkit.application.project.Project` (the same
            convention :class:`~femtoolkit.studies.scenarios.Scenario` uses),
            e.g. ``"material.youngs_modulus"``. Defaults to :attr:`name` when
            not given, for the common case where the two coincide.
    """

    name: str
    current_value: float
    lower_bound: float
    upper_bound: float
    units: str = ""
    updatable: bool = True
    path: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise ValidationError("ModelParameter requires a non-empty name.")
        if not self.path:
            self.path = self.name
        if not math.isfinite(self.current_value):
            raise ValidationError(
                f"ModelParameter {self.name!r} current_value must be finite, "
                f"got {self.current_value!r}."
            )
        if not (math.isfinite(self.lower_bound) and math.isfinite(self.upper_bound)):
            raise ValidationError(
                f"ModelParameter {self.name!r} bounds must be finite."
            )
        if self.lower_bound >= self.upper_bound:
            raise ValidationError(
                f"ModelParameter {self.name!r} requires lower_bound < upper_bound; got "
                f"lower_bound={self.lower_bound!r}, upper_bound={self.upper_bound!r}."
            )
        if not (self.lower_bound <= self.current_value <= self.upper_bound):
            raise ValidationError(
                f"ModelParameter {self.name!r} current_value {self.current_value!r} must lie "
                f"within [{self.lower_bound!r}, {self.upper_bound!r}]."
            )

    def to_design_variable(self) -> DesignVariable:
        """Return this parameter as a continuous
        :class:`~femtoolkit.optimization.variables.DesignVariable`.

        Returns:
            A :class:`DesignVariable` with :attr:`current_value` as its
            default/starting value.

        Raises:
            ValidationError: If this parameter is not :attr:`updatable`.
        """
        if not self.updatable:
            raise ValidationError(
                f"ModelParameter {self.name!r} is not updatable; it cannot become a "
                "design variable for calibration."
            )
        return DesignVariable(
            name=self.name,
            path=self.path,
            variable_type=DesignVariableType.CONTINUOUS,
            lower_bound=self.lower_bound,
            upper_bound=self.upper_bound,
            default_value=self.current_value,
        )


def validate_unique_parameter_names(parameters: list[ModelParameter]) -> None:
    """Check that every parameter in ``parameters`` has a unique name.

    Args:
        parameters: The parameters to check.

    Raises:
        ValidationError: If any ``name`` appears more than once.
    """
    seen: set[str] = set()
    for parameter in parameters:
        if parameter.name in seen:
            raise ValidationError(
                f"Duplicate model parameter name {parameter.name!r}; every parameter must "
                "have a unique name."
            )
        seen.add(parameter.name)


__all__ = ["ModelParameter", "validate_unique_parameter_names"]
