"""Design variables: what an engineer is allowed to change (Version 32).

**Engineering concept.** A *design variable* is a parameter the
optimization is allowed to change -- a plate thickness, a beam
diameter, an applied load, a material choice. It is defined by a
dotted override path, the same convention
:class:`~femtoolkit.studies.scenarios.Scenario`/
:class:`~femtoolkit.studies.parameter_sweep.ParameterDefinition`
(Version 30) already use to locate a project field, so a design
variable is a drop-in replacement for a deterministic parameter-sweep
value. Only parameters that can safely override a project field this
way should become design variables -- this module does not attempt to
validate that a given override path is physically meaningful, only
that the *value itself* stays within the declared domain.

**The design space.** For variables ``x1 in [a1, b1]``, ``x2 in [a2,
b2]``, ..., the design space is the Cartesian product of every
variable's domain -- the set of all allowable combinations an
optimization algorithm searches over. Optimization never modifies
solver internals directly; it only ever proposes a point in this space
and asks the existing simulation pipeline to evaluate it (see
:mod:`femtoolkit.optimization.evaluation`).

Three variable types are supported, matching the common engineering
cases:

- **Continuous** -- ``x in [lower, upper]`` (e.g. a plate thickness in
  meters).
- **Integer** -- ``n in {lower, lower+1, ..., upper}`` (e.g. a count of
  stiffeners).
- **Categorical** -- one of a fixed list of values (e.g. a material
  name), only usable where the underlying project field genuinely
  accepts a discrete choice (see ``docs/optimization.md`` for the
  current support boundary).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from femtoolkit.exceptions import ValidationError


class DesignVariableType(Enum):
    """The kind of domain a design variable ranges over.

    Attributes:
        CONTINUOUS: A real-valued interval ``[lower_bound, upper_bound]``.
        INTEGER: An integer interval ``{lower_bound, ..., upper_bound}``.
        CATEGORICAL: One of a fixed, finite list of values.
    """

    CONTINUOUS = "continuous"
    INTEGER = "integer"
    CATEGORICAL = "categorical"


@dataclass
class DesignVariable:
    """One parameter an optimization is allowed to change.

    Attributes:
        name: A unique, human-readable identifier (used as the key in
            every design-variable-values dict throughout this package).
        path: The dotted override path this variable controls on a
            :class:`~femtoolkit.application.project.Project` (the same
            convention :class:`~femtoolkit.studies.scenarios.Scenario`
            uses), e.g. ``"mesh.thickness"``.
        variable_type: Continuous, integer, or categorical.
        lower_bound: The inclusive lower bound (continuous/integer only).
        upper_bound: The inclusive upper bound (continuous/integer only).
        categories: The allowed values (categorical only).
        default_value: The value used as this variable's starting point
            (e.g. for the baseline design and coordinate search's
            starting point). Defaults to the domain midpoint
            (continuous), lower bound (integer), or first category
            (categorical) if not given.
        units: A units string for display, purely descriptive.
        description: A longer, free-text description.
    """

    name: str
    path: str
    variable_type: DesignVariableType
    lower_bound: float | None = None
    upper_bound: float | None = None
    categories: list[Any] | None = None
    default_value: Any = None
    units: str = ""
    description: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise ValidationError("DesignVariable requires a non-empty name.")

        if self.variable_type is DesignVariableType.CONTINUOUS:
            self._validate_bounds(strict=True)
        elif self.variable_type is DesignVariableType.INTEGER:
            self._validate_bounds(strict=False)
            if not float(self.lower_bound).is_integer() or not float(self.upper_bound).is_integer():
                raise ValidationError(
                    f"Integer design variable {self.name!r} requires integer bounds, got "
                    f"[{self.lower_bound}, {self.upper_bound}]."
                )
            self.lower_bound = int(self.lower_bound)
            self.upper_bound = int(self.upper_bound)
        else:
            if not self.categories:
                raise ValidationError(
                    f"Categorical design variable {self.name!r} requires a non-empty "
                    "list of categories."
                )
            if len(self.categories) != len(set(self.categories)):
                raise ValidationError(
                    f"Categorical design variable {self.name!r} has duplicate categories."
                )

        if self.default_value is None:
            self.default_value = self._domain_midpoint()
        elif not self.is_valid_value(self.default_value):
            raise ValidationError(
                f"Default value {self.default_value!r} for design variable {self.name!r} "
                "is outside its domain."
            )

    def _validate_bounds(self, strict: bool) -> None:
        if self.lower_bound is None or self.upper_bound is None:
            raise ValidationError(
                f"Design variable {self.name!r} requires both lower_bound and upper_bound."
            )
        if not (math.isfinite(self.lower_bound) and math.isfinite(self.upper_bound)):
            raise ValidationError(f"Design variable {self.name!r} bounds must be finite.")
        valid = (
            self.lower_bound < self.upper_bound
            if strict
            else self.lower_bound <= self.upper_bound
        )
        if not valid:
            operator = "<" if strict else "<="
            raise ValidationError(
                f"Design variable {self.name!r} requires lower_bound {operator} upper_bound "
                f"(got lower={self.lower_bound}, upper={self.upper_bound})."
            )

    def _domain_midpoint(self) -> Any:
        if self.variable_type is DesignVariableType.CONTINUOUS:
            return (self.lower_bound + self.upper_bound) / 2.0
        if self.variable_type is DesignVariableType.INTEGER:
            return (self.lower_bound + self.upper_bound) // 2
        return self.categories[0]

    def is_valid_value(self, value: Any) -> bool:
        """Whether ``value`` lies within this variable's domain.

        Args:
            value: A candidate value.

        Returns:
            ``True`` if ``value`` is a valid member of this variable's domain.
        """
        if self.variable_type is DesignVariableType.CONTINUOUS:
            if isinstance(value, bool) or not isinstance(value, int | float):
                return False
            return math.isfinite(value) and self.lower_bound <= value <= self.upper_bound
        if self.variable_type is DesignVariableType.INTEGER:
            if isinstance(value, bool) or not isinstance(value, int):
                return False
            return self.lower_bound <= value <= self.upper_bound
        return value in self.categories

    def clip(self, value: Any) -> Any:
        """Clamp ``value`` into this variable's domain.

        Args:
            value: A candidate value (continuous/integer only; a
                categorical value must already be a valid category).

        Returns:
            ``value`` clamped to ``[lower_bound, upper_bound]``, or
            rounded to the nearest integer bound for an integer
            variable. Returns ``value`` unchanged for a categorical
            variable.

        Raises:
            ValidationError: If a categorical ``value`` is not one of
                :attr:`categories`.
        """
        if self.variable_type is DesignVariableType.CONTINUOUS:
            return min(max(float(value), self.lower_bound), self.upper_bound)
        if self.variable_type is DesignVariableType.INTEGER:
            return int(min(max(round(value), self.lower_bound), self.upper_bound))
        if value not in self.categories:
            raise ValidationError(
                f"{value!r} is not one of design variable {self.name!r}'s categories: "
                f"{self.categories!r}."
            )
        return value

    def sample(self, rng: np.random.Generator) -> Any:
        """Draw one value uniformly at random from this variable's domain.

        Args:
            rng: The random generator to draw from.

        Returns:
            A random value within the domain.
        """
        if self.variable_type is DesignVariableType.CONTINUOUS:
            return float(rng.uniform(self.lower_bound, self.upper_bound))
        if self.variable_type is DesignVariableType.INTEGER:
            return int(rng.integers(self.lower_bound, self.upper_bound + 1))
        index = int(rng.integers(0, len(self.categories)))
        return self.categories[index]


def validate_unique_variable_names(design_variables: list[DesignVariable]) -> None:
    """Check that every design variable in ``design_variables`` has a unique name.

    Args:
        design_variables: The design variables to check.

    Raises:
        ValidationError: If any ``name`` appears more than once.
    """
    seen: set[str] = set()
    for variable in design_variables:
        if variable.name in seen:
            raise ValidationError(
                f"Duplicate design variable name {variable.name!r}; every design variable "
                "must have a unique name."
            )
        seen.add(variable.name)


__all__ = ["DesignVariable", "DesignVariableType", "validate_unique_variable_names"]
