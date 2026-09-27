"""Constraints: the acceptable engineering region (Version 32).

A constraint defines a relation an evaluated quantity must satisfy --
``sigma_max <= sigma_allow``, ``u_max <= u_allow``, ``m <= m_max`` -- as
distinct from an objective (:mod:`femtoolkit.optimization.objectives`):
an objective is what the optimization is trying to improve, a
constraint is a line it may not cross. For a constraint ``g(x) <= limit``
(or ``>=``/``==``), the **violation** is a non-negative measure of how
far a value falls outside the allowed region:

.. code-block:: text

    v = max(0, value - limit)          (<=)
    v = max(0, limit - value)          (>=)
    v = max(0, |value - limit| - tol)  (==)

Every constraint's violation is tracked independently -- this module
never collapses multiple constraints into one opaque combined score.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

from femtoolkit.exceptions import ValidationError
from femtoolkit.optimization.context import DesignContext

ConstraintFunction = Callable[[DesignContext], float | None]

DEFAULT_CONSTRAINT_TOLERANCE = 1e-9


class ConstraintRelation(Enum):
    """The relation a constraint's evaluated value must satisfy against its limit."""

    LESS_EQUAL = "<="
    GREATER_EQUAL = ">="
    EQUAL = "=="


@dataclass
class Constraint:
    """One acceptable-region boundary a design must satisfy.

    Attributes:
        name: A unique, human-readable identifier.
        evaluate: A callable computing this constraint's evaluated
            quantity from a
            :class:`~femtoolkit.optimization.context.DesignContext`,
            returning ``None`` if it cannot be computed.
        relation: ``<=``, ``>=``, or ``==``.
        limit: The right-hand-side limit value.
        units: A units string for display.
        description: A longer, free-text description.
        tolerance: A small allowance before a value counts as
            violating the relation (always ``>= 0``); for ``==``, the
            half-width of the accepted band around ``limit``.
    """

    name: str
    evaluate: ConstraintFunction
    relation: ConstraintRelation
    limit: float
    units: str = ""
    description: str = ""
    tolerance: float = DEFAULT_CONSTRAINT_TOLERANCE

    def __post_init__(self) -> None:
        if not self.name:
            raise ValidationError("Constraint requires a non-empty name.")
        if self.tolerance < 0:
            raise ValidationError(f"Constraint {self.name!r} tolerance must be non-negative.")

    def violation(self, value: float) -> float:
        """The non-negative violation magnitude for an evaluated ``value``.

        Args:
            value: The constraint's evaluated quantity.

        Returns:
            ``0.0`` if ``value`` satisfies the relation (within
            tolerance); otherwise a positive number measuring how far
            it falls outside.
        """
        if self.relation is ConstraintRelation.LESS_EQUAL:
            return max(0.0, value - self.limit - self.tolerance)
        if self.relation is ConstraintRelation.GREATER_EQUAL:
            return max(0.0, self.limit - value - self.tolerance)
        return max(0.0, abs(value - self.limit) - self.tolerance)

    def is_satisfied(self, value: float) -> bool:
        """Whether ``value`` satisfies this constraint (violation is zero)."""
        return self.violation(value) <= 0.0


def validate_unique_constraint_names(constraints: list[Constraint]) -> None:
    """Check that every constraint in ``constraints`` has a unique name.

    Args:
        constraints: The constraints to check.

    Raises:
        ValidationError: If any ``name`` appears more than once.
    """
    seen: set[str] = set()
    for constraint in constraints:
        if constraint.name in seen:
            raise ValidationError(
                f"Duplicate constraint name {constraint.name!r}; every constraint must "
                "have a unique name."
            )
        seen.add(constraint.name)


__all__ = [
    "DEFAULT_CONSTRAINT_TOLERANCE",
    "Constraint",
    "ConstraintFunction",
    "ConstraintRelation",
    "validate_unique_constraint_names",
]
