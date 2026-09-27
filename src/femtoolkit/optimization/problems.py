"""`OptimizationProblem`: a problem definition, separate from any algorithm (Version 32).

Deliberately holds only the *definition* of what to optimize -- design
variables, objectives, constraints, the base project -- and nothing
about *how* to search the design space. That is
:mod:`femtoolkit.optimization.algorithms`'s job; a problem is defined
once and can be handed to any algorithm.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from femtoolkit.application.project import Project
from femtoolkit.exceptions import ValidationError
from femtoolkit.optimization.constraints import Constraint, validate_unique_constraint_names
from femtoolkit.optimization.objectives import Objective, validate_unique_objective_names
from femtoolkit.optimization.variables import DesignVariable, validate_unique_variable_names


class OptimizationMode(Enum):
    """Whether a problem is evaluated deterministically or with uncertainty awareness.

    Attributes:
        DETERMINISTIC: Every design is evaluated by exactly one FEA run
            (the mode this version's algorithms implement).
        UNCERTAINTY_AWARE: An objective or constraint may internally
            run a Version 31 Monte Carlo study to evaluate a design
            (e.g. :func:`~femtoolkit.optimization.objectives.robust_objective_mean`)
            -- a lightweight, optional building block toward robust
            optimization, not a robust-optimization algorithm itself
            (see the Version 33 preview in ``README.md``). Declaring
            this mode does not change how
            :mod:`femtoolkit.optimization.algorithms` searches the
            design space; it is informational, surfaced in reports so
            a reader knows a "single" evaluation may itself represent
            many FEA runs.
    """

    DETERMINISTIC = "deterministic"
    UNCERTAINTY_AWARE = "uncertainty_aware"


@dataclass
class OptimizationProblem:
    """The complete definition of one engineering optimization problem.

    Attributes:
        name: A short, human-readable name.
        base_project: The unmodified base project every design overrides.
        design_variables: The parameters the optimization may change.
            Must be non-empty.
        objectives: What the optimization is trying to minimize/maximize.
            Must be non-empty; more than one makes this a multi-objective
            problem (see :func:`~femtoolkit.optimization.pareto.pareto_front`).
        constraints: The acceptable-region boundaries every design is
            checked against. May be empty (an unconstrained problem).
        mode: Whether this problem is deterministic or uncertainty-aware
            (see :class:`OptimizationMode`).
        description: A longer, free-text description.
    """

    name: str
    base_project: Project
    design_variables: list[DesignVariable]
    objectives: list[Objective]
    constraints: list[Constraint] = field(default_factory=list)
    mode: OptimizationMode = OptimizationMode.DETERMINISTIC
    description: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise ValidationError("OptimizationProblem requires a non-empty name.")
        if not self.design_variables:
            raise ValidationError(
                f"OptimizationProblem {self.name!r} requires at least one design variable."
            )
        if not self.objectives:
            raise ValidationError(
                f"OptimizationProblem {self.name!r} requires at least one objective."
            )
        validate_unique_variable_names(self.design_variables)
        validate_unique_objective_names(self.objectives)
        validate_unique_constraint_names(self.constraints)

    @property
    def is_multi_objective(self) -> bool:
        """Whether this problem has more than one objective."""
        return len(self.objectives) > 1

    def default_values(self) -> dict[str, Any]:
        """Every design variable's default value, keyed by name."""
        return {variable.name: variable.default_value for variable in self.design_variables}

    def variable(self, name: str) -> DesignVariable:
        """Look up a design variable by name.

        Raises:
            ValidationError: If no design variable has that name.
        """
        for variable in self.design_variables:
            if variable.name == name:
                return variable
        raise ValidationError(f"OptimizationProblem {self.name!r} has no design variable {name!r}.")

    def objective(self, name: str) -> Objective:
        """Look up an objective by name.

        Raises:
            ValidationError: If no objective has that name.
        """
        for objective in self.objectives:
            if objective.name == name:
                return objective
        raise ValidationError(f"OptimizationProblem {self.name!r} has no objective {name!r}.")


__all__ = ["OptimizationMode", "OptimizationProblem"]
