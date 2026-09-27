"""Design evaluation: variables -> scenario -> FEA run -> objectives/constraints (Version 32).

.. code-block:: text

    Design Variables
            |
            v
    Build Scenario (femtoolkit.studies.scenarios)
            |
            v
    Run FEA (femtoolkit.runs.manager.SimulationRunManager -- unchanged)
            |
            v
    Extract Results
            |
            v
    Calculate Objectives
            |
            v
    Evaluate Constraints
            |
            v
    DesignEvaluation

**This module performs no FEA computation and no scenario/run
bookkeeping of its own.** A design's variable values become a
:class:`~femtoolkit.studies.scenarios.Scenario` override the exact same
way a Version 30 parameter-sweep value or a Version 31 Monte Carlo
sample does, and execution is delegated entirely to
:class:`~femtoolkit.runs.manager.SimulationRunManager` -- there is
exactly one simulation-execution system in this toolkit, not two.

**Feasible vs. infeasible vs. failed vs. invalid.** Every evaluated
design ends up in exactly one of four states
(:class:`DesignStatus`): ``FEASIBLE`` (the simulation completed and
every constraint was satisfied), ``INFEASIBLE`` (the simulation
completed but at least one constraint was violated),
``FAILED`` (the simulation itself did not complete, or an objective/
constraint could not be computed from a completed run), or ``INVALID``
(the design's variable values do not lie within their own declared
domain -- should never occur when an algorithm respects variable
bounds; reserved for a directly-supplied design).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

from femtoolkit.optimization.constraints import Constraint, ConstraintRelation
from femtoolkit.optimization.context import DesignContext
from femtoolkit.optimization.objectives import Objective
from femtoolkit.optimization.variables import DesignVariable
from femtoolkit.runs.manager import SimulationRunManager
from femtoolkit.runs.models import RunStatus
from femtoolkit.studies.scenarios import Scenario, apply_scenario
from femtoolkit.verification.status import VerificationStatus

if TYPE_CHECKING:
    from femtoolkit.application.project import Project


class DesignStatus(Enum):
    """The outcome status of one evaluated design.

    Attributes:
        FEASIBLE: The simulation completed and every constraint was satisfied.
        INFEASIBLE: The simulation completed but at least one constraint
            was violated.
        FAILED: The simulation did not complete (a solver/validation
            error), or an objective/constraint could not be computed
            from an otherwise-completed run -- no objective or
            constraint value is available.
        INVALID: The design's variable values do not lie within their
            own declared domain -- rejected before any scenario was
            built or any solver ran.
    """

    FEASIBLE = "feasible"
    INFEASIBLE = "infeasible"
    FAILED = "failed"
    INVALID = "invalid"


_FEASIBILITY_RANK = {
    DesignStatus.FEASIBLE: 0,
    DesignStatus.INFEASIBLE: 1,
    DesignStatus.FAILED: 2,
    DesignStatus.INVALID: 2,
}


def feasibility_rank(status: DesignStatus) -> int:
    """A total order over :class:`DesignStatus` for comparison: lower is better.

    ``FEASIBLE`` (0) < ``INFEASIBLE`` (1) < ``FAILED``/``INVALID`` (2, tied
    -- neither carries a usable objective value to compare further).
    """
    return _FEASIBILITY_RANK[status]


@dataclass(frozen=True)
class ConstraintEvaluation:
    """One constraint's evaluated outcome for one design.

    Attributes:
        name: The constraint's name.
        value: The constraint's evaluated quantity.
        limit: The constraint's limit.
        relation: The constraint's relation.
        violation: The non-negative violation magnitude (``0.0`` if satisfied).
        satisfied: Whether the constraint was satisfied.
    """

    name: str
    value: float
    limit: float
    relation: ConstraintRelation
    violation: float
    satisfied: bool


@dataclass
class DesignEvaluation:
    """The complete, structured outcome of evaluating one design.

    Attributes:
        design_id: A unique identifier for this evaluation.
        design_variables: This design's variable values, keyed by
            :attr:`~femtoolkit.optimization.variables.DesignVariable.name`.
        run_id: The originating
            :class:`~femtoolkit.runs.models.SimulationRun`'s ID, or
            ``None`` if the design was rejected as ``INVALID`` before
            any run was attempted.
        status: This design's :class:`DesignStatus`.
        objective_values: Every objective's evaluated value, by name --
            empty for ``FAILED``/``INVALID``.
        constraint_evaluations: Every constraint's
            :class:`ConstraintEvaluation` -- empty for
            ``FAILED``/``INVALID``.
        total_violation: The sum of every constraint's violation
            (``0.0`` if there are no constraints, or if the design is
            ``FAILED``/``INVALID`` and no violation could be computed).
        execution_time_seconds: The run's wall-clock solve time, or
            ``None`` if no run was attempted.
        verification_status: The run's Version 29 equilibrium-check
            status (see :attr:`~femtoolkit.runs.models.SimulationRun.verification_status`),
            or :attr:`~femtoolkit.verification.status.VerificationStatus.NOT_RUN`.
        error_message: A human-readable description of why the design
            is ``FAILED``/``INVALID``, or ``None``.
    """

    design_id: str
    design_variables: dict[str, Any]
    run_id: str | None
    status: DesignStatus
    objective_values: dict[str, float] = field(default_factory=dict)
    constraint_evaluations: list[ConstraintEvaluation] = field(default_factory=list)
    total_violation: float = 0.0
    execution_time_seconds: float | None = None
    verification_status: VerificationStatus = VerificationStatus.NOT_RUN
    error_message: str | None = None

    @property
    def is_feasible(self) -> bool:
        """Whether this design's status is :attr:`DesignStatus.FEASIBLE`."""
        return self.status is DesignStatus.FEASIBLE


def is_better_evaluation(
    candidate: DesignEvaluation, incumbent: DesignEvaluation, objective: Objective
) -> bool:
    """Whether ``candidate`` should replace ``incumbent`` as the current best design.

    Implements a feasibility-first comparison (spec: constraint
    handling): a feasible design always beats an infeasible one, which
    always beats a failed/invalid one; among two feasible designs, the
    better objective value wins; among two infeasible designs, the
    smaller total constraint violation is used only as a secondary
    tie-breaking mechanism (never a hidden combined score). Two
    failed/invalid designs are never considered comparable -- the
    incumbent is kept.

    Args:
        candidate: The newly evaluated design.
        incumbent: The current best design.
        objective: The objective to compare feasible designs against.

    Returns:
        ``True`` if ``candidate`` is better than ``incumbent``.
    """
    candidate_rank = feasibility_rank(candidate.status)
    incumbent_rank = feasibility_rank(incumbent.status)
    if candidate_rank != incumbent_rank:
        return candidate_rank < incumbent_rank
    if candidate.status is DesignStatus.FEASIBLE:
        candidate_value = candidate.objective_values[objective.name]
        incumbent_value = incumbent.objective_values[objective.name]
        if objective.direction.value == "minimize":
            return candidate_value < incumbent_value
        return candidate_value > incumbent_value
    if candidate.status is DesignStatus.INFEASIBLE:
        return candidate.total_violation < incumbent.total_violation
    return False


def _invalid_reason(values: dict[str, Any], design_variables: list[DesignVariable]) -> str | None:
    for variable in design_variables:
        if variable.name not in values:
            return f"Missing value for design variable {variable.name!r}."
        if not variable.is_valid_value(values[variable.name]):
            return (
                f"Value {values[variable.name]!r} is outside the domain of design "
                f"variable {variable.name!r}."
            )
    return None


def evaluate_design(
    design_id: str,
    values: dict[str, Any],
    design_variables: list[DesignVariable],
    base_project: Project,
    objectives: list[Objective],
    constraints: list[Constraint],
    run_manager: SimulationRunManager,
) -> DesignEvaluation:
    """Evaluate one design: build its scenario, run it, and score its objectives/constraints.

    Args:
        design_id: A unique identifier for this evaluation (used as the
            underlying scenario/run ID too).
        values: This design's variable values, keyed by design-variable name.
        design_variables: The problem's design variable definitions
            (for domain validation and path lookup).
        base_project: The unmodified base
            :class:`~femtoolkit.application.project.Project` every
            design overrides.
        objectives: The problem's objectives.
        constraints: The problem's constraints.
        run_manager: The run manager to execute the design's scenario with.

    Returns:
        A fully populated :class:`DesignEvaluation`.
    """
    reason = _invalid_reason(values, design_variables)
    if reason is not None:
        return DesignEvaluation(
            design_id=design_id,
            design_variables=dict(values),
            run_id=None,
            status=DesignStatus.INVALID,
            error_message=reason,
        )

    parameter_overrides = {variable.path: values[variable.name] for variable in design_variables}
    scenario = Scenario(
        scenario_id=design_id,
        name=f"Design {design_id}",
        parameter_overrides=parameter_overrides,
        tags=["optimization"],
    )
    project = apply_scenario(base_project, scenario)
    run = run_manager.execute(project, scenario_id=design_id)

    if run.status != RunStatus.COMPLETED:
        return DesignEvaluation(
            design_id=design_id,
            design_variables=dict(values),
            run_id=run.run_id,
            status=DesignStatus.FAILED,
            execution_time_seconds=run.execution_time_seconds,
            verification_status=run.verification_status,
            error_message=run.error_message,
        )

    context = DesignContext(design_variables=dict(values), project=project, run=run)

    objective_values: dict[str, float] = {}
    for objective in objectives:
        value = objective.evaluate(context)
        if value is None:
            return DesignEvaluation(
                design_id=design_id,
                design_variables=dict(values),
                run_id=run.run_id,
                status=DesignStatus.FAILED,
                execution_time_seconds=run.execution_time_seconds,
                verification_status=run.verification_status,
                error_message=f"Objective {objective.name!r} could not be evaluated.",
            )
        objective_values[objective.name] = value

    constraint_evaluations: list[ConstraintEvaluation] = []
    total_violation = 0.0
    for constraint in constraints:
        value = constraint.evaluate(context)
        if value is None:
            return DesignEvaluation(
                design_id=design_id,
                design_variables=dict(values),
                run_id=run.run_id,
                status=DesignStatus.FAILED,
                execution_time_seconds=run.execution_time_seconds,
                verification_status=run.verification_status,
                error_message=f"Constraint {constraint.name!r} could not be evaluated.",
            )
        violation = constraint.violation(value)
        total_violation += violation
        constraint_evaluations.append(
            ConstraintEvaluation(
                name=constraint.name,
                value=value,
                limit=constraint.limit,
                relation=constraint.relation,
                violation=violation,
                satisfied=violation <= 0.0,
            )
        )

    status = DesignStatus.FEASIBLE if total_violation <= 0.0 else DesignStatus.INFEASIBLE

    return DesignEvaluation(
        design_id=design_id,
        design_variables=dict(values),
        run_id=run.run_id,
        status=status,
        objective_values=objective_values,
        constraint_evaluations=constraint_evaluations,
        total_violation=total_violation,
        execution_time_seconds=run.execution_time_seconds,
        verification_status=run.verification_status,
    )


__all__ = [
    "ConstraintEvaluation",
    "DesignEvaluation",
    "DesignStatus",
    "evaluate_design",
    "feasibility_rank",
    "is_better_evaluation",
]
