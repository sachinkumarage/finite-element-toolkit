"""Coordinate search: change one design variable at a time (Version 32).

.. code-block:: text

    Current design (starts at the baseline)
            |
            v
    For each design variable, try current value +/- step:
        Evaluate the +step candidate
        Evaluate the -step candidate
        Keep whichever (including the unmoved current design) is best
            (feasibility-first, see
             femtoolkit.optimization.evaluation.is_better_evaluation)
            |
            v
    Repeat full passes over every variable until a pass makes no
    improvement, or another stopping condition is reached

A step moves a continuous variable by ``step_size`` (a fraction of its
``[lower, upper]`` range, clamped back into range at a bound), an
integer variable by at least one integer step, and a categorical
variable to the next/previous entry in its category list (wrapping is
not used -- the ends of the list are the ends of the search). This
algorithm is deliberately simple and transparent -- a reader can trace
exactly why it moved from one design to the next -- at the cost of
being a **local** search: it can stop at a local optimum a global
search might avoid, and it says nothing about the value of variables it
never explores because an early pass already stopped improving.
"""

from __future__ import annotations

from typing import Any

from femtoolkit.optimization.algorithms.base import (
    OptimizationAlgorithm,
    OptimizationConfig,
    StopReason,
    should_stop,
)
from femtoolkit.optimization.evaluation import (
    DesignEvaluation,
    DesignStatus,
    evaluate_design,
    is_better_evaluation,
)
from femtoolkit.optimization.history import OptimizationHistory
from femtoolkit.optimization.problems import OptimizationProblem
from femtoolkit.optimization.variables import DesignVariable, DesignVariableType
from femtoolkit.orchestration.config import OrchestrationConfig
from femtoolkit.runs.manager import SimulationRunManager


def _step_value(variable: DesignVariable, current: Any, step_size: float, direction: int) -> Any:
    """The neighboring value one step of ``direction`` (+1 or -1) away from ``current``."""
    if variable.variable_type is DesignVariableType.CONTINUOUS:
        span = variable.upper_bound - variable.lower_bound
        return variable.clip(current + direction * step_size * span)
    if variable.variable_type is DesignVariableType.INTEGER:
        span = variable.upper_bound - variable.lower_bound
        step = max(1, round(step_size * span))
        return variable.clip(current + direction * step)
    index = variable.categories.index(current)
    next_index = index + direction
    if not (0 <= next_index < len(variable.categories)):
        return current
    return variable.categories[next_index]


class CoordinateSearch(OptimizationAlgorithm):
    """Improves one design variable at a time, starting from the problem's baseline.

    **No Version 34 batched evaluation.** Each move's candidates depend
    on the previous move's outcome -- there is no independent batch of
    evaluations to parallelize. ``orchestration_config`` is accepted for
    interface consistency and ignored.
    """

    def optimize(
        self,
        problem: OptimizationProblem,
        config: OptimizationConfig,
        run_manager: SimulationRunManager,
        history: OptimizationHistory,
        starting_evaluation: DesignEvaluation | None = None,
        orchestration_config: OrchestrationConfig | None = None,
    ) -> StopReason:
        del orchestration_config  # not batched in this version -- see class docstring
        primary_objective = problem.objectives[0]
        evaluation_index = 0

        if starting_evaluation is not None:
            current_evaluation = starting_evaluation
            current_values = dict(starting_evaluation.design_variables)
        else:
            current_values = problem.default_values()
            current_evaluation = evaluate_design(
                design_id=f"{problem.name}-cs-{evaluation_index}",
                values=current_values,
                design_variables=problem.design_variables,
                base_project=problem.base_project,
                objectives=problem.objectives,
                constraints=problem.constraints,
                run_manager=run_manager,
            )
            history.add(current_evaluation)
            evaluation_index += 1

        consecutive_failures = (
            1 if current_evaluation.status in (DesignStatus.FAILED, DesignStatus.INVALID) else 0
        )

        while True:
            stop_reason = should_stop(history, primary_objective, config, consecutive_failures)
            if stop_reason is not None:
                return stop_reason

            improved_this_pass = False
            for variable in problem.design_variables:
                for direction in (+1, -1):
                    stop_reason = should_stop(
                        history, primary_objective, config, consecutive_failures
                    )
                    if stop_reason is not None:
                        return stop_reason

                    candidate_value = _step_value(
                        variable, current_values[variable.name], config.step_size, direction
                    )
                    if candidate_value == current_values[variable.name]:
                        continue

                    candidate_values = dict(current_values)
                    candidate_values[variable.name] = candidate_value
                    candidate_evaluation = evaluate_design(
                        design_id=f"{problem.name}-cs-{evaluation_index}",
                        values=candidate_values,
                        design_variables=problem.design_variables,
                        base_project=problem.base_project,
                        objectives=problem.objectives,
                        constraints=problem.constraints,
                        run_manager=run_manager,
                    )
                    history.add(candidate_evaluation)
                    evaluation_index += 1
                    failed = candidate_evaluation.status in (
                        DesignStatus.FAILED,
                        DesignStatus.INVALID,
                    )
                    consecutive_failures = consecutive_failures + 1 if failed else 0

                    if is_better_evaluation(
                        candidate_evaluation, current_evaluation, primary_objective
                    ):
                        current_values = candidate_values
                        current_evaluation = candidate_evaluation
                        improved_this_pass = True

            if not improved_this_pass:
                return StopReason.COMPLETED


__all__ = ["CoordinateSearch"]
