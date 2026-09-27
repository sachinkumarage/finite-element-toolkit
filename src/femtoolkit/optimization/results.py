"""`OptimizationResult`: the collected, reportable outcome of one optimization run (Version 32)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from femtoolkit.exceptions import ValidationError
from femtoolkit.optimization.algorithms.base import OptimizationConfig, StopReason
from femtoolkit.optimization.evaluation import DesignEvaluation
from femtoolkit.optimization.history import OptimizationHistory
from femtoolkit.optimization.objectives import Objective
from femtoolkit.optimization.pareto import pareto_front
from femtoolkit.optimization.variables import DesignVariable
from femtoolkit.studies.comparison import absolute_difference, percentage_change


@dataclass
class OptimizationResult:
    """The complete, collected outcome of one optimization run.

    Attributes:
        problem_name: The optimized problem's name.
        config: The configuration the run used.
        design_variables: The problem's design variable definitions
            (carried alongside the history so a report/plot always
            knows each variable's path, type, bounds, and units).
        objectives: The problem's objectives (carried alongside the
            history so a report/plot always knows each objective's
            direction and units).
        baseline: The baseline design's evaluation -- always evaluated
            separately from, and never counted against, ``history``'s
            evaluation budget (see spec: "clearly distinguish baseline
            vs. optimized/candidate designs").
        history: Every design evaluated during the search.
        stop_reason: Why the run stopped.
        generated_at: ISO-8601 UTC timestamp when this result was produced.
    """

    problem_name: str
    config: OptimizationConfig
    design_variables: list[DesignVariable]
    objectives: list[Objective]
    baseline: DesignEvaluation
    history: OptimizationHistory
    stop_reason: StopReason
    generated_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    @property
    def is_multi_objective(self) -> bool:
        """Whether this run optimized more than one objective."""
        return len(self.objectives) > 1

    def _resolve_objective(self, objective_name: str | None) -> Objective:
        if objective_name is None:
            return self.objectives[0]
        for objective in self.objectives:
            if objective.name == objective_name:
                return objective
        raise ValidationError(f"No objective named {objective_name!r} in this result.")

    def best_feasible(self, objective_name: str | None = None) -> DesignEvaluation | None:
        """The best feasible evaluation found, or ``None`` if none was feasible.

        Args:
            objective_name: Which objective to rank by; defaults to the
                first-listed objective.
        """
        return self.history.best_feasible(self._resolve_objective(objective_name))

    def pareto_front(self) -> list[DesignEvaluation]:
        """The non-dominated feasible evaluations across every objective (see
        :func:`~femtoolkit.optimization.pareto.pareto_front`). Never
        ranked into a single preferred solution.
        """
        return pareto_front(self.history.evaluations, self.objectives)

    def improvement_over_baseline(self, objective_name: str | None = None) -> dict[str, Any] | None:
        """The signed change from the baseline to the best feasible candidate.

        Reuses Version 30's signed comparison convention
        (:func:`~femtoolkit.studies.comparison.absolute_difference`/
        :func:`~femtoolkit.studies.comparison.percentage_change`) --
        this reports the quantitative difference only; it never claims
        the optimized design is "better" beyond the raw numbers shown.

        Args:
            objective_name: Which objective to compare; defaults to the
                first-listed objective.

        Returns:
            ``None`` if the baseline is not feasible or no feasible
            candidate was found; otherwise a dict with
            ``objective``/``baseline_value``/``best_value``/
            ``absolute_difference``/``percentage_change``.
        """
        objective = self._resolve_objective(objective_name)
        best = self.best_feasible(objective.name)
        if best is None or not self.baseline.is_feasible:
            return None
        baseline_value = self.baseline.objective_values[objective.name]
        best_value = best.objective_values[objective.name]
        return {
            "objective": objective.name,
            "baseline_value": baseline_value,
            "best_value": best_value,
            "absolute_difference": absolute_difference(baseline_value, best_value),
            "percentage_change": percentage_change(baseline_value, best_value),
        }


__all__ = ["OptimizationResult"]
