"""`OptimizationHistory`: every evaluated design, and convergence tracking (Version 32).

Recording every evaluation -- not just the best one found so far --
lets an optimization run be inspected after the fact rather than
treated as a black box: which designs were tried, in what order, which
were feasible, and how the best-so-far objective value moved as the
search progressed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from femtoolkit.exceptions import ValidationError
from femtoolkit.optimization.evaluation import DesignEvaluation, DesignStatus, is_better_evaluation
from femtoolkit.optimization.objectives import Objective

DEFAULT_CONVERGENCE_EPSILON = 1e-12


@dataclass
class OptimizationHistory:
    """Every design evaluated during one optimization run, in evaluation order.

    Attributes:
        evaluations: Every :class:`~femtoolkit.optimization.evaluation.DesignEvaluation`,
            in the order they were evaluated.
    """

    evaluations: list[DesignEvaluation] = field(default_factory=list)

    def add(self, evaluation: DesignEvaluation) -> None:
        """Append one evaluation to this history."""
        self.evaluations.append(evaluation)

    @property
    def n_evaluations(self) -> int:
        """How many designs have been evaluated."""
        return len(self.evaluations)

    def feasible_evaluations(self) -> list[DesignEvaluation]:
        """Every evaluation with
        :attr:`~femtoolkit.optimization.evaluation.DesignStatus.FEASIBLE`.
        """
        return [
            evaluation
            for evaluation in self.evaluations
            if evaluation.status is DesignStatus.FEASIBLE
        ]

    def infeasible_evaluations(self) -> list[DesignEvaluation]:
        """Every evaluation with
        :attr:`~femtoolkit.optimization.evaluation.DesignStatus.INFEASIBLE`.
        """
        return [
            evaluation
            for evaluation in self.evaluations
            if evaluation.status is DesignStatus.INFEASIBLE
        ]

    def failed_evaluations(self) -> list[DesignEvaluation]:
        """Every evaluation with status ``FAILED`` or ``INVALID``."""
        return [
            evaluation
            for evaluation in self.evaluations
            if evaluation.status in (DesignStatus.FAILED, DesignStatus.INVALID)
        ]

    def best_feasible(self, objective: Objective) -> DesignEvaluation | None:
        """The best feasible evaluation for ``objective``, or ``None`` if none is feasible.

        Args:
            objective: The objective to compare feasible evaluations against.
        """
        best: DesignEvaluation | None = None
        for evaluation in self.feasible_evaluations():
            if best is None or is_better_evaluation(evaluation, best, objective):
                best = evaluation
        return best

    def best_overall(self, objective: Objective) -> DesignEvaluation | None:
        """The best evaluation regardless of feasibility (feasible-first, see
        :func:`~femtoolkit.optimization.evaluation.is_better_evaluation`).

        Useful for reporting even when no feasible design was found.

        Args:
            objective: The objective to compare evaluations against.
        """
        best: DesignEvaluation | None = None
        for evaluation in self.evaluations:
            if best is None or is_better_evaluation(evaluation, best, objective):
                best = evaluation
        return best

    def best_so_far_series(self, objective: Objective) -> list[float | None]:
        """The running best-feasible objective value after each evaluation, in order.

        Args:
            objective: The objective to track.

        Returns:
            One entry per evaluation; ``None`` until the first feasible
            evaluation is reached.
        """
        best_evaluation: DesignEvaluation | None = None
        series: list[float | None] = []
        for evaluation in self.evaluations:
            is_new_best = best_evaluation is None or is_better_evaluation(
                evaluation, best_evaluation, objective
            )
            if evaluation.status is DesignStatus.FEASIBLE and is_new_best:
                best_evaluation = evaluation
            series.append(
                best_evaluation.objective_values[objective.name]
                if best_evaluation is not None
                else None
            )
        return series


@dataclass(frozen=True)
class GenerationSummary:
    """Aggregated statistics for one generation of a population-based algorithm (Version 33).

    Attributes:
        generation: The generation/iteration index.
        n_evaluations: How many designs were evaluated in this generation.
        n_feasible: How many of those were feasible.
        n_infeasible: How many were infeasible.
        n_failed: How many failed or were invalid.
        best_feasible_objective: The best feasible objective value found
            *within this generation*, or ``None`` if none was feasible.
        pareto_front_size: How many of this generation's own evaluations
            are non-dominated among each other (see
            :func:`~femtoolkit.optimization.pareto.pareto_front`),
            meaningful only for a multi-objective problem.
    """

    generation: int
    n_evaluations: int
    n_feasible: int
    n_infeasible: int
    n_failed: int
    best_feasible_objective: float | None
    pareto_front_size: int


def generation_summaries(
    history: OptimizationHistory, objectives: list[Objective]
) -> list[GenerationSummary]:
    """Summarize a population-based algorithm's history generation by generation.

    **Not necessarily monotonically improving.** Unlike
    :func:`compute_convergence`'s running best-so-far series, each
    generation's own ``best_feasible_objective`` here describes *that
    generation's* evaluations alone -- a later generation's best value
    can be worse than an earlier one's (e.g. differential evolution only
    replaces a population member when a trial strictly improves on it,
    so a generation that happens to produce few successful replacements
    is not evidence the search regressed). Track the running best-so-far
    value via :func:`compute_convergence` instead when that distinction
    matters.

    Args:
        history: The optimization history to summarize. Evaluations
            with ``generation is None`` (random search, coordinate
            search, and any baseline evaluation) are excluded.
        objectives: The problem's objectives. The first is used to
            report each generation's best feasible value; the full list
            defines the objective vector for ``pareto_front_size``
            (meaningful only for a multi-objective problem -- for a
            single objective, ``pareto_front_size`` is simply the count
            of feasible designs tied for the best value).

    Returns:
        One :class:`GenerationSummary` per distinct generation index
        present in ``history``, in ascending generation order.
    """
    from femtoolkit.optimization.pareto import pareto_front as _pareto_front

    primary_objective = objectives[0]
    by_generation: dict[int, list[DesignEvaluation]] = {}
    for evaluation in history.evaluations:
        if evaluation.generation is None:
            continue
        by_generation.setdefault(evaluation.generation, []).append(evaluation)

    summaries: list[GenerationSummary] = []
    for generation in sorted(by_generation):
        evaluations = by_generation[generation]
        feasible = [e for e in evaluations if e.status is DesignStatus.FEASIBLE]
        infeasible = [e for e in evaluations if e.status is DesignStatus.INFEASIBLE]
        failed = [
            e for e in evaluations if e.status in (DesignStatus.FAILED, DesignStatus.INVALID)
        ]
        best_value: float | None = None
        for evaluation in feasible:
            candidate_value = evaluation.objective_values[primary_objective.name]
            if best_value is None:
                best_value = candidate_value
            elif primary_objective.direction.value == "minimize":
                best_value = min(best_value, candidate_value)
            else:
                best_value = max(best_value, candidate_value)
        summaries.append(
            GenerationSummary(
                generation=generation,
                n_evaluations=len(evaluations),
                n_feasible=len(feasible),
                n_infeasible=len(infeasible),
                n_failed=len(failed),
                best_feasible_objective=best_value,
                pareto_front_size=len(_pareto_front(evaluations, objectives)),
            )
        )
    return summaries


@dataclass(frozen=True)
class ConvergenceStep:
    """The running best-feasible objective value and its improvement at one evaluation.

    Attributes:
        evaluation_number: The 1-indexed position in the history.
        best_value: The best-feasible objective value so far, or
            ``None`` if no feasible design had been found yet.
        absolute_improvement: ``|best_value_k - best_value_{k-1}|``, or
            ``None`` if either value is unavailable.
        relative_improvement: ``absolute_improvement / max(|best_value_{k-1}|, epsilon)``,
            or ``None`` under the same condition.
    """

    evaluation_number: int
    best_value: float | None
    absolute_improvement: float | None
    relative_improvement: float | None


def compute_convergence(
    history: OptimizationHistory, objective: Objective, epsilon: float = DEFAULT_CONVERGENCE_EPSILON
) -> list[ConvergenceStep]:
    """Trace the running best-feasible objective value and its improvement across a history.

    This describes how a derivative-free heuristic search's *observed*
    best value evolved -- it is not evidence of mathematical
    convergence to a global optimum (no such guarantee exists for the
    algorithms in this version).

    Args:
        history: The optimization history to trace.
        objective: The objective to track.
        epsilon: The denominator floor for the relative improvement.

    Returns:
        One :class:`ConvergenceStep` per evaluation.

    Raises:
        ValidationError: If the history has no evaluations.
    """
    if history.n_evaluations == 0:
        raise ValidationError("compute_convergence requires at least one evaluation.")

    series = history.best_so_far_series(objective)
    steps: list[ConvergenceStep] = []
    previous: float | None = None
    for index, value in enumerate(series, start=1):
        absolute_improvement = None
        relative_improvement = None
        if value is not None and previous is not None:
            absolute_improvement = abs(value - previous)
            relative_improvement = absolute_improvement / max(abs(previous), epsilon)
        steps.append(
            ConvergenceStep(
                evaluation_number=index,
                best_value=value,
                absolute_improvement=absolute_improvement,
                relative_improvement=relative_improvement,
            )
        )
        if value is not None:
            previous = value
    return steps


__all__ = [
    "DEFAULT_CONVERGENCE_EPSILON",
    "ConvergenceStep",
    "GenerationSummary",
    "OptimizationHistory",
    "compute_convergence",
    "generation_summaries",
]
