"""Tests for femtoolkit.optimization.history."""

import pytest

from femtoolkit.exceptions import ValidationError
from femtoolkit.optimization.evaluation import DesignEvaluation, DesignStatus
from femtoolkit.optimization.history import (
    OptimizationHistory,
    compute_convergence,
    generation_summaries,
)
from femtoolkit.optimization.objectives import Objective, ObjectiveDirection


def _evaluation(
    design_id: str, status: DesignStatus, value: float | None = None
) -> DesignEvaluation:
    objective_values = {"f": value} if value is not None else {}
    total_violation = 0.0 if status is not DesignStatus.INFEASIBLE else 1.0
    return DesignEvaluation(
        design_id=design_id, design_variables={}, run_id=design_id, status=status,
        objective_values=objective_values, total_violation=total_violation,
    )


def _objective(direction: ObjectiveDirection = ObjectiveDirection.MINIMIZE) -> Objective:
    return Objective(name="f", direction=direction, evaluate=lambda ctx: None)


def test_history_starts_empty() -> None:
    history = OptimizationHistory()
    assert history.n_evaluations == 0
    assert history.feasible_evaluations() == []


def test_history_add_and_counts() -> None:
    history = OptimizationHistory()
    history.add(_evaluation("a", DesignStatus.FEASIBLE, 1.0))
    history.add(_evaluation("b", DesignStatus.INFEASIBLE))
    history.add(_evaluation("c", DesignStatus.FAILED))
    assert history.n_evaluations == 3
    assert len(history.feasible_evaluations()) == 1
    assert len(history.infeasible_evaluations()) == 1
    assert len(history.failed_evaluations()) == 1


def test_best_feasible_minimize() -> None:
    history = OptimizationHistory()
    history.add(_evaluation("a", DesignStatus.FEASIBLE, 10.0))
    history.add(_evaluation("b", DesignStatus.FEASIBLE, 5.0))
    history.add(_evaluation("c", DesignStatus.FEASIBLE, 8.0))
    best = history.best_feasible(_objective(ObjectiveDirection.MINIMIZE))
    assert best.design_id == "b"


def test_best_feasible_maximize() -> None:
    history = OptimizationHistory()
    history.add(_evaluation("a", DesignStatus.FEASIBLE, 10.0))
    history.add(_evaluation("b", DesignStatus.FEASIBLE, 25.0))
    best = history.best_feasible(_objective(ObjectiveDirection.MAXIMIZE))
    assert best.design_id == "b"


def test_best_feasible_none_when_no_feasible_evaluation() -> None:
    history = OptimizationHistory()
    history.add(_evaluation("a", DesignStatus.FAILED))
    assert history.best_feasible(_objective()) is None


def test_best_overall_falls_back_to_infeasible() -> None:
    history = OptimizationHistory()
    history.add(_evaluation("a", DesignStatus.FAILED))
    history.add(_evaluation("b", DesignStatus.INFEASIBLE))
    best = history.best_overall(_objective())
    assert best.design_id == "b"


def test_best_so_far_series_ignores_worse_and_infeasible() -> None:
    history = OptimizationHistory()
    history.add(_evaluation("a", DesignStatus.FEASIBLE, 10.0))
    history.add(_evaluation("b", DesignStatus.FEASIBLE, 15.0))
    history.add(_evaluation("c", DesignStatus.FEASIBLE, 5.0))
    history.add(_evaluation("d", DesignStatus.FAILED))
    series = history.best_so_far_series(_objective())
    assert series == [10.0, 10.0, 5.0, 5.0]


def test_best_so_far_series_none_until_first_feasible() -> None:
    history = OptimizationHistory()
    history.add(_evaluation("a", DesignStatus.FAILED))
    history.add(_evaluation("b", DesignStatus.FEASIBLE, 3.0))
    series = history.best_so_far_series(_objective())
    assert series == [None, 3.0]


def test_compute_convergence_known_sequence() -> None:
    history = OptimizationHistory()
    history.add(_evaluation("a", DesignStatus.FEASIBLE, 10.0))
    history.add(_evaluation("b", DesignStatus.FEASIBLE, 5.0))
    steps = compute_convergence(history, _objective())
    assert steps[0].absolute_improvement is None
    assert steps[1].absolute_improvement == pytest.approx(5.0)
    assert steps[1].relative_improvement == pytest.approx(0.5)


def test_compute_convergence_requires_at_least_one_evaluation() -> None:
    with pytest.raises(ValidationError):
        compute_convergence(OptimizationHistory(), _objective())


# --- Version 33: generation_summaries ---


def _generation_evaluation(
    design_id: str, status: DesignStatus, value: float, generation: int
) -> DesignEvaluation:
    total_violation = 0.0 if status is not DesignStatus.INFEASIBLE else 1.0
    return DesignEvaluation(
        design_id=design_id, design_variables={}, run_id=design_id, status=status,
        objective_values={"f": value}, total_violation=total_violation, generation=generation,
    )


def test_generation_summaries_excludes_evaluations_without_a_generation() -> None:
    history = OptimizationHistory()
    history.add(_evaluation("a", DesignStatus.FEASIBLE, 1.0))  # generation=None (baseline-style)
    history.add(_generation_evaluation("b", DesignStatus.FEASIBLE, 2.0, generation=0))
    summaries = generation_summaries(history, [_objective()])
    assert len(summaries) == 1
    assert summaries[0].generation == 0


def test_generation_summaries_counts_feasible_infeasible_failed() -> None:
    history = OptimizationHistory()
    history.add(_generation_evaluation("a", DesignStatus.FEASIBLE, 1.0, generation=0))
    history.add(_generation_evaluation("b", DesignStatus.INFEASIBLE, 2.0, generation=0))
    history.add(_generation_evaluation("c", DesignStatus.FAILED, 0.0, generation=0))
    summaries = generation_summaries(history, [_objective()])
    assert summaries[0].n_evaluations == 3
    assert summaries[0].n_feasible == 1
    assert summaries[0].n_infeasible == 1
    assert summaries[0].n_failed == 1


def test_generation_summaries_best_feasible_objective_per_generation() -> None:
    history = OptimizationHistory()
    history.add(_generation_evaluation("a", DesignStatus.FEASIBLE, 10.0, generation=0))
    history.add(_generation_evaluation("b", DesignStatus.FEASIBLE, 5.0, generation=0))
    history.add(_generation_evaluation("c", DesignStatus.FEASIBLE, 20.0, generation=1))
    summaries = generation_summaries(history, [_objective()])
    assert summaries[0].best_feasible_objective == pytest.approx(5.0)
    assert summaries[1].best_feasible_objective == pytest.approx(20.0)


def test_generation_summaries_not_necessarily_monotonically_improving() -> None:
    """A later generation's own best value can be worse than an earlier
    generation's -- this is expected, documented behavior, not a bug."""
    history = OptimizationHistory()
    history.add(_generation_evaluation("a", DesignStatus.FEASIBLE, 1.0, generation=0))
    history.add(_generation_evaluation("b", DesignStatus.FEASIBLE, 50.0, generation=1))
    summaries = generation_summaries(history, [_objective()])
    assert summaries[0].best_feasible_objective < summaries[1].best_feasible_objective


def test_generation_summaries_pareto_front_size_single_objective() -> None:
    history = OptimizationHistory()
    history.add(_generation_evaluation("a", DesignStatus.FEASIBLE, 5.0, generation=0))
    history.add(_generation_evaluation("b", DesignStatus.FEASIBLE, 5.0, generation=0))
    summaries = generation_summaries(history, [_objective()])
    # both evaluations tied for the best value are non-dominated
    assert summaries[0].pareto_front_size == 2
