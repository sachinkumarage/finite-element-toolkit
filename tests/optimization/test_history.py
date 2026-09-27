"""Tests for femtoolkit.optimization.history."""

import pytest

from femtoolkit.exceptions import ValidationError
from femtoolkit.optimization.evaluation import DesignEvaluation, DesignStatus
from femtoolkit.optimization.history import OptimizationHistory, compute_convergence
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
