"""Tests for femtoolkit.optimization.pareto."""

import math

from femtoolkit.optimization.evaluation import DesignEvaluation, DesignStatus
from femtoolkit.optimization.objectives import Objective, ObjectiveDirection
from femtoolkit.optimization.pareto import (
    constrained_dominates,
    crowding_distance,
    dominates,
    fast_non_dominated_sort,
    pareto_front,
)

_MIN_OBJECTIVES = [
    Objective(name="mass", direction=ObjectiveDirection.MINIMIZE, evaluate=lambda ctx: None),
    Objective(name="disp", direction=ObjectiveDirection.MINIMIZE, evaluate=lambda ctx: None),
]


def _evaluation(design_id: str, mass: float, disp: float, status=DesignStatus.FEASIBLE):
    return DesignEvaluation(
        design_id=design_id, design_variables={}, run_id=design_id, status=status,
        objective_values={"mass": mass, "disp": disp},
    )


def test_dominates_strictly_better_on_one_equal_on_other() -> None:
    a = {"mass": 10.0, "disp": 5.0}
    b = {"mass": 12.0, "disp": 5.0}
    assert dominates(a, b, _MIN_OBJECTIVES)
    assert not dominates(b, a, _MIN_OBJECTIVES)


def test_dominates_trade_off_is_non_dominated() -> None:
    a = {"mass": 10.0, "disp": 5.0}
    c = {"mass": 8.0, "disp": 9.0}
    assert not dominates(a, c, _MIN_OBJECTIVES)
    assert not dominates(c, a, _MIN_OBJECTIVES)


def test_dominates_identical_vectors_do_not_dominate() -> None:
    a = {"mass": 10.0, "disp": 5.0}
    d = {"mass": 10.0, "disp": 5.0}
    assert not dominates(a, d, _MIN_OBJECTIVES)
    assert not dominates(d, a, _MIN_OBJECTIVES)


def test_dominates_respects_maximize_direction() -> None:
    objectives = [
        Objective(
            name="stiffness", direction=ObjectiveDirection.MAXIMIZE, evaluate=lambda ctx: None
        ),
        Objective(name="mass", direction=ObjectiveDirection.MINIMIZE, evaluate=lambda ctx: None),
    ]
    better = {"stiffness": 100.0, "mass": 5.0}
    worse = {"stiffness": 80.0, "mass": 5.0}
    assert dominates(better, worse, objectives)
    assert not dominates(worse, better, objectives)


def test_pareto_front_excludes_dominated_designs() -> None:
    a = _evaluation("A", 10.0, 5.0)
    b = _evaluation("B", 12.0, 5.0)  # dominated by A
    c = _evaluation("C", 8.0, 9.0)  # trade-off, non-dominated
    front = pareto_front([a, b, c], _MIN_OBJECTIVES)
    front_ids = {e.design_id for e in front}
    assert front_ids == {"A", "C"}


def test_pareto_front_keeps_duplicate_objective_vectors() -> None:
    a = _evaluation("A", 10.0, 5.0)
    d = _evaluation("D", 10.0, 5.0)
    front = pareto_front([a, d], _MIN_OBJECTIVES)
    assert {e.design_id for e in front} == {"A", "D"}


def test_pareto_front_excludes_infeasible_and_failed() -> None:
    a = _evaluation("A", 10.0, 5.0)
    infeasible = _evaluation("B", 1.0, 1.0, status=DesignStatus.INFEASIBLE)
    failed = DesignEvaluation(
        design_id="C", design_variables={}, run_id="C", status=DesignStatus.FAILED
    )
    front = pareto_front([a, infeasible, failed], _MIN_OBJECTIVES)
    assert {e.design_id for e in front} == {"A"}


def test_pareto_front_empty_when_no_feasible_designs() -> None:
    infeasible = _evaluation("B", 1.0, 1.0, status=DesignStatus.INFEASIBLE)
    assert pareto_front([infeasible], _MIN_OBJECTIVES) == []


# --- Version 33: constrained_dominates, fast_non_dominated_sort, crowding_distance ---


def test_constrained_dominates_feasible_beats_infeasible_regardless_of_objectives() -> None:
    feasible = _evaluation("A", mass=100.0, disp=100.0, status=DesignStatus.FEASIBLE)
    infeasible = _evaluation("B", mass=1.0, disp=1.0, status=DesignStatus.INFEASIBLE)
    assert constrained_dominates(feasible, infeasible, _MIN_OBJECTIVES)
    assert not constrained_dominates(infeasible, feasible, _MIN_OBJECTIVES)


def test_constrained_dominates_infeasible_compares_by_total_violation() -> None:
    less_violated = DesignEvaluation(
        design_id="A", design_variables={}, run_id="A", status=DesignStatus.INFEASIBLE,
        objective_values={"mass": 1.0, "disp": 1.0}, total_violation=0.5,
    )
    more_violated = DesignEvaluation(
        design_id="B", design_variables={}, run_id="B", status=DesignStatus.INFEASIBLE,
        objective_values={"mass": 1.0, "disp": 1.0}, total_violation=2.0,
    )
    assert constrained_dominates(less_violated, more_violated, _MIN_OBJECTIVES)
    assert not constrained_dominates(more_violated, less_violated, _MIN_OBJECTIVES)


def test_constrained_dominates_two_feasible_uses_plain_dominance() -> None:
    better = _evaluation("A", mass=8.0, disp=5.0)
    worse = _evaluation("B", mass=10.0, disp=5.0)
    assert constrained_dominates(better, worse, _MIN_OBJECTIVES)
    assert not constrained_dominates(worse, better, _MIN_OBJECTIVES)


def test_constrained_dominates_two_failed_never_comparable() -> None:
    a = DesignEvaluation(design_id="A", design_variables={}, run_id="A", status=DesignStatus.FAILED)
    b = DesignEvaluation(design_id="B", design_variables={}, run_id="B", status=DesignStatus.FAILED)
    assert not constrained_dominates(a, b, _MIN_OBJECTIVES)
    assert not constrained_dominates(b, a, _MIN_OBJECTIVES)


def test_fast_non_dominated_sort_first_front_is_non_dominated_set() -> None:
    a = _evaluation("A", 10.0, 5.0)
    b = _evaluation("B", 12.0, 5.0)  # dominated by A
    c = _evaluation("C", 8.0, 9.0)  # trade-off with A, non-dominated
    fronts = fast_non_dominated_sort([a, b, c], _MIN_OBJECTIVES)
    assert {e.design_id for e in fronts[0]} == {"A", "C"}
    assert {e.design_id for e in fronts[1]} == {"B"}


def test_fast_non_dominated_sort_every_evaluation_appears_exactly_once() -> None:
    evaluations = [_evaluation(str(i), float(i), float(10 - i)) for i in range(6)]
    fronts = fast_non_dominated_sort(evaluations, _MIN_OBJECTIVES)
    all_ids = [e.design_id for front in fronts for e in front]
    assert sorted(all_ids) == sorted(e.design_id for e in evaluations)
    assert len(all_ids) == len(set(all_ids))


def test_fast_non_dominated_sort_ranks_feasible_above_infeasible() -> None:
    feasible = _evaluation("A", mass=100.0, disp=100.0)
    infeasible = _evaluation("B", mass=1.0, disp=1.0, status=DesignStatus.INFEASIBLE)
    fronts = fast_non_dominated_sort([feasible, infeasible], _MIN_OBJECTIVES)
    assert fronts[0][0].design_id == "A"
    assert fronts[1][0].design_id == "B"


def test_crowding_distance_boundary_points_are_infinite() -> None:
    front = [_evaluation(str(i), float(i), float(10 - i)) for i in range(5)]
    distances = crowding_distance(front, _MIN_OBJECTIVES)
    ordered = sorted(front, key=lambda e: e.objective_values["mass"])
    assert distances[ordered[0].design_id] == math.inf
    assert distances[ordered[-1].design_id] == math.inf


def test_crowding_distance_small_front_returns_infinite_for_all() -> None:
    front = [_evaluation("A", 1.0, 1.0), _evaluation("B", 2.0, 2.0)]
    distances = crowding_distance(front, _MIN_OBJECTIVES)
    assert all(value == math.inf for value in distances.values())


def test_crowding_distance_interior_point_is_finite_and_positive() -> None:
    front = [_evaluation(str(i), float(i), float(10 - i)) for i in range(5)]
    distances = crowding_distance(front, _MIN_OBJECTIVES)
    ordered = sorted(front, key=lambda e: e.objective_values["mass"])
    for interior in ordered[1:-1]:
        assert distances[interior.design_id] > 0.0
        assert math.isfinite(distances[interior.design_id])
