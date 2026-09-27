"""Tests for femtoolkit.optimization.pareto."""

from femtoolkit.optimization.evaluation import DesignEvaluation, DesignStatus
from femtoolkit.optimization.objectives import Objective, ObjectiveDirection
from femtoolkit.optimization.pareto import dominates, pareto_front

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
