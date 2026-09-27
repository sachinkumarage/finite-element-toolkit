"""Tests for femtoolkit.optimization.algorithms.random_search.

Uses a known mathematical objective, ``f(t) = (t - 0.1)^2`` (analytic
minimum at ``t = 0.1``), computed directly from the design variable
value rather than from the FEA result, to verify the algorithm's
mechanics independent of any particular FEA physics -- a real
(inexpensive, always-solvable) simulation still runs for every
evaluation, since there is only one evaluation pipeline in this
toolkit, but the objective ignores its result.
"""

from __future__ import annotations

import pytest

from femtoolkit.application.project import BoundaryConditionConfig, LoadConfig, Project
from femtoolkit.optimization.algorithms import OptimizationConfig, StopReason
from femtoolkit.optimization.algorithms.random_search import BoundedRandomSearch
from femtoolkit.optimization.constraints import Constraint, ConstraintRelation
from femtoolkit.optimization.evaluation import DesignStatus
from femtoolkit.optimization.history import OptimizationHistory
from femtoolkit.optimization.objectives import Objective, ObjectiveDirection
from femtoolkit.optimization.problems import OptimizationProblem
from femtoolkit.optimization.variables import DesignVariable, DesignVariableType
from femtoolkit.runs.manager import SimulationRunManager


def _base_project() -> Project:
    project = Project(name="Random Search Test", analysis_type="linear_static")
    project.material.youngs_modulus = 200e9
    project.material.poisson_ratio = 0.3
    project.mesh.width, project.mesh.height = 2.0, 0.4
    project.mesh.nx, project.mesh.ny = 6, 2
    project.mesh.thickness = 0.02
    project.boundary_conditions = [
        BoundaryConditionConfig(region="left", dof="X", value=0.0),
        BoundaryConditionConfig(region="left", dof="Y", value=0.0),
    ]
    project.loads = [LoadConfig(region="right", dof="Y", magnitude=-1000.0)]
    return project


def _quadratic_problem() -> OptimizationProblem:
    variable = DesignVariable(
        name="t", path="mesh.thickness", variable_type=DesignVariableType.CONTINUOUS,
        lower_bound=0.001, upper_bound=0.5, default_value=0.001,
    )
    objective = Objective(
        name="f", direction=ObjectiveDirection.MINIMIZE,
        evaluate=lambda ctx: (ctx.design_variables["t"] - 0.1) ** 2,
    )
    return OptimizationProblem(
        name="quadratic", base_project=_base_project(), design_variables=[variable],
        objectives=[objective],
    )


def test_random_search_finds_near_the_known_minimum() -> None:
    problem = _quadratic_problem()
    config = OptimizationConfig(algorithm="random_search", max_evaluations=300, seed=42)
    history = OptimizationHistory()
    BoundedRandomSearch().optimize(problem, config, SimulationRunManager(), history)
    best = history.best_feasible(problem.objectives[0])
    assert best is not None
    assert abs(best.design_variables["t"] - 0.1) < 0.02


def test_random_search_is_reproducible_with_same_seed() -> None:
    problem = _quadratic_problem()
    config = OptimizationConfig(algorithm="random_search", max_evaluations=50, seed=7)

    history_a = OptimizationHistory()
    BoundedRandomSearch().optimize(problem, config, SimulationRunManager(), history_a)
    history_b = OptimizationHistory()
    BoundedRandomSearch().optimize(problem, config, SimulationRunManager(), history_b)

    values_a = [e.design_variables["t"] for e in history_a.evaluations]
    values_b = [e.design_variables["t"] for e in history_b.evaluations]
    assert values_a == values_b


def test_random_search_differs_with_different_seed() -> None:
    problem = _quadratic_problem()
    config_a = OptimizationConfig(algorithm="random_search", max_evaluations=20, seed=1)
    config_b = OptimizationConfig(algorithm="random_search", max_evaluations=20, seed=2)

    history_a = OptimizationHistory()
    BoundedRandomSearch().optimize(problem, config_a, SimulationRunManager(), history_a)
    history_b = OptimizationHistory()
    BoundedRandomSearch().optimize(problem, config_b, SimulationRunManager(), history_b)

    values_a = [e.design_variables["t"] for e in history_a.evaluations]
    values_b = [e.design_variables["t"] for e in history_b.evaluations]
    assert values_a != values_b


def test_random_search_respects_evaluation_limit_when_not_converged() -> None:
    problem = _quadratic_problem()
    # A very tight tolerance and long patience makes convergence effectively unreachable
    # within the budget, so the run must stop at max_evaluations instead.
    config = OptimizationConfig(
        algorithm="random_search", max_evaluations=15, seed=3, tolerance=1e-300, patience=1000
    )
    history = OptimizationHistory()
    stop_reason = BoundedRandomSearch().optimize(problem, config, SimulationRunManager(), history)
    assert stop_reason == StopReason.MAX_EVALUATIONS
    assert history.n_evaluations == 15


def test_random_search_stays_within_variable_bounds() -> None:
    problem = _quadratic_problem()
    config = OptimizationConfig(algorithm="random_search", max_evaluations=100, seed=5)
    history = OptimizationHistory()
    BoundedRandomSearch().optimize(problem, config, SimulationRunManager(), history)
    for evaluation in history.evaluations:
        assert evaluation.status != DesignStatus.INVALID
        assert 0.001 <= evaluation.design_variables["t"] <= 0.5


def test_random_search_handles_constraint_failure_policy() -> None:
    """An impossible-to-satisfy constraint means every evaluation is infeasible;
    random search must still run to completion without crashing or misclassifying."""
    variable = DesignVariable(
        name="t", path="mesh.thickness", variable_type=DesignVariableType.CONTINUOUS,
        lower_bound=0.001, upper_bound=0.02, default_value=0.001,
    )
    objective = Objective(
        name="f", direction=ObjectiveDirection.MINIMIZE,
        evaluate=lambda ctx: (ctx.design_variables["t"] - 0.01) ** 2,
    )
    impossible_constraint = Constraint(
        name="impossible", evaluate=lambda ctx: 1.0,
        relation=ConstraintRelation.LESS_EQUAL, limit=-1.0,
    )
    problem = OptimizationProblem(
        name="impossible", base_project=_base_project(), design_variables=[variable],
        objectives=[objective], constraints=[impossible_constraint],
    )
    config = OptimizationConfig(algorithm="random_search", max_evaluations=10, seed=1)
    history = OptimizationHistory()
    stop_reason = BoundedRandomSearch().optimize(problem, config, SimulationRunManager(), history)
    assert stop_reason == StopReason.MAX_EVALUATIONS
    assert history.n_evaluations == 10
    assert all(e.status == DesignStatus.INFEASIBLE for e in history.evaluations)


@pytest.mark.parametrize("max_evaluations", [1, 5])
def test_random_search_never_exceeds_max_evaluations(max_evaluations: int) -> None:
    problem = _quadratic_problem()
    config = OptimizationConfig(
        algorithm="random_search", max_evaluations=max_evaluations, seed=9
    )
    history = OptimizationHistory()
    BoundedRandomSearch().optimize(problem, config, SimulationRunManager(), history)
    assert history.n_evaluations == max_evaluations
