"""Tests for femtoolkit.optimization.algorithms.coordinate_search.

Uses the same known mathematical objective as
``test_random_search.py``, ``f(t) = (t - 0.1)^2``, to verify the
algorithm's mechanics independent of any particular FEA physics.
"""

from __future__ import annotations

from femtoolkit.application.project import BoundaryConditionConfig, LoadConfig, Project
from femtoolkit.optimization.algorithms import OptimizationConfig, StopReason
from femtoolkit.optimization.algorithms.coordinate_search import CoordinateSearch
from femtoolkit.optimization.evaluation import evaluate_design
from femtoolkit.optimization.history import OptimizationHistory
from femtoolkit.optimization.objectives import Objective, ObjectiveDirection
from femtoolkit.optimization.problems import OptimizationProblem
from femtoolkit.optimization.variables import DesignVariable, DesignVariableType
from femtoolkit.runs.manager import SimulationRunManager


def _base_project() -> Project:
    project = Project(name="Coordinate Search Test", analysis_type="linear_static")
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


def _quadratic_problem(default_value: float = 0.001) -> OptimizationProblem:
    variable = DesignVariable(
        name="t", path="mesh.thickness", variable_type=DesignVariableType.CONTINUOUS,
        lower_bound=0.001, upper_bound=0.5, default_value=default_value,
    )
    objective = Objective(
        name="f", direction=ObjectiveDirection.MINIMIZE,
        evaluate=lambda ctx: (ctx.design_variables["t"] - 0.1) ** 2,
    )
    return OptimizationProblem(
        name="quadratic", base_project=_base_project(), design_variables=[variable],
        objectives=[objective],
    )


def test_coordinate_search_converges_near_known_minimum() -> None:
    problem = _quadratic_problem()
    config = OptimizationConfig(
        algorithm="coordinate_search", max_evaluations=200, step_size=0.2, seed=1
    )
    history = OptimizationHistory()
    stop_reason = CoordinateSearch().optimize(problem, config, SimulationRunManager(), history)
    best = history.best_feasible(problem.objectives[0])
    assert best is not None
    assert abs(best.design_variables["t"] - 0.1) < 0.01
    assert stop_reason == StopReason.COMPLETED


def test_coordinate_search_improves_monotonically() -> None:
    problem = _quadratic_problem()
    config = OptimizationConfig(
        algorithm="coordinate_search", max_evaluations=200, step_size=0.2, seed=1
    )
    history = OptimizationHistory()
    CoordinateSearch().optimize(problem, config, SimulationRunManager(), history)
    raw_series = history.best_so_far_series(problem.objectives[0])
    series = [value for value in raw_series if value is not None]
    for earlier, later in zip(series, series[1:], strict=False):
        assert later <= earlier


def test_coordinate_search_reuses_starting_evaluation_without_duplicate_run() -> None:
    problem = _quadratic_problem()
    run_manager = SimulationRunManager()
    starting = evaluate_design(
        f"{problem.name}-baseline", problem.default_values(), problem.design_variables,
        problem.base_project, problem.objectives, problem.constraints, run_manager,
    )
    config = OptimizationConfig(algorithm="coordinate_search", max_evaluations=50, seed=1)
    history = OptimizationHistory()
    CoordinateSearch().optimize(
        problem, config, run_manager, history, starting_evaluation=starting
    )
    # the starting evaluation itself must not appear a second time in the history
    assert starting.design_id not in [e.design_id for e in history.evaluations]


def test_coordinate_search_respects_step_size_bounds() -> None:
    problem = _quadratic_problem(default_value=0.001)
    config = OptimizationConfig(
        algorithm="coordinate_search", max_evaluations=6, step_size=0.05, seed=1
    )
    history = OptimizationHistory()
    CoordinateSearch().optimize(problem, config, SimulationRunManager(), history)
    for evaluation in history.evaluations:
        assert 0.001 <= evaluation.design_variables["t"] <= 0.5


def test_coordinate_search_never_exceeds_max_evaluations() -> None:
    problem = _quadratic_problem()
    config = OptimizationConfig(
        algorithm="coordinate_search", max_evaluations=3, step_size=0.2, seed=1
    )
    history = OptimizationHistory()
    stop_reason = CoordinateSearch().optimize(problem, config, SimulationRunManager(), history)
    assert history.n_evaluations <= 3
    assert stop_reason == StopReason.MAX_EVALUATIONS


def test_coordinate_search_on_integer_variable() -> None:
    variable = DesignVariable(
        name="n", path="mesh.nx", variable_type=DesignVariableType.INTEGER,
        lower_bound=1, upper_bound=20, default_value=1,
    )
    objective = Objective(
        name="f", direction=ObjectiveDirection.MINIMIZE,
        evaluate=lambda ctx: (ctx.design_variables["n"] - 8) ** 2,
    )
    problem = OptimizationProblem(
        name="integer-quadratic", base_project=_base_project(), design_variables=[variable],
        objectives=[objective],
    )
    # A coarse step_size on a wide integer domain can strand coordinate search at a
    # local point (a genuine, expected limitation, not a bug) -- use a finer step here.
    config = OptimizationConfig(
        algorithm="coordinate_search", max_evaluations=100, step_size=0.1, seed=1
    )
    history = OptimizationHistory()
    CoordinateSearch().optimize(problem, config, SimulationRunManager(), history)
    best = history.best_feasible(objective)
    assert best is not None
    assert abs(best.design_variables["n"] - 8) <= 1
