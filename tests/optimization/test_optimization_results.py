"""Tests for femtoolkit.optimization.results."""

import pytest

from femtoolkit.application.project import BoundaryConditionConfig, LoadConfig, Project
from femtoolkit.exceptions import ValidationError
from femtoolkit.optimization.algorithms import OptimizationConfig
from femtoolkit.optimization.objectives import (
    Objective,
    ObjectiveDirection,
    from_result_extractor,
    rectangular_mass,
)
from femtoolkit.optimization.problems import OptimizationProblem
from femtoolkit.optimization.runner import OptimizationRunner
from femtoolkit.optimization.variables import DesignVariable, DesignVariableType
from femtoolkit.studies.extractors import get_extractor


def _base_project() -> Project:
    project = Project(name="Results Test", analysis_type="linear_static")
    project.material.youngs_modulus = 200e9
    project.material.poisson_ratio = 0.3
    project.material.density = 7850.0
    project.mesh.width, project.mesh.height = 2.0, 0.4
    project.mesh.nx, project.mesh.ny = 8, 2
    project.mesh.thickness = 0.005
    project.boundary_conditions = [
        BoundaryConditionConfig(region="left", dof="X", value=0.0),
        BoundaryConditionConfig(region="left", dof="Y", value=0.0),
    ]
    project.loads = [LoadConfig(region="right", dof="Y", magnitude=-2000.0)]
    return project


def _build_result(objectives):
    variable = DesignVariable(
        name="thickness", path="mesh.thickness", variable_type=DesignVariableType.CONTINUOUS,
        lower_bound=0.005, upper_bound=0.020, default_value=0.005,
    )
    problem = OptimizationProblem(
        name="results-test", base_project=_base_project(), design_variables=[variable],
        objectives=objectives,
    )
    config = OptimizationConfig(algorithm="coordinate_search", max_evaluations=20, seed=1)
    return OptimizationRunner().run(problem, config)


def test_result_best_feasible_default_objective() -> None:
    objective = Objective(
        name="maximum_displacement", direction=ObjectiveDirection.MINIMIZE,
        evaluate=from_result_extractor(get_extractor("maximum_displacement")),
    )
    result = _build_result([objective])
    best = result.best_feasible()
    assert best is not None
    assert best.is_feasible


def test_result_best_feasible_unknown_objective_raises() -> None:
    objective = Objective(
        name="maximum_displacement", direction=ObjectiveDirection.MINIMIZE,
        evaluate=from_result_extractor(get_extractor("maximum_displacement")),
    )
    result = _build_result([objective])
    with pytest.raises(ValidationError):
        result.best_feasible("not_a_real_objective")


def test_result_is_multi_objective_false_for_single_objective() -> None:
    objective = Objective(
        name="maximum_displacement", direction=ObjectiveDirection.MINIMIZE,
        evaluate=from_result_extractor(get_extractor("maximum_displacement")),
    )
    result = _build_result([objective])
    assert not result.is_multi_objective
    # A single-objective "Pareto front" is a well-defined but trivial concept: exactly
    # the evaluation(s) tied for the best objective value, not an empty set.
    front = result.pareto_front()
    best = result.best_feasible()
    assert best in front


def test_result_improvement_over_baseline_signed_comparison() -> None:
    objective = Objective(
        name="maximum_displacement", direction=ObjectiveDirection.MINIMIZE,
        evaluate=from_result_extractor(get_extractor("maximum_displacement")),
    )
    result = _build_result([objective])
    improvement = result.improvement_over_baseline()
    assert improvement is not None
    assert set(improvement) == {
        "objective", "baseline_value", "best_value", "absolute_difference", "percentage_change",
    }
    # improved (minimized) displacement means the best value is <= the baseline value
    assert improvement["best_value"] <= improvement["baseline_value"]
    assert improvement["absolute_difference"] == pytest.approx(
        improvement["best_value"] - improvement["baseline_value"]
    )


def test_result_multi_objective_pareto_front() -> None:
    displacement = Objective(
        name="maximum_displacement", direction=ObjectiveDirection.MINIMIZE,
        evaluate=from_result_extractor(get_extractor("maximum_displacement")),
    )
    mass = Objective(name="mass", direction=ObjectiveDirection.MINIMIZE, evaluate=rectangular_mass)
    result = _build_result([displacement, mass])
    assert result.is_multi_objective
    front = result.pareto_front()
    assert len(front) >= 1
