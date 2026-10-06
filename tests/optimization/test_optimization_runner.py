"""Integration tests for femtoolkit.optimization.runner (Optimization -> Simulation Runner)."""

from __future__ import annotations

import pytest

from femtoolkit.application.project import BoundaryConditionConfig, LoadConfig, Project
from femtoolkit.exceptions import StudySizeExceededError, ValidationError
from femtoolkit.optimization.algorithms import OptimizationConfig
from femtoolkit.optimization.constraints import Constraint, ConstraintRelation
from femtoolkit.optimization.evaluation import DesignStatus
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
from femtoolkit.verification.status import VerificationStatus


def _base_project() -> Project:
    project = Project(name="Runner Test", analysis_type="linear_static")
    project.material.youngs_modulus = 200e9
    project.material.poisson_ratio = 0.3
    project.material.density = 7850.0
    project.mesh.width, project.mesh.height = 2.0, 0.4
    project.mesh.nx, project.mesh.ny = 10, 3
    project.mesh.thickness = 0.005
    project.boundary_conditions = [
        BoundaryConditionConfig(region="left", dof="X", value=0.0),
        BoundaryConditionConfig(region="left", dof="Y", value=0.0),
    ]
    project.loads = [LoadConfig(region="right", dof="Y", magnitude=-2000.0)]
    return project


def _thickness_variable() -> DesignVariable:
    return DesignVariable(
        name="thickness", path="mesh.thickness", variable_type=DesignVariableType.CONTINUOUS,
        lower_bound=0.005, upper_bound=0.020, default_value=0.005,
    )


def _displacement_objective() -> Objective:
    return Objective(
        name="maximum_displacement", direction=ObjectiveDirection.MINIMIZE,
        evaluate=from_result_extractor(get_extractor("maximum_displacement")),
    )


def test_runner_evaluates_baseline_separately_from_history() -> None:
    problem = OptimizationProblem(
        name="beam", base_project=_base_project(), design_variables=[_thickness_variable()],
        objectives=[_displacement_objective()],
    )
    config = OptimizationConfig(algorithm="coordinate_search", max_evaluations=20, seed=1)
    result = OptimizationRunner().run(problem, config)

    assert result.baseline.design_variables["thickness"] == 0.005
    assert result.baseline.is_feasible
    # the baseline is not present a second time as an entry in the history
    assert result.baseline.design_id not in [e.design_id for e in result.history.evaluations]


def test_runner_minimizing_displacement_finds_thicker_beam() -> None:
    stress_constraint = Constraint(
        name="stress_limit",
        evaluate=from_result_extractor(get_extractor("maximum_von_mises_stress")),
        relation=ConstraintRelation.LESS_EQUAL, limit=500e6,
    )
    problem = OptimizationProblem(
        name="beam", base_project=_base_project(), design_variables=[_thickness_variable()],
        objectives=[_displacement_objective()], constraints=[stress_constraint],
    )
    config = OptimizationConfig(algorithm="coordinate_search", max_evaluations=40, seed=2)
    result = OptimizationRunner().run(problem, config)

    best = result.best_feasible()
    assert best is not None
    # minimizing displacement with no mass penalty should push thickness toward the
    # upper bound (a physically expected consequence of thicker == stiffer)
    assert best.design_variables["thickness"] > result.baseline.design_variables["thickness"]


def test_runner_improvement_over_baseline_is_signed_and_quantitative() -> None:
    problem = OptimizationProblem(
        name="beam", base_project=_base_project(), design_variables=[_thickness_variable()],
        objectives=[_displacement_objective()],
    )
    config = OptimizationConfig(algorithm="coordinate_search", max_evaluations=30, seed=3)
    result = OptimizationRunner().run(problem, config)

    improvement = result.improvement_over_baseline()
    assert improvement is not None
    assert improvement["objective"] == "maximum_displacement"
    assert improvement["percentage_change"] <= 0  # displacement decreased


def test_runner_multi_objective_pareto_front_is_feasible_only() -> None:
    mass_objective = Objective(
        name="mass", direction=ObjectiveDirection.MINIMIZE, evaluate=rectangular_mass
    )
    problem = OptimizationProblem(
        name="beam-multi", base_project=_base_project(), design_variables=[_thickness_variable()],
        objectives=[_displacement_objective(), mass_objective],
    )
    config = OptimizationConfig(algorithm="random_search", max_evaluations=20, seed=4)
    result = OptimizationRunner().run(problem, config)

    assert result.is_multi_objective
    front = result.pareto_front()
    assert all(e.is_feasible for e in front)


def test_runner_verification_status_is_populated_for_feasible_runs() -> None:
    problem = OptimizationProblem(
        name="beam", base_project=_base_project(), design_variables=[_thickness_variable()],
        objectives=[_displacement_objective()],
    )
    config = OptimizationConfig(algorithm="random_search", max_evaluations=5, seed=5)
    result = OptimizationRunner().run(problem, config)

    assert result.baseline.verification_status == VerificationStatus.PASS
    for evaluation in result.history.feasible_evaluations():
        assert evaluation.verification_status == VerificationStatus.PASS


def test_runner_rejects_oversized_evaluation_budget_before_execution() -> None:
    with pytest.raises(StudySizeExceededError):
        OptimizationConfig(algorithm="random_search", max_evaluations=5000, evaluation_limit=1000)


def test_runner_config_rejects_unknown_algorithm() -> None:
    with pytest.raises(ValidationError):
        OptimizationConfig(algorithm="not_a_real_algorithm")


def test_runner_handles_unsolvable_design_without_crashing() -> None:
    # Entirely negative range: every sample is a physically invalid Young's modulus,
    # so every evaluation reaches (and fails) project validation, with zero chance
    # of a stray sample happening to land on a valid E.
    bad_variable = DesignVariable(
        name="E", path="material.youngs_modulus", variable_type=DesignVariableType.CONTINUOUS,
        lower_bound=-1e11, upper_bound=-1.0, default_value=-1e10,
    )
    objective = _displacement_objective()
    problem = OptimizationProblem(
        name="unsolvable", base_project=_base_project(), design_variables=[bad_variable],
        objectives=[objective],
    )
    config = OptimizationConfig(
        algorithm="random_search", max_evaluations=5, seed=1, max_consecutive_failures=100
    )
    result = OptimizationRunner().run(problem, config)
    assert result.baseline.status == DesignStatus.FAILED
    assert all(e.status == DesignStatus.FAILED for e in result.history.evaluations)


# --- Version 34: orchestration_config forwarding ---


def test_runner_forwards_orchestration_config_to_algorithm() -> None:
    from femtoolkit.orchestration.config import OrchestrationConfig

    problem = OptimizationProblem(
        name="beam-parallel", base_project=_base_project(),
        design_variables=[_thickness_variable()], objectives=[_displacement_objective()],
    )
    config = OptimizationConfig(
        algorithm="genetic_algorithm", population_size=6, max_evaluations=12, seed=2,
    )
    result = OptimizationRunner().run(
        problem, config,
        orchestration_config=OrchestrationConfig(execution_mode="parallel", max_workers=2),
    )
    assert result.history.n_evaluations <= 12
    assert result.baseline is not None


def test_runner_default_orchestration_config_is_serial() -> None:
    problem = OptimizationProblem(
        name="beam-serial", base_project=_base_project(),
        design_variables=[_thickness_variable()], objectives=[_displacement_objective()],
    )
    config = OptimizationConfig(algorithm="coordinate_search", max_evaluations=10, seed=1)
    result = OptimizationRunner().run(problem, config)
    assert result.history.n_evaluations > 0
