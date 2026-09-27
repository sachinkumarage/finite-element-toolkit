"""Tests for femtoolkit.optimization.problems."""

import pytest

from femtoolkit.application.project import BoundaryConditionConfig, LoadConfig, Project
from femtoolkit.exceptions import ValidationError
from femtoolkit.optimization.objectives import Objective, ObjectiveDirection
from femtoolkit.optimization.problems import OptimizationMode, OptimizationProblem
from femtoolkit.optimization.variables import DesignVariable, DesignVariableType


def _base_project() -> Project:
    project = Project(name="Problem Test", analysis_type="linear_static")
    project.material.youngs_modulus = 200e9
    project.material.poisson_ratio = 0.3
    project.mesh.width, project.mesh.height, project.mesh.thickness = 2.0, 0.4, 0.02
    project.mesh.nx, project.mesh.ny = 8, 2
    project.boundary_conditions = [
        BoundaryConditionConfig(region="left", dof="X", value=0.0),
        BoundaryConditionConfig(region="left", dof="Y", value=0.0),
    ]
    project.loads = [LoadConfig(region="right", dof="Y", magnitude=-1000.0)]
    return project


def _variable(name: str = "t") -> DesignVariable:
    return DesignVariable(
        name=name, path="mesh.thickness", variable_type=DesignVariableType.CONTINUOUS,
        lower_bound=0.005, upper_bound=0.020,
    )


def _objective(name: str = "f") -> Objective:
    return Objective(name=name, direction=ObjectiveDirection.MINIMIZE, evaluate=lambda ctx: None)


def test_problem_requires_at_least_one_variable() -> None:
    with pytest.raises(ValidationError):
        OptimizationProblem(
            name="p", base_project=_base_project(), design_variables=[], objectives=[_objective()]
        )


def test_problem_requires_at_least_one_objective() -> None:
    with pytest.raises(ValidationError):
        OptimizationProblem(
            name="p", base_project=_base_project(), design_variables=[_variable()], objectives=[]
        )


def test_problem_rejects_duplicate_variable_names() -> None:
    with pytest.raises(ValidationError):
        OptimizationProblem(
            name="p", base_project=_base_project(),
            design_variables=[_variable("t"), _variable("t")], objectives=[_objective()],
        )


def test_problem_rejects_duplicate_objective_names() -> None:
    with pytest.raises(ValidationError):
        OptimizationProblem(
            name="p", base_project=_base_project(), design_variables=[_variable()],
            objectives=[_objective("f"), _objective("f")],
        )


def test_problem_default_values() -> None:
    variable = _variable()
    problem = OptimizationProblem(
        name="p", base_project=_base_project(), design_variables=[variable],
        objectives=[_objective()],
    )
    assert problem.default_values() == {"t": variable.default_value}


def test_problem_is_multi_objective() -> None:
    single = OptimizationProblem(
        name="p", base_project=_base_project(), design_variables=[_variable()],
        objectives=[_objective("f")],
    )
    multi = OptimizationProblem(
        name="p", base_project=_base_project(), design_variables=[_variable()],
        objectives=[_objective("f"), _objective("g")],
    )
    assert not single.is_multi_objective
    assert multi.is_multi_objective


def test_problem_variable_and_objective_lookup() -> None:
    variable = _variable()
    objective = _objective()
    problem = OptimizationProblem(
        name="p", base_project=_base_project(), design_variables=[variable], objectives=[objective]
    )
    assert problem.variable("t") is variable
    assert problem.objective("f") is objective
    with pytest.raises(ValidationError):
        problem.variable("not_a_variable")
    with pytest.raises(ValidationError):
        problem.objective("not_an_objective")


def test_problem_default_mode_is_deterministic() -> None:
    problem = OptimizationProblem(
        name="p", base_project=_base_project(), design_variables=[_variable()],
        objectives=[_objective()],
    )
    assert problem.mode is OptimizationMode.DETERMINISTIC
