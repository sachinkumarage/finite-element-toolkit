"""Tests for femtoolkit.optimization.plots."""

import pytest
from matplotlib.figure import Figure

from femtoolkit.application.project import BoundaryConditionConfig, LoadConfig, Project
from femtoolkit.exceptions import ValidationError
from femtoolkit.optimization.algorithms import OptimizationConfig
from femtoolkit.optimization.history import OptimizationHistory
from femtoolkit.optimization.objectives import (
    Objective,
    ObjectiveDirection,
    from_result_extractor,
    rectangular_mass,
)
from femtoolkit.optimization.plots import (
    plot_constraint_violation_history,
    plot_design_variable_history,
    plot_generation_objective_history,
    plot_objective_history,
    plot_pareto_front,
    plot_pareto_front_3d,
    plot_pareto_front_size_history,
)
from femtoolkit.optimization.problems import OptimizationProblem
from femtoolkit.optimization.runner import OptimizationRunner
from femtoolkit.optimization.variables import DesignVariable, DesignVariableType
from femtoolkit.studies.extractors import get_extractor


def _base_project() -> Project:
    project = Project(name="Plots Test", analysis_type="linear_static")
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


def _thickness_variable() -> DesignVariable:
    return DesignVariable(
        name="thickness", path="mesh.thickness", variable_type=DesignVariableType.CONTINUOUS,
        lower_bound=0.005, upper_bound=0.020, default_value=0.005, units="m",
    )


def test_plot_objective_history_returns_figure() -> None:
    objective = Objective(
        name="maximum_displacement", direction=ObjectiveDirection.MINIMIZE,
        evaluate=from_result_extractor(get_extractor("maximum_displacement")),
    )
    problem = OptimizationProblem(
        name="plots-test", base_project=_base_project(), design_variables=[_thickness_variable()],
        objectives=[objective],
    )
    config = OptimizationConfig(algorithm="coordinate_search", max_evaluations=20, seed=1)
    result = OptimizationRunner().run(problem, config)

    figure = plot_objective_history(result.history, objective)
    assert isinstance(figure, Figure)
    assert "maximum_displacement" in figure.axes[0].get_ylabel()


def test_plot_objective_history_requires_at_least_one_evaluation() -> None:
    objective = Objective(
        name="f", direction=ObjectiveDirection.MINIMIZE, evaluate=lambda ctx: None
    )
    with pytest.raises(ValidationError):
        plot_objective_history(OptimizationHistory(), objective)


def test_plot_constraint_violation_history_returns_figure() -> None:
    objective = Objective(
        name="maximum_displacement", direction=ObjectiveDirection.MINIMIZE,
        evaluate=from_result_extractor(get_extractor("maximum_displacement")),
    )
    problem = OptimizationProblem(
        name="plots-test", base_project=_base_project(), design_variables=[_thickness_variable()],
        objectives=[objective],
    )
    config = OptimizationConfig(algorithm="random_search", max_evaluations=10, seed=1)
    result = OptimizationRunner().run(problem, config)
    figure = plot_constraint_violation_history(result.history)
    assert isinstance(figure, Figure)


def test_plot_design_variable_history_returns_figure() -> None:
    objective = Objective(
        name="maximum_displacement", direction=ObjectiveDirection.MINIMIZE,
        evaluate=from_result_extractor(get_extractor("maximum_displacement")),
    )
    variable = _thickness_variable()
    problem = OptimizationProblem(
        name="plots-test", base_project=_base_project(), design_variables=[variable],
        objectives=[objective],
    )
    config = OptimizationConfig(algorithm="random_search", max_evaluations=10, seed=1)
    result = OptimizationRunner().run(problem, config)
    figure = plot_design_variable_history(result.history, variable)
    assert isinstance(figure, Figure)
    assert "thickness" in figure.axes[0].get_ylabel()


def test_plot_pareto_front_labels_feasible_pareto_and_baseline() -> None:
    displacement = Objective(
        name="maximum_displacement", direction=ObjectiveDirection.MINIMIZE,
        evaluate=from_result_extractor(get_extractor("maximum_displacement")),
    )
    mass = Objective(name="mass", direction=ObjectiveDirection.MINIMIZE, evaluate=rectangular_mass)
    problem = OptimizationProblem(
        name="multi-plots-test", base_project=_base_project(),
        design_variables=[_thickness_variable()], objectives=[displacement, mass],
    )
    config = OptimizationConfig(algorithm="random_search", max_evaluations=20, seed=1)
    result = OptimizationRunner().run(problem, config)

    figure = plot_pareto_front(
        result.history.evaluations, [displacement, mass], result.pareto_front(),
        baseline=result.baseline,
    )
    assert isinstance(figure, Figure)
    legend_labels = [text.get_text() for text in figure.axes[0].get_legend().get_texts()]
    assert "Pareto (non-dominated)" in legend_labels


def test_plot_pareto_front_requires_exactly_two_objectives() -> None:
    single = [Objective(name="f", direction=ObjectiveDirection.MINIMIZE, evaluate=lambda ctx: None)]
    with pytest.raises(ValidationError):
        plot_pareto_front([], single, [])


# --- Version 33: plot_pareto_front_3d, generation plots ---


def test_plot_pareto_front_3d_requires_exactly_three_objectives() -> None:
    two = [
        Objective(name="f1", direction=ObjectiveDirection.MINIMIZE, evaluate=lambda ctx: None),
        Objective(name="f2", direction=ObjectiveDirection.MINIMIZE, evaluate=lambda ctx: None),
    ]
    with pytest.raises(ValidationError):
        plot_pareto_front_3d([], two, [])


def test_plot_pareto_front_3d_returns_figure() -> None:
    displacement = Objective(
        name="maximum_displacement", direction=ObjectiveDirection.MINIMIZE,
        evaluate=from_result_extractor(get_extractor("maximum_displacement")),
    )
    mass = Objective(name="mass", direction=ObjectiveDirection.MINIMIZE, evaluate=rectangular_mass)
    stress = Objective(
        name="maximum_von_mises_stress", direction=ObjectiveDirection.MINIMIZE,
        evaluate=from_result_extractor(get_extractor("maximum_von_mises_stress")),
    )
    objectives = [displacement, mass, stress]
    problem = OptimizationProblem(
        name="plots-3d-test", base_project=_base_project(),
        design_variables=[_thickness_variable()], objectives=objectives,
    )
    config = OptimizationConfig(algorithm="random_search", max_evaluations=15, seed=1)
    result = OptimizationRunner().run(problem, config)

    figure = plot_pareto_front_3d(
        result.history.evaluations, objectives, result.pareto_front(), baseline=result.baseline
    )
    assert isinstance(figure, Figure)


def test_plot_generation_objective_history_returns_figure() -> None:
    from femtoolkit.optimization.history import generation_summaries

    objective = Objective(
        name="maximum_displacement", direction=ObjectiveDirection.MINIMIZE,
        evaluate=from_result_extractor(get_extractor("maximum_displacement")),
    )
    problem = OptimizationProblem(
        name="de-plot-test", base_project=_base_project(),
        design_variables=[_thickness_variable()], objectives=[objective],
    )
    config = OptimizationConfig(
        algorithm="differential_evolution", population_size=4, max_evaluations=12, seed=1
    )
    result = OptimizationRunner().run(problem, config)
    summaries = generation_summaries(result.history, result.objectives)

    figure = plot_generation_objective_history(summaries, objective)
    assert isinstance(figure, Figure)


def test_plot_generation_objective_history_requires_at_least_one_summary() -> None:
    objective = Objective(
        name="f", direction=ObjectiveDirection.MINIMIZE, evaluate=lambda ctx: None
    )
    with pytest.raises(ValidationError):
        plot_generation_objective_history([], objective)


def test_plot_pareto_front_size_history_returns_figure() -> None:
    from femtoolkit.optimization.history import generation_summaries

    displacement = Objective(
        name="maximum_displacement", direction=ObjectiveDirection.MINIMIZE,
        evaluate=from_result_extractor(get_extractor("maximum_displacement")),
    )
    mass = Objective(name="mass", direction=ObjectiveDirection.MINIMIZE, evaluate=rectangular_mass)
    problem = OptimizationProblem(
        name="nsga2-plot-test", base_project=_base_project(),
        design_variables=[_thickness_variable()], objectives=[displacement, mass],
    )
    config = OptimizationConfig(algorithm="nsga2", population_size=6, max_evaluations=24, seed=1)
    result = OptimizationRunner().run(problem, config)
    summaries = generation_summaries(result.history, result.objectives)

    figure = plot_pareto_front_size_history(summaries)
    assert isinstance(figure, Figure)


def test_plot_pareto_front_size_history_requires_at_least_one_summary() -> None:
    with pytest.raises(ValidationError):
        plot_pareto_front_size_history([])
