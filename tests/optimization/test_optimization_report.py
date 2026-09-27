"""Tests for femtoolkit.optimization.report."""

from femtoolkit.application.project import BoundaryConditionConfig, LoadConfig, Project
from femtoolkit.optimization.algorithms import OptimizationConfig
from femtoolkit.optimization.constraints import Constraint, ConstraintRelation
from femtoolkit.optimization.objectives import (
    Objective,
    ObjectiveDirection,
    from_result_extractor,
    rectangular_mass,
)
from femtoolkit.optimization.problems import OptimizationProblem
from femtoolkit.optimization.report import (
    build_optimization_report,
    render_optimization_report_html,
    render_optimization_report_markdown,
    save_optimization_report,
)
from femtoolkit.optimization.runner import OptimizationRunner
from femtoolkit.optimization.variables import DesignVariable, DesignVariableType
from femtoolkit.studies.extractors import get_extractor


def _base_project() -> Project:
    project = Project(name="Report Test", analysis_type="linear_static")
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


def _single_objective_result():
    objective = Objective(
        name="maximum_displacement", direction=ObjectiveDirection.MINIMIZE,
        evaluate=from_result_extractor(get_extractor("maximum_displacement")),
    )
    stress_constraint = Constraint(
        name="stress_limit",
        evaluate=from_result_extractor(get_extractor("maximum_von_mises_stress")),
        relation=ConstraintRelation.LESS_EQUAL, limit=500e6,
    )
    problem = OptimizationProblem(
        name="report-test", base_project=_base_project(), design_variables=[_thickness_variable()],
        objectives=[objective], constraints=[stress_constraint],
    )
    config = OptimizationConfig(algorithm="coordinate_search", max_evaluations=20, seed=1)
    return OptimizationRunner().run(problem, config)


def test_render_markdown_has_twenty_sections() -> None:
    result = _single_objective_result()
    report = build_optimization_report(
        title="Report Test", summary="Summary.", base_model_description="Model.", result=result,
        conclusions="Conclusion text.",
    )
    markdown = render_optimization_report_markdown(report)
    for index in range(1, 21):
        assert f"## {index}." in markdown, f"missing section {index}"
    assert "# Report Test" in markdown
    assert "Conclusion text." in markdown


def test_render_markdown_never_claims_global_optimum() -> None:
    result = _single_objective_result()
    report = build_optimization_report(
        title="Report Test", summary="", base_model_description="", result=result
    )
    markdown = render_optimization_report_markdown(report)
    assert "global" in markdown.lower()
    assert "no distributed" in markdown.lower() or "sequentially" in markdown.lower()


def test_render_markdown_reports_design_variables_and_bounds() -> None:
    result = _single_objective_result()
    report = build_optimization_report(
        title="Report Test", summary="", base_model_description="", result=result
    )
    markdown = render_optimization_report_markdown(report)
    assert "thickness" in markdown
    assert "mesh.thickness" in markdown
    assert "0.005" in markdown  # lower bound / default appears somewhere


def test_render_markdown_multi_objective_reports_non_dominated_set() -> None:
    displacement = Objective(
        name="maximum_displacement", direction=ObjectiveDirection.MINIMIZE,
        evaluate=from_result_extractor(get_extractor("maximum_displacement")),
    )
    mass = Objective(name="mass", direction=ObjectiveDirection.MINIMIZE, evaluate=rectangular_mass)
    problem = OptimizationProblem(
        name="multi-report-test", base_project=_base_project(),
        design_variables=[_thickness_variable()], objectives=[displacement, mass],
    )
    config = OptimizationConfig(algorithm="random_search", max_evaluations=15, seed=2)
    result = OptimizationRunner().run(problem, config)
    report = build_optimization_report(
        title="Multi Report", summary="", base_model_description="", result=result
    )
    markdown = render_optimization_report_markdown(report)
    assert "non-dominated" in markdown
    assert "No single design is" in markdown


def test_render_html_wraps_markdown() -> None:
    result = _single_objective_result()
    report = build_optimization_report(
        title="HTML Report", summary="s", base_model_description="m", result=result
    )
    html = render_optimization_report_html(report)
    assert "<html>" in html
    assert "HTML Report" in html


def test_save_report_markdown(tmp_path) -> None:
    result = _single_objective_result()
    report = build_optimization_report(
        title="Saved Report", summary="s", base_model_description="m", result=result
    )
    path = tmp_path / "report.md"
    save_optimization_report(report, path, "markdown")
    assert path.exists()
    assert "Saved Report" in path.read_text()
