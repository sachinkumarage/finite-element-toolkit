"""Tests for femtoolkit.studies.report."""

from femtoolkit.application.project import BoundaryConditionConfig, LoadConfig, Project
from femtoolkit.studies.extractors import get_extractor
from femtoolkit.studies.parameter_sweep import ParameterDefinition
from femtoolkit.studies.report import (
    build_study_report,
    render_study_report_html,
    render_study_report_markdown,
    save_study_report,
)
from femtoolkit.studies.runner import SimulationStudy, StudyRunner


def _base_project() -> Project:
    project = Project(name="Beam", analysis_type="linear_static")
    project.material.youngs_modulus = 200e9
    project.material.poisson_ratio = 0.3
    project.mesh.width, project.mesh.height = 2.0, 0.4
    project.mesh.nx, project.mesh.ny, project.mesh.thickness = 6, 2, 0.02
    project.boundary_conditions = [
        BoundaryConditionConfig(region="left", dof="X", value=0.0),
        BoundaryConditionConfig(region="left", dof="Y", value=0.0),
    ]
    project.loads = [LoadConfig(region="right", dof="Y", magnitude=-1000.0)]
    return project


def _study_result():
    parameter = ParameterDefinition(
        path="loads.0.magnitude", label="Load", values=[-1000.0, -2000.0, -3000.0]
    )
    study = SimulationStudy(
        study_id="s1", name="Load Study", base_project=_base_project(), parameters=[parameter]
    )
    result = StudyRunner().run(study)
    return result, parameter


def test_build_study_report_fills_reproducibility_from_first_successful_run() -> None:
    result, _ = _study_result()
    report = build_study_report(
        title="Report", study_summary="Summary", base_model_description="Model", result=result
    )
    assert report.reproducibility is not None
    assert report.reproducibility is result.successful_runs[0].reproducibility_metadata


def test_render_study_report_markdown_has_thirteen_sections() -> None:
    result, parameter = _study_result()
    extractor = get_extractor("maximum_displacement")
    report = build_study_report(
        title="Cantilever Study",
        study_summary="Summary",
        base_model_description="Model",
        result=result,
        comparisons=[result.compare(extractor, "Maximum displacement")],
        sensitivities=[result.sensitivity(parameter, extractor, "Maximum displacement")],
        conclusions="Linear proportionality confirmed.",
    )
    markdown = render_study_report_markdown(report)

    for section in [
        "## 1. Study Summary",
        "## 2. Base Model",
        "## 3. Parameter Definitions",
        "## 4. Scenarios",
        "## 5. Run Status",
        "## 6. Solver Information",
        "## 7. Verification Summary",
        "## 8. Validation Summary",
        "## 9. Result Comparison",
        "## 10. Sensitivity Results",
        "## 11. Plots",
        "## 12. Failed Runs",
        "## 13. Reproducibility Metadata",
    ]:
        assert section in markdown

    assert "# Cantilever Study" in markdown
    assert "Linear proportionality confirmed." in markdown
    assert "No runs failed." in markdown


def test_render_study_report_markdown_reports_failed_runs_without_aborting() -> None:
    bad_project = _base_project()
    parameter = ParameterDefinition(path="material.youngs_modulus", label="E", values=[-1.0, 200e9])
    study = SimulationStudy(
        study_id="s2", name="Mixed", base_project=bad_project, parameters=[parameter]
    )
    result = StudyRunner().run(study)
    report = build_study_report(
        title="Mixed Study", study_summary="", base_model_description="", result=result
    )
    markdown = render_study_report_markdown(report)

    assert "No runs failed." not in markdown
    assert "validation" in markdown


def test_render_study_report_html_wraps_markdown() -> None:
    result, _ = _study_result()
    report = build_study_report(
        title="HTML Report", study_summary="s", base_model_description="m", result=result
    )
    html = render_study_report_html(report)
    assert "<html>" in html
    assert "HTML Report" in html


def test_save_study_report_markdown(tmp_path) -> None:
    result, _ = _study_result()
    report = build_study_report(
        title="Saved Report", study_summary="s", base_model_description="m", result=result
    )
    path = tmp_path / "report.md"
    save_study_report(report, path, "markdown")
    assert path.exists()
    assert "Saved Report" in path.read_text()
