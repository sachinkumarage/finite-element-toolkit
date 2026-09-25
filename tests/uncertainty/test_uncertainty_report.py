"""Tests for femtoolkit.uncertainty.report."""

import numpy as np

from femtoolkit.application.project import BoundaryConditionConfig, LoadConfig, Project
from femtoolkit.studies.extractors import get_extractor
from femtoolkit.uncertainty.confidence import confidence_interval_mean
from femtoolkit.uncertainty.correlation import correlation_summary
from femtoolkit.uncertainty.distributions import NormalDistribution
from femtoolkit.uncertainty.monte_carlo import MonteCarloConfig, MonteCarloRunner
from femtoolkit.uncertainty.parameters import UncertainParameter
from femtoolkit.uncertainty.reliability import exceedance_probability
from femtoolkit.uncertainty.report import (
    build_uncertainty_report,
    render_uncertainty_report_html,
    render_uncertainty_report_markdown,
    save_uncertainty_report,
)
from femtoolkit.uncertainty.statistics import compute_output_statistics


def _base_project() -> Project:
    project = Project(name="Report Beam", analysis_type="linear_static")
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


def _build_report():
    parameter = UncertainParameter(
        path="material.youngs_modulus",
        label="Young's Modulus",
        distribution=NormalDistribution(200e9, 5e9),
        units="Pa",
        physical_lower_bound=0.0,
    )
    config = MonteCarloConfig(
        study_id="mc-report", name="Report Study", base_project=_base_project(),
        parameters=[parameter], output_quantities=["maximum_displacement"], n_samples=25, seed=1,
    )
    result = MonteCarloRunner().run(config)
    extractor = get_extractor("maximum_displacement")
    values = result.output_values(extractor)

    stats = compute_output_statistics(
        values, "Maximum displacement", units="m", n_requested=config.n_samples,
        n_failed=result.n_failed, n_invalid=result.n_invalid,
    )
    ci = confidence_interval_mean(values, "Maximum displacement")
    x, y = result.successful_pairs("material.youngs_modulus", extractor)
    correlations = correlation_summary(
        "Maximum displacement", {"material.youngs_modulus": ("Young's Modulus", x, y)}
    )
    exceedance = exceedance_probability(
        values, threshold=float(np.percentile(values, 90)), quantity_label="Maximum displacement"
    )

    return build_uncertainty_report(
        title="Report Study Report",
        study_summary="Summary.",
        base_model_description="Model.",
        result=result,
        output_statistics=[stats],
        confidence_intervals=[ci],
        correlations=correlations,
        exceedances=[exceedance],
        conclusions="Conclusion text.",
    )


def test_render_markdown_has_nineteen_sections() -> None:
    report = _build_report()
    markdown = render_uncertainty_report_markdown(report)
    for index in range(1, 20):
        assert f"## {index}." in markdown, f"missing section {index}"
    assert "# Report Study Report" in markdown
    assert "Conclusion text." in markdown


def test_render_markdown_distinguishes_percentiles_from_confidence_interval() -> None:
    report = _build_report()
    markdown = render_uncertainty_report_markdown(report)
    assert "not a confidence interval" in markdown
    assert "not a percentile of the output" in markdown


def test_render_markdown_states_exceedance_is_empirical() -> None:
    report = _build_report()
    markdown = render_uncertainty_report_markdown(report)
    assert "empirical frequency" in markdown
    assert "not a rigorous" in markdown


def test_render_html_wraps_markdown() -> None:
    report = _build_report()
    html = render_uncertainty_report_html(report)
    assert "<html>" in html
    assert "Report Study Report" in html


def test_save_report_markdown(tmp_path) -> None:
    report = _build_report()
    path = tmp_path / "report.md"
    save_uncertainty_report(report, path, "markdown")
    assert path.exists()
    assert "Report Study Report" in path.read_text()
