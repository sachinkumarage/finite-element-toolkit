"""Tests for femtoolkit.reporting (Version 29)."""

from __future__ import annotations

from femtoolkit.reporting import (
    EngineeringReport,
    collect_dependency_versions,
    collect_reproducibility_metadata,
    render_html,
    render_markdown,
    save_report,
)
from femtoolkit.verification.benchmarks import axial_bar_cases
from femtoolkit.verification.checks import EquilibriumCheckResult
from femtoolkit.verification.runner import VerificationRunner
from femtoolkit.verification.status import VerificationStatus


def _minimal_report(**overrides) -> EngineeringReport:
    metadata = collect_reproducibility_metadata(
        model_name="Test Model",
        analysis_type="linear_static",
        mesh_statistics={"nodes": 2, "elements": 1},
        element_types=["BarElement"],
        material_properties={"youngs_modulus": 200e9},
        boundary_conditions_summary="Fixed at node 1.",
        loads_summary="Load at node 2.",
    )
    defaults = dict(
        title="Test Report",
        simulation_summary="A test simulation.",
        model_description="A test model.",
        geometry_description="A single bar.",
        mesh_summary={"nodes": 2, "elements": 1},
        materials_summary={"Steel": {"E": 200e9}},
        boundary_conditions_summary="Fixed at node 1.",
        loads_summary="Load at node 2.",
        analysis_type="linear_static",
        solver_configuration="Dense direct.",
        reproducibility=metadata,
    )
    defaults.update(overrides)
    return EngineeringReport(**defaults)


# --- metadata ----------------------------------------------------------------


def test_collect_dependency_versions_returns_known_packages() -> None:
    versions = collect_dependency_versions(("numpy",))
    assert "numpy" in versions
    assert versions["numpy"]


def test_collect_dependency_versions_omits_unknown_package() -> None:
    versions = collect_dependency_versions(("this-package-does-not-exist-xyz",))
    assert versions == {}


def test_collect_reproducibility_metadata_records_toolkit_version() -> None:
    from femtoolkit.config import __version__ as toolkit_version

    metadata = collect_reproducibility_metadata(
        model_name="M",
        analysis_type="linear_static",
        mesh_statistics={},
        element_types=[],
        material_properties={},
        boundary_conditions_summary="",
        loads_summary="",
    )
    assert metadata.toolkit_version == toolkit_version
    assert metadata.preconditioner is None


def test_reproducibility_metadata_to_dict_is_serializable() -> None:
    import json

    metadata = collect_reproducibility_metadata(
        model_name="M",
        analysis_type="linear_static",
        mesh_statistics={"nodes": 2},
        element_types=["BarElement"],
        material_properties={"E": 200e9},
        boundary_conditions_summary="",
        loads_summary="",
    )
    json.dumps(metadata.to_dict())  # must not raise


# --- models --------------------------------------------------------------------


def test_overall_status_not_run_when_empty() -> None:
    report = _minimal_report()
    assert report.overall_status is VerificationStatus.NOT_RUN


def test_overall_status_pass_when_all_results_pass() -> None:
    cases = axial_bar_cases()
    results = VerificationRunner().run_all(cases).results
    report = _minimal_report(verification_results=results)
    assert report.overall_status is VerificationStatus.PASS


def test_overall_status_fail_when_any_result_fails() -> None:
    check = EquilibriumCheckResult(
        name="Synthetic",
        description="",
        components=[],
        tolerance=None,
        status=VerificationStatus.FAIL,
        message="synthetic failure",
    )
    report = _minimal_report(equilibrium_checks=[check])
    assert report.overall_status is VerificationStatus.FAIL


def test_overall_status_warning_takes_priority_over_not_available() -> None:
    warning_check = EquilibriumCheckResult(
        name="W",
        description="",
        components=[],
        tolerance=None,
        status=VerificationStatus.WARNING,
        message="",
    )
    not_available_check = EquilibriumCheckResult(
        name="NA",
        description="",
        components=[],
        tolerance=None,
        status=VerificationStatus.NOT_AVAILABLE,
        message="",
    )
    report = _minimal_report(equilibrium_checks=[warning_check, not_available_check])
    assert report.overall_status is VerificationStatus.WARNING


# --- renderers -------------------------------------------------------------------


def test_render_markdown_contains_all_eighteen_sections() -> None:
    text = render_markdown(_minimal_report())
    for section_number in range(1, 19):
        assert f"## {section_number}." in text


def test_render_markdown_never_fabricates_validation_when_absent() -> None:
    text = render_markdown(_minimal_report())
    assert "has not been validated against external data" in text


def test_render_markdown_never_synthesizes_conclusions() -> None:
    text = render_markdown(_minimal_report(conclusions=""))
    assert "No conclusions were supplied" in text


def test_render_markdown_includes_caller_supplied_conclusions() -> None:
    text = render_markdown(_minimal_report(conclusions="Everything checks out."))
    assert "Everything checks out." in text


def test_render_html_is_valid_wrapper_around_markdown_content() -> None:
    html_text = render_html(_minimal_report())
    assert html_text.startswith("<!DOCTYPE html>")
    assert "<title>Test Report</title>" in html_text
    assert "Test Report" in html_text


def test_save_report_writes_markdown_file(tmp_path) -> None:
    path = tmp_path / "report.md"
    save_report(_minimal_report(), path, "markdown")
    assert path.exists()
    assert "# Test Report" in path.read_text()


def test_save_report_writes_html_file(tmp_path) -> None:
    path = tmp_path / "report.html"
    save_report(_minimal_report(), path, "html")
    assert path.exists()
    assert path.read_text().startswith("<!DOCTYPE html>")


def test_save_report_rejects_unknown_format(tmp_path) -> None:
    import pytest

    from femtoolkit.exceptions import ValidationError

    with pytest.raises(ValidationError):
        save_report(_minimal_report(), tmp_path / "report.pdf", "pdf")
