"""Renders an :class:`~femtoolkit.reporting.models.EngineeringReport` to text (Version 29).

Two renderers, Markdown and HTML, both built from plain string
formatting -- no templating engine, no new dependency. Markdown is the
primary, most human- and diff-friendly format (suitable for archiving
alongside a project's version control history, per spec section 13's
"easy to archive"); HTML wraps the same content for direct viewing in a
browser. PDF export was evaluated and deliberately **not** added: doing
it well needs either a heavy rendering dependency (a browser engine, a
LaTeX toolchain) or a hand-rolled layout engine, neither of which this
toolkit's dependency policy accepts for a document format Markdown/HTML
already serve -- see ``docs/reporting.md`` for the full rationale.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from femtoolkit.exceptions import ValidationError
from femtoolkit.reporting.models import EngineeringReport
from femtoolkit.verification.status import VerificationStatus

ReportFormat = Literal["markdown", "html"]


def _status_marker(status: VerificationStatus) -> str:
    return {
        VerificationStatus.PASS: "PASS",
        VerificationStatus.FAIL: "FAIL",
        VerificationStatus.WARNING: "WARNING",
        VerificationStatus.NOT_AVAILABLE: "NOT AVAILABLE",
        VerificationStatus.NOT_RUN: "NOT RUN",
    }[status]


def render_markdown(report: EngineeringReport) -> str:
    """Render an :class:`EngineeringReport` as a Markdown document.

    Args:
        report: The report to render.

    Returns:
        The complete report as a Markdown string.
    """
    lines: list[str] = [f"# {report.title}", ""]

    lines += ["## 1. Simulation Summary", "", report.simulation_summary, ""]
    lines += ["## 2. Model Description", "", report.model_description, ""]
    lines += ["## 3. Geometry", "", report.geometry_description, ""]

    lines += ["## 4. Mesh", ""]
    for key, value in report.mesh_summary.items():
        lines.append(f"- **{key}:** {value}")
    lines.append("")

    lines += ["## 5. Materials", ""]
    for material_name, properties in report.materials_summary.items():
        property_text = ", ".join(f"{name}={value}" for name, value in properties.items())
        lines.append(f"- **{material_name}:** {property_text}")
    lines.append("")

    lines += ["## 6. Boundary Conditions", "", report.boundary_conditions_summary, ""]
    lines += ["## 7. Loads", "", report.loads_summary, ""]
    lines += ["## 8. Analysis Type", "", report.analysis_type, ""]
    lines += ["## 9. Solver Configuration", "", report.solver_configuration, ""]

    lines += ["## 10. Solver Convergence", ""]
    if report.solver_convergence is None:
        lines.append("No solver convergence record available (default direct solve).")
    else:
        record = report.solver_convergence
        lines.append(f"- **Solver:** {record.solver_name} ({record.matrix_type})")
        lines.append(
            f"- **Preconditioner:** {record.preconditioner or 'none (not yet implemented)'}"
        )
        lines.append(f"- **Converged:** {record.converged}")
        if record.iterations is not None:
            lines.append(f"- **Iterations:** {record.iterations}")
        lines.append(f"- **Final residual:** {record.final_residual:.6e}")
        lines.append(f"- **Relative residual:** {record.relative_residual:.6e}")
        lines.append(f"- **Solve time:** {record.solve_time:.6f} s")
    lines.append("")

    lines += ["## 11. Verification Results", ""]
    if not report.verification_results:
        lines.append("No verification cases were run for this report.")
    else:
        lines.append("| Case | Quantity | Numerical | Reference | Rel. Error | Status |")
        lines.append("|---|---|---|---|---|---|")
        for result in report.verification_results:
            lines.append(
                f"| {result.case_name} | {result.quantity} | {result.numerical_value} | "
                f"{result.reference_value} | "
                f"{'-' if result.relative_error is None else f'{result.relative_error:.3e}'} | "
                f"{_status_marker(result.status)} |"
            )
    lines.append("")

    lines += ["## 12. Validation Results", ""]
    if not report.validation_results:
        lines.append(
            "No reference/experimental dataset was supplied for this report -- "
            "this model has not been validated against external data. "
            "(See docs/validation.md for the verification/validation distinction.)"
        )
    else:
        lines.append("| Dataset | Source | Quantity | Rel. Error | Status |")
        lines.append("|---|---|---|---|---|")
        for validation_result in report.validation_results:
            lines.append(
                f"| {validation_result.dataset_name} | {validation_result.dataset_source} | "
                f"{validation_result.quantity} | {validation_result.relative_error:.3e} | "
                f"{_status_marker(validation_result.status)} |"
            )
    lines.append("")

    lines += ["## 13. Mesh Convergence", ""]
    if report.mesh_convergence is None:
        lines.append("No mesh convergence study was run for this report.")
    else:
        study = report.mesh_convergence
        lines.append(f"Quantity: **{study.quantity}**, tolerance: {study.tolerance:.3e}")
        lines.append("")
        lines.append("| Level | Nodes | Elements | DOFs | Value | Relative Change |")
        lines.append("|---|---|---|---|---|---|")
        for point in study.points:
            change = "-" if point.relative_change is None else f"{point.relative_change:.3e}"
            lines.append(
                f"| {point.label} | {point.num_nodes} | {point.num_elements} | "
                f"{point.num_dofs} | {point.result_value} | {change} |"
            )
        lines.append("")
        lines.append(f"Study status: **{_status_marker(study.status)}**")
    lines.append("")

    lines += ["## 14. Equilibrium Checks", ""]
    if not report.equilibrium_checks:
        lines.append("No equilibrium checks were run for this report.")
    else:
        for check in report.equilibrium_checks:
            lines.append(f"- **{check.name}** [{_status_marker(check.status)}]: {check.message}")
    lines.append("")

    lines += ["## 15. Key Results", ""]
    for name, value in report.key_results.items():
        lines.append(f"- **{name}:** {value}")
    lines.append("")

    lines += ["## 16. Warnings", ""]
    if not report.warnings:
        lines.append("No warnings.")
    else:
        for warning in report.warnings:
            lines.append(f"- {warning}")
    lines.append("")

    lines += ["## 17. Reproducibility Metadata", ""]
    metadata = report.reproducibility
    lines.append(f"- **Toolkit version:** {metadata.toolkit_version}")
    lines.append(f"- **Python version:** {metadata.python_version}")
    lines.append(f"- **Platform:** {metadata.platform_description}")
    dependency_text = ", ".join(
        f"{name}={version}" for name, version in metadata.dependency_versions.items()
    )
    lines.append(f"- **Dependencies:** {dependency_text or 'none recorded'}")
    lines.append(f"- **Generated at:** {metadata.generated_at}")
    if metadata.degrees_of_freedom is not None:
        lines.append(f"- **Degrees of freedom:** {metadata.degrees_of_freedom}")
    if metadata.execution_mode is not None:
        lines.append(f"- **Execution mode:** {metadata.execution_mode}")
    if metadata.random_seed is not None:
        lines.append(f"- **Random seed:** {metadata.random_seed}")
    lines.append("")

    lines += ["## 18. Conclusions / Status", ""]
    lines.append(f"**Overall status: {_status_marker(report.overall_status)}**")
    lines.append("")
    if report.conclusions:
        lines.append(report.conclusions)
    else:
        lines.append(
            "No conclusions were supplied for this report -- see the Key Results and "
            "Verification/Validation sections above for the underlying data."
        )
    lines.append("")

    return "\n".join(lines)


def render_html(report: EngineeringReport) -> str:
    """Render an :class:`EngineeringReport` as a minimal, self-contained HTML document.

    Args:
        report: The report to render.

    Returns:
        The complete report as an HTML string (plain formatting, no
        external stylesheet or script dependency).
    """
    import html as html_module

    markdown_text = render_markdown(report)
    escaped_lines = [html_module.escape(line) for line in markdown_text.split("\n")]
    body = "\n".join(f"<p>{line}</p>" if line else "<br/>" for line in escaped_lines)
    return (
        '<!DOCTYPE html>\n<html><head><meta charset="utf-8">'
        f"<title>{html_module.escape(report.title)}</title></head>"
        f"<body>{body}</body></html>"
    )


def save_report(report: EngineeringReport, path: str | Path, report_format: ReportFormat) -> None:
    """Render and write an :class:`EngineeringReport` to a file.

    Args:
        report: The report to render.
        path: Output file path.
        report_format: ``"markdown"`` or ``"html"``.

    Raises:
        ValidationError: If ``report_format`` is not one of the
            supported formats.
    """
    if report_format == "markdown":
        text = render_markdown(report)
    elif report_format == "html":
        text = render_html(report)
    else:
        raise ValidationError(
            f"Unsupported report_format {report_format!r}, expected 'markdown' or 'html'."
        )

    Path(path).write_text(text)


__all__ = ["ReportFormat", "render_html", "render_markdown", "save_report"]
