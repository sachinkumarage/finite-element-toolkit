"""Surrogate/ROM report generation (Version 35).

Follows the same per-package reporting convention Versions 30-33
established (:mod:`femtoolkit.studies.report`,
:mod:`femtoolkit.uncertainty.report`,
:mod:`femtoolkit.optimization.report`): a rendering-independent
dataclass plus plain Markdown/HTML renderer functions, living in this
package rather than :mod:`femtoolkit.reporting` to preserve the same
one-way dependency direction (surrogate depends on reporting, never the
reverse).

Mirrors spec section 34's six-part structure: Dataset, Surrogate, ROM,
Validation, Applicability, and High-Fidelity Verification. Every number
in this report comes from an already-computed
:class:`~femtoolkit.surrogate.validation.SurrogateValidationReport`,
:class:`~femtoolkit.surrogate.models.base.SurrogateModel`,
:class:`~femtoolkit.surrogate.rom.pod.PODModel`, or
:class:`~femtoolkit.surrogate.workflows.verification.SurrogateVerificationRecord`
-- this module performs no new calculation, only formatting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from femtoolkit.exceptions import ValidationError

if TYPE_CHECKING:
    from femtoolkit.surrogate.datasets import SnapshotDataset
    from femtoolkit.surrogate.models.base import SurrogateModel
    from femtoolkit.surrogate.rom.pod import PODModel
    from femtoolkit.surrogate.validation import SurrogateValidationReport
    from femtoolkit.surrogate.workflows.verification import SurrogateVerificationRecord

ReportFormat = Literal["markdown", "html"]


@dataclass
class SurrogateReport:
    """The complete, rendering-independent content of one surrogate/ROM report.

    Attributes:
        title: The report title.
        summary: A short, one-paragraph summary of what this surrogate/ROM represents.
        dataset: The training :class:`~femtoolkit.surrogate.datasets.SnapshotDataset`, if any.
        model: The trained surrogate model, if any.
        validation: The surrogate's
            :class:`~femtoolkit.surrogate.validation.SurrogateValidationReport`, if any.
        rom: The trained :class:`~femtoolkit.surrogate.rom.pod.PODModel`, if any.
        verification_records: Every high-fidelity verification performed, if any.
        conclusions: Free text the caller supplies -- never generated automatically.
    """

    title: str
    summary: str
    dataset: SnapshotDataset | None = None
    model: SurrogateModel | None = None
    validation: SurrogateValidationReport | None = None
    rom: PODModel | None = None
    verification_records: list[SurrogateVerificationRecord] = field(default_factory=list)
    conclusions: str = ""


def _metric_row(name: str, metrics) -> str:
    return (
        f"| {name} | {metrics.n_samples} | {metrics.mae:.4e} | {metrics.rmse:.4e} | "
        f"{metrics.r2:.4f} | {metrics.mean_relative_error:.4%} | {metrics.max_relative_error:.4%} |"
    )


def render_surrogate_report_markdown(report: SurrogateReport) -> str:
    """Render a :class:`SurrogateReport` as a six-section Markdown document.

    Args:
        report: The report to render.

    Returns:
        The complete report as a Markdown string.
    """
    lines: list[str] = [f"# {report.title}", "", report.summary, ""]

    lines += ["## 1. Dataset", ""]
    if report.dataset is None:
        lines.append("No training dataset is attached to this report.")
    else:
        dataset = report.dataset
        lines.append(
            f"- **Dataset ID:** `{dataset.dataset_id}` (version {dataset.dataset_version})"
        )
        lines.append(f"- **Snapshots:** {dataset.n_samples}")
        lines.append(f"- **Input variables:** {', '.join(dataset.feature_names)}")
        lines.append(f"- **Response variables:** {', '.join(dataset.response_names)}")
        lines.append(
            f"- **Source simulations:** {len(dataset.source_simulations)} tracked simulation run(s)"
        )
    lines.append("")

    lines += ["## 2. Surrogate", ""]
    if report.model is None:
        lines.append("No surrogate model is attached to this report.")
    else:
        model = report.model
        metadata = model.training_metadata
        lines.append(f"- **Model type:** `{model.name}`")
        if metadata is not None:
            lines.append(f"- **Training samples used:** {metadata.n_training_samples}")
            lines.append(f"- **Model version:** {metadata.model_version}")
            lines.append(f"- **Software version:** {metadata.software_version}")
            lines.append(f"- **Random seed:** {metadata.random_seed}")
    lines.append("")

    lines += ["## 3. Reduced-Order Model (POD)", ""]
    if report.rom is None:
        lines.append("No reduced-order model is attached to this report.")
    else:
        rom = report.rom
        lines.append(f"- **Total modes available:** {rom.total_modes}")
        lines.append(f"- **Selected modes:** {rom.selected_modes}")
        lines.append(f"- **Captured energy:** {rom.captured_energy:.6f}")
        lines.append(f"- **Discarded energy:** {rom.discarded_energy:.2e}")
    lines.append("")

    lines += ["## 4. Validation", ""]
    if report.validation is None:
        lines.append("No validation report is attached to this report.")
    else:
        validation = report.validation
        lines.append("| Response | N | MAE | RMSE | R^2 | Mean Rel. Error | Max Rel. Error |")
        lines.append("|---|---|---|---|---|---|---|")
        for name, metrics in validation.test_metrics.items():
            lines.append(_metric_row(name, metrics))
        lines.append("")
        meets = validation.meets_engineering_tolerances
        lines.append(f"**Meets configured engineering tolerances:** {meets}")
        if validation.warnings:
            lines.append("")
            lines.append("**Warnings:**")
            for warning in validation.warnings:
                lines.append(f"- {warning}")
    lines.append("")

    lines += ["## 5. Applicability Domain", ""]
    if report.model is None or report.model.training_metadata is None:
        lines.append("No surrogate model is attached to this report.")
    else:
        lines.append(
            "See each prediction's own `domain_status` "
            "(`within_training_domain`/`boundary`/`outside_training_domain`/`invalid`) -- "
            "this report does not repeat per-prediction domain checks."
        )
    lines.append("")

    lines += ["## 6. High-Fidelity Verification", ""]
    if not report.verification_records:
        lines.append("No high-fidelity verification was performed for this report.")
    else:
        lines.append("| Design ID | Domain Status | Acceptance |")
        lines.append("|---|---|---|")
        for record in report.verification_records:
            lines.append(
                f"| {record.design_id} | {record.domain_status} | {record.acceptance.value} |"
            )
    lines.append("")

    lines += ["## Conclusions", "", report.conclusions or "Not provided.", ""]
    return "\n".join(lines)


def render_surrogate_report_html(report: SurrogateReport) -> str:
    """Render a :class:`SurrogateReport` as a minimal, self-contained HTML document.

    Args:
        report: The report to render.

    Returns:
        The complete report as an HTML string.
    """
    import html as html_module

    markdown_text = render_surrogate_report_markdown(report)
    escaped_lines = [html_module.escape(line) for line in markdown_text.split("\n")]
    body = "\n".join(f"<p>{line}</p>" if line else "<br/>" for line in escaped_lines)
    return (
        '<!DOCTYPE html>\n<html><head><meta charset="utf-8">'
        f"<title>{html_module.escape(report.title)}</title></head>"
        f"<body>{body}</body></html>"
    )


def save_surrogate_report(
    report: SurrogateReport, path: str | Path, report_format: ReportFormat
) -> None:
    """Render and write a :class:`SurrogateReport` to a file.

    Args:
        report: The report to render.
        path: Output file path.
        report_format: ``"markdown"`` or ``"html"``.

    Raises:
        ValidationError: If ``report_format`` is not supported.
    """
    if report_format == "markdown":
        text = render_surrogate_report_markdown(report)
    elif report_format == "html":
        text = render_surrogate_report_html(report)
    else:
        raise ValidationError(
            f"Unsupported report_format {report_format!r}, expected 'markdown' or 'html'."
        )
    Path(path).write_text(text)


__all__ = [
    "ReportFormat",
    "SurrogateReport",
    "render_surrogate_report_html",
    "render_surrogate_report_markdown",
    "save_surrogate_report",
]
