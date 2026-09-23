"""Engineering-quality simulation reporting (Version 29).

Turns a simulation's verification results, validation results, mesh
convergence study, equilibrium checks, and reproducibility metadata into
one archivable, human-readable document
(:class:`~femtoolkit.reporting.models.EngineeringReport`, rendered by
:mod:`femtoolkit.reporting.renderers`). See ``docs/reporting.md`` for
the full guide.
"""

from __future__ import annotations

from femtoolkit.reporting.metadata import (
    ReproducibilityMetadata,
    collect_dependency_versions,
    collect_reproducibility_metadata,
)
from femtoolkit.reporting.models import EngineeringReport
from femtoolkit.reporting.renderers import ReportFormat, render_html, render_markdown, save_report

__all__ = [
    "EngineeringReport",
    "ReportFormat",
    "ReproducibilityMetadata",
    "collect_dependency_versions",
    "collect_reproducibility_metadata",
    "render_html",
    "render_markdown",
    "save_report",
]
