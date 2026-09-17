"""The whole-mesh quality evaluator (Version 25, spec section 5).

:class:`QualityEvaluator` is the consistent, top-level entry point this
version's brief asks for (``quality = evaluator.evaluate(mesh)``): it
evaluates every element in a mesh that has a defined shape-quality
metric (:func:`~femtoolkit.mesh.quality.metrics.compute_any_element_quality`
dispatches by element type -- CST/Q4 via :class:`~femtoolkit.mesh.quality.metrics.ElementQuality`,
TET4/HEX8 via :class:`~femtoolkit.mesh.quality.metrics.SolidElementQuality`),
silently skipping element types with no such concept (bar, truss,
frame), and aggregates the result into a single
:class:`~femtoolkit.mesh.quality.quality_report.MeshQualityReport`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from femtoolkit.exceptions import UnsupportedQualityMetricError, ValidationError
from femtoolkit.mesh.quality.metrics import (
    ElementQuality,
    SolidElementQuality,
    compute_any_element_quality,
)
from femtoolkit.mesh.quality.quality_report import MeshQualityReport

if TYPE_CHECKING:
    from femtoolkit.mesh.mesh import Mesh

DEFAULT_POOR_QUALITY_THRESHOLD = 0.3
"""Default ``quality`` value (see
:class:`~femtoolkit.mesh.quality.metrics.ElementQuality`/
:class:`~femtoolkit.mesh.quality.metrics.SolidElementQuality`) below
which an element is flagged as poor quality. Chosen as a conservative,
clearly-distorted cutoff (an aspect ratio worse than about 3.3:1);
adjust per :class:`QualityEvaluator`'s constructor for a project's own
engineering standards."""


@dataclass(frozen=True)
class QualityEvaluator:
    """Evaluates whole-mesh shape quality across any mix of supported element types.

    Attributes:
        poor_quality_threshold: Elements with a ``quality`` value below
            this are flagged in the report's ``poor_quality_element_ids``
            and summarized in a warning. Must be in ``(0, 1]``.

    Raises:
        ValidationError: If ``poor_quality_threshold`` is not in ``(0, 1]``.

    Example:
        >>> evaluator = QualityEvaluator()
        >>> report = evaluator.evaluate(mesh)
        >>> report.minimum_quality
        0.42
    """

    poor_quality_threshold: float = DEFAULT_POOR_QUALITY_THRESHOLD

    def __post_init__(self) -> None:
        """Validate the poor-quality threshold immediately after construction.

        Raises:
            ValidationError: If ``poor_quality_threshold`` is not in ``(0, 1]``.
        """
        if not (0.0 < self.poor_quality_threshold <= 1.0):
            raise ValidationError(
                "poor_quality_threshold must be in (0, 1], got "
                f"{self.poor_quality_threshold}."
            )

    def evaluate(self, mesh: Mesh) -> MeshQualityReport:
        """Evaluate every quality-evaluable element in a mesh.

        Args:
            mesh: The mesh to evaluate.

        Returns:
            A :class:`~femtoolkit.mesh.quality.quality_report.MeshQualityReport`.

        Raises:
            UnsupportedQualityMetricError: If the mesh contains no
                element with a defined shape-quality metric.
        """
        element_qualities: dict[int, ElementQuality | SolidElementQuality] = {}
        for element in mesh.elements:
            try:
                element_qualities[element.id] = compute_any_element_quality(element)
            except UnsupportedQualityMetricError:
                continue

        if not element_qualities:
            raise UnsupportedQualityMetricError(
                "Mesh contains no element with a defined shape-quality metric "
                "(CST, Q4, TET4, or HEX8)."
            )

        quality_values = {
            element_id: quality.quality for element_id, quality in element_qualities.items()
        }
        poor_quality_ids = sorted(
            element_id
            for element_id, value in quality_values.items()
            if value < self.poor_quality_threshold
        )

        warnings: list[str] = []
        if poor_quality_ids:
            warnings.append(
                f"{len(poor_quality_ids)} element(s) have poor quality "
                f"(below {self.poor_quality_threshold:g}): {poor_quality_ids}"
            )

        values = list(quality_values.values())
        return MeshQualityReport(
            element_qualities=element_qualities,
            num_elements_evaluated=len(element_qualities),
            minimum_quality=min(values),
            maximum_quality=max(values),
            mean_quality=sum(values) / len(values),
            poor_quality_element_ids=poor_quality_ids,
            warnings=warnings,
        )


__all__ = ["DEFAULT_POOR_QUALITY_THRESHOLD", "QualityEvaluator"]
