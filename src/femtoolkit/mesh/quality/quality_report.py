"""Whole-mesh quality reports (Version 8's `MeshQualitySummary`, extended in Version 25).

Two typed result objects, deliberately kept separate:

* :class:`MeshQualitySummary` -- Version 8's original, 2D-only
  (CST/Q4) whole-mesh statistic, unchanged, still produced by
  :func:`compute_mesh_quality_summary`. Every Version 8-24 caller
  (including :meth:`~femtoolkit.mesh.mesh.Mesh.quality_summary`) keeps
  working exactly as before.
* :class:`MeshQualityReport` -- new in Version 25, produced by
  :class:`~femtoolkit.mesh.quality.evaluator.QualityEvaluator`. Works
  across *any* mix of 2D continuum and 3D solid elements in one mesh,
  and additionally identifies specific poor-quality elements and raises
  human-readable warnings, rather than only reporting aggregate
  statistics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from femtoolkit.exceptions import ValidationError
from femtoolkit.mesh.cst_element import CSTElement2D
from femtoolkit.mesh.quad_element import QuadElement2D
from femtoolkit.mesh.quality.metrics import (
    ElementQuality,
    SolidElementQuality,
    compute_element_quality,
)

if TYPE_CHECKING:
    from femtoolkit.mesh.mesh import Mesh


@dataclass(frozen=True)
class MeshQualitySummary:
    """Whole-mesh shape-quality summary, aggregated over its continuum elements.

    Attributes:
        num_nodes: Total number of nodes in the mesh.
        num_elements: Total number of elements in the mesh (of any type).
        min_area: Smallest element area, in square meters.
        max_area: Largest element area, in square meters.
        min_edge_length: Shortest edge across all elements, in meters.
        max_edge_length: Longest edge across all elements, in meters.
        min_quality: Worst (smallest) per-element quality value.
        max_quality: Best (largest) per-element quality value.
        average_quality: Mean per-element quality value.
        num_invalid_elements: Number of elements with non-positive or
            non-finite area. Elements are validated at construction time
            (see :mod:`femtoolkit.mesh.cst_element`/:mod:`femtoolkit.mesh.quad_element`),
            so for any mesh built through normal APIs this is expected to
            always be 0; the check remains as defense-in-depth.
    """

    num_nodes: int
    num_elements: int
    min_area: float
    max_area: float
    min_edge_length: float
    max_edge_length: float
    min_quality: float
    max_quality: float
    average_quality: float
    num_invalid_elements: int


@dataclass(frozen=True)
class MeshQualityReport:
    """A structured, whole-mesh quality report (Version 25, spec section 5).

    Unlike :class:`MeshQualitySummary`, this works across a mesh
    containing any mix of 2D continuum (CST/Q4) and 3D solid (TET4/HEX8)
    elements, and identifies specific problem elements rather than only
    aggregate numbers.

    Attributes:
        element_qualities: Every evaluated element's quality metrics,
            keyed by element ID.
        num_elements_evaluated: How many elements were evaluated (bar,
            truss, and frame elements have no shape-quality concept and
            are excluded, not counted as an error).
        minimum_quality: Worst (smallest) per-element ``quality`` value.
        maximum_quality: Best (largest) per-element ``quality`` value.
        mean_quality: Mean per-element ``quality`` value.
        poor_quality_element_ids: IDs of elements whose ``quality`` fell
            below the evaluator's ``poor_quality_threshold``.
        warnings: Human-readable warnings (e.g. "N elements have poor
            quality"). Empty when nothing noteworthy was found.
    """

    element_qualities: dict[int, ElementQuality | SolidElementQuality]
    num_elements_evaluated: int
    minimum_quality: float
    maximum_quality: float
    mean_quality: float
    poor_quality_element_ids: list[int] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def has_poor_quality_elements(self) -> bool:
        """Whether any element fell below the poor-quality threshold."""
        return bool(self.poor_quality_element_ids)


def compute_mesh_quality_summary(mesh: Mesh) -> MeshQualitySummary:
    """Compute a whole-mesh shape-quality summary over its continuum elements.

    Unchanged since Version 8 -- see the module docstring for why this
    stays 2D-only (CST/Q4) rather than being generalized; use
    :class:`~femtoolkit.mesh.quality.evaluator.QualityEvaluator` for a
    mesh that may also contain TET4/HEX8 elements.

    Non-continuum elements (bar, truss, frame) are counted in
    ``num_nodes``/``num_elements`` but have no shape-quality concept and
    are excluded from the area/edge/quality statistics.

    Args:
        mesh: The mesh to summarize.

    Returns:
        The mesh's :class:`MeshQualitySummary`.

    Raises:
        ValidationError: If the mesh contains no CST or Q4 elements.
    """
    qualities = [
        compute_element_quality(element)
        for element in mesh.elements
        if isinstance(element, (CSTElement2D, QuadElement2D))
    ]
    if not qualities:
        raise ValidationError(
            "Mesh contains no continuum (CST or Q4) elements to compute a quality summary for."
        )

    num_invalid_elements = sum(
        1 for quality in qualities if not math.isfinite(quality.area) or quality.area <= 0
    )
    areas = [quality.area for quality in qualities]
    quality_values = [quality.quality for quality in qualities]

    return MeshQualitySummary(
        num_nodes=len(mesh.nodes),
        num_elements=len(mesh.elements),
        min_area=min(areas),
        max_area=max(areas),
        min_edge_length=min(quality.min_edge_length for quality in qualities),
        max_edge_length=max(quality.max_edge_length for quality in qualities),
        min_quality=min(quality_values),
        max_quality=max(quality_values),
        average_quality=sum(quality_values) / len(quality_values),
        num_invalid_elements=num_invalid_elements,
    )


__all__ = ["MeshQualityReport", "MeshQualitySummary", "compute_mesh_quality_summary"]
