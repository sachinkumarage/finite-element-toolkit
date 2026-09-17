"""Element and mesh quality metrics (Version 8, extended in Version 25).

A finite element mesh's numerical accuracy depends not just on how many
elements it has, but on their *shape*. A poorly shaped element -- one
that is extremely elongated, skewed, or whose isoparametric mapping is
nearly singular -- can locally reduce solution accuracy and, in more
extreme cases, cause conditioning problems in the global stiffness
matrix (see ``docs/meshing.md`` for the full engineering discussion).

This package was a single module (``mesh/quality.py``) through Version
24; it is reorganized here into three focused submodules without
changing any existing public name or import path --
``from femtoolkit.mesh.quality import ElementQuality,
MeshQualitySummary, compute_element_quality,
compute_mesh_quality_summary`` (used by
:meth:`~femtoolkit.mesh.mesh.Mesh.element_quality`/
:meth:`~femtoolkit.mesh.mesh.Mesh.quality_summary` and by
:mod:`femtoolkit.application.model_service`) keeps working unchanged:

* :mod:`femtoolkit.mesh.quality.metrics` -- per-element metrics,
  :class:`ElementQuality` (2D, unchanged) and the new
  :class:`SolidElementQuality` (TET4/HEX8).
* :mod:`femtoolkit.mesh.quality.quality_report` -- whole-mesh reports,
  :class:`MeshQualitySummary` (2D, unchanged) and the new
  :class:`MeshQualityReport` (any supported element mix).
* :mod:`femtoolkit.mesh.quality.evaluator` -- :class:`QualityEvaluator`,
  the new ``evaluator.evaluate(mesh)`` entry point.
"""

from femtoolkit.mesh.quality.evaluator import DEFAULT_POOR_QUALITY_THRESHOLD, QualityEvaluator
from femtoolkit.mesh.quality.metrics import (
    ElementQuality,
    SolidElementQuality,
    compute_any_element_quality,
    compute_element_quality,
    compute_solid_element_quality,
)
from femtoolkit.mesh.quality.quality_report import (
    MeshQualityReport,
    MeshQualitySummary,
    compute_mesh_quality_summary,
)

__all__ = [
    "DEFAULT_POOR_QUALITY_THRESHOLD",
    "ElementQuality",
    "MeshQualityReport",
    "MeshQualitySummary",
    "QualityEvaluator",
    "SolidElementQuality",
    "compute_any_element_quality",
    "compute_element_quality",
    "compute_mesh_quality_summary",
    "compute_solid_element_quality",
]
