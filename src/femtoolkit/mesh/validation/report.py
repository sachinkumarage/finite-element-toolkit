"""Structured, non-raising mesh validation reports (Version 25, spec sections 6-7).

:func:`~femtoolkit.mesh.validation.validate_mesh` (unchanged since
Version 8) is a **fail-fast** check: it raises on the first problem it
finds, which is exactly right for "should the solver trust this mesh?"
gates like :mod:`femtoolkit.mesh.generator`'s own output check. This
module adds a complementary, **non-raising** sweep that collects every
problem it can find and returns them as one structured
:class:`MeshValidationReport` -- the engineering-report style spec
section 7 asks for ("Do not automatically delete questionable data. The
validation system should report problems clearly").

Most of the individual checks here can never actually fail for a mesh
built through this toolkit's normal APIs (:class:`~femtoolkit.mesh.mesh.Mesh`
already guards against invalid element/node references at construction
time, and every element type validates its own geometry in
``__post_init__``) -- the same honest caveat
:func:`~femtoolkit.mesh.validation.validate_mesh`'s own module docstring
makes. They remain valuable for a mesh reconstructed from external data
(:mod:`femtoolkit.mesh.serialization`) or produced by future,
less-constrained generators.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from femtoolkit.exceptions import UnsupportedQualityMetricError
from femtoolkit.mesh.cst_element import CSTElement2D
from femtoolkit.mesh.hex8_element import Hex8Element3D
from femtoolkit.mesh.quad_element import QuadElement2D
from femtoolkit.mesh.quality.evaluator import DEFAULT_POOR_QUALITY_THRESHOLD, QualityEvaluator
from femtoolkit.mesh.quality.quality_report import MeshQualityReport
from femtoolkit.mesh.tet4_element import Tet4Element3D

if TYPE_CHECKING:
    from femtoolkit.mesh.mesh import Mesh

STATUS_OK = "OK"
STATUS_WARNING = "WARNING"
STATUS_ERROR = "ERROR"

_COORDINATE_TOLERANCE_DECIMALS = 9
_VOLUME_ELEMENT_TYPES = (Tet4Element3D, Hex8Element3D)
_AREA_ELEMENT_TYPES = (CSTElement2D, QuadElement2D)


@dataclass(frozen=True)
class MeshValidationReport:
    """A structured mesh validation report (spec section 7).

    Attributes:
        status: ``"OK"`` (no problems found), ``"WARNING"`` (the mesh is
            usable but has quality or non-fatal structural concerns), or
            ``"ERROR"`` (the mesh has a structural problem that should
            be fixed before it is used).
        num_nodes: Total node count.
        num_duplicate_nodes: Nodes occupying the same physical location
            as an earlier node.
        num_isolated_nodes: Nodes referenced by no element.
        num_elements: Total element count.
        num_invalid_connectivity: Elements referencing a node not
            present in the mesh.
        num_degenerate_elements: Elements with a non-positive or
            non-finite area/volume.
        num_duplicate_elements: Elements sharing another element's
            exact node set and type.
        quality: A :class:`~femtoolkit.mesh.quality.quality_report.MeshQualityReport`,
            if the mesh contains at least one quality-evaluable element;
            ``None`` otherwise (e.g. a mesh of only bar elements).
        warnings: Human-readable, non-fatal problem descriptions.
        errors: Human-readable, fatal problem descriptions (only
            populated when ``status == "ERROR"``).
    """

    status: str
    num_nodes: int
    num_duplicate_nodes: int
    num_isolated_nodes: int
    num_elements: int
    num_invalid_connectivity: int
    num_degenerate_elements: int
    num_duplicate_elements: int
    quality: MeshQualityReport | None
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """Whether the mesh has no fatal (``"ERROR"``) problems."""
        return self.status != STATUS_ERROR


def _find_duplicate_node_ids(mesh: Mesh) -> list[int]:
    seen: dict[tuple[float, float, float], int] = {}
    duplicates: list[int] = []
    for node in mesh.nodes:
        key = (
            round(node.x, _COORDINATE_TOLERANCE_DECIMALS),
            round(node.y, _COORDINATE_TOLERANCE_DECIMALS),
            round(node.z, _COORDINATE_TOLERANCE_DECIMALS),
        )
        if key in seen:
            duplicates.append(node.id)
        else:
            seen[key] = node.id
    return duplicates


def _find_isolated_node_ids(mesh: Mesh) -> list[int]:
    referenced = {node.id for element in mesh.elements for node in element.nodes}
    return [node.id for node in mesh.nodes if node.id not in referenced]


def _find_invalid_connectivity_ids(mesh: Mesh) -> list[int]:
    node_ids = {node.id for node in mesh.nodes}
    return [
        element.id
        for element in mesh.elements
        if any(node.id not in node_ids for node in element.nodes)
    ]


def _find_degenerate_element_ids(mesh: Mesh) -> list[int]:
    degenerate = []
    for element in mesh.elements:
        if isinstance(element, _AREA_ELEMENT_TYPES):
            measure = element.area
        elif isinstance(element, _VOLUME_ELEMENT_TYPES):
            measure = element.volume
        else:
            continue
        if not math.isfinite(measure) or measure <= 0:
            degenerate.append(element.id)
    return degenerate


def _find_duplicate_element_ids(mesh: Mesh) -> list[int]:
    seen: dict[tuple[str, frozenset[int]], int] = {}
    duplicates: list[int] = []
    for element in mesh.elements:
        key = (type(element).__name__, frozenset(node.id for node in element.nodes))
        if key in seen:
            duplicates.append(element.id)
        else:
            seen[key] = element.id
    return duplicates


def generate_validation_report(
    mesh: Mesh, poor_quality_threshold: float = DEFAULT_POOR_QUALITY_THRESHOLD
) -> MeshValidationReport:
    """Run a comprehensive, non-raising validity sweep over a mesh.

    Args:
        mesh: The mesh to validate.
        poor_quality_threshold: Forwarded to
            :class:`~femtoolkit.mesh.quality.evaluator.QualityEvaluator`.

    Returns:
        A :class:`MeshValidationReport` describing every problem found.
    """
    duplicate_node_ids = _find_duplicate_node_ids(mesh)
    isolated_node_ids = _find_isolated_node_ids(mesh)
    invalid_connectivity_ids = _find_invalid_connectivity_ids(mesh)
    degenerate_ids = _find_degenerate_element_ids(mesh)
    duplicate_element_ids = _find_duplicate_element_ids(mesh)

    try:
        quality = QualityEvaluator(poor_quality_threshold).evaluate(mesh)
    except UnsupportedQualityMetricError:
        quality = None

    warnings: list[str] = []
    errors: list[str] = []

    if duplicate_node_ids:
        errors.append(f"{len(duplicate_node_ids)} duplicate node(s): {duplicate_node_ids}")
    if invalid_connectivity_ids:
        errors.append(
            f"{len(invalid_connectivity_ids)} element(s) with invalid connectivity: "
            f"{invalid_connectivity_ids}"
        )
    if isolated_node_ids:
        warnings.append(f"{len(isolated_node_ids)} isolated node(s): {isolated_node_ids}")
    if degenerate_ids:
        warnings.append(f"{len(degenerate_ids)} degenerate element(s): {degenerate_ids}")
    if duplicate_element_ids:
        warnings.append(
            f"{len(duplicate_element_ids)} duplicate element(s): {duplicate_element_ids}"
        )
    if quality is not None:
        warnings.extend(quality.warnings)

    if errors:
        status = STATUS_ERROR
    elif warnings:
        status = STATUS_WARNING
    else:
        status = STATUS_OK

    return MeshValidationReport(
        status=status,
        num_nodes=len(mesh.nodes),
        num_duplicate_nodes=len(duplicate_node_ids),
        num_isolated_nodes=len(isolated_node_ids),
        num_elements=len(mesh.elements),
        num_invalid_connectivity=len(invalid_connectivity_ids),
        num_degenerate_elements=len(degenerate_ids),
        num_duplicate_elements=len(duplicate_element_ids),
        quality=quality,
        warnings=warnings,
        errors=errors,
    )


def format_report(report: MeshValidationReport) -> str:
    """Render a :class:`MeshValidationReport` as human-readable text (spec section 7).

    Args:
        report: The report to render.

    Returns:
        A multi-line text block matching this version's specified format.
    """
    lines = [
        "Mesh Validation Report",
        "-" * 28,
        "",
        f"Status: {report.status}",
        "",
        "Nodes:",
        f"  Total: {report.num_nodes}",
        f"  Duplicate: {report.num_duplicate_nodes}",
        f"  Isolated: {report.num_isolated_nodes}",
        "",
        "Elements:",
        f"  Total: {report.num_elements}",
        f"  Invalid connectivity: {report.num_invalid_connectivity}",
        f"  Degenerate: {report.num_degenerate_elements}",
        f"  Duplicate: {report.num_duplicate_elements}",
    ]
    if report.quality is not None:
        lines += [
            "",
            "Quality:",
            f"  Minimum: {report.quality.minimum_quality:.2f}",
            f"  Mean: {report.quality.mean_quality:.2f}",
        ]
    if report.warnings:
        lines += ["", "Warnings:"]
        lines += [f"  {warning}" for warning in report.warnings]
    if report.errors:
        lines += ["", "Errors:"]
        lines += [f"  {error}" for error in report.errors]
    return "\n".join(lines)


__all__ = [
    "STATUS_ERROR",
    "STATUS_OK",
    "STATUS_WARNING",
    "MeshValidationReport",
    "format_report",
    "generate_validation_report",
]
