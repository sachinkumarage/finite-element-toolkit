"""Whole-mesh validation.

Most structural validity checks in the toolkit are already enforced
**at construction time**, before an invalid object can ever exist:

* :class:`~femtoolkit.mesh.node.Node` rejects non-finite coordinates.
* :class:`~femtoolkit.mesh.mesh.Mesh.add_node`/:meth:`~femtoolkit.mesh.mesh.Mesh.add_element`
  reject duplicate node/element IDs and elements that reference nodes not
  already in the mesh.
* :class:`~femtoolkit.mesh.cst_element.CSTElement2D` and
  :class:`~femtoolkit.mesh.quad_element.QuadElement2D` reject degenerate,
  self-intersecting, or inverted (clockwise, for Q4) geometry via
  :class:`~femtoolkit.exceptions.DegenerateElementError`.

This module covers the one class of problem that fail-fast construction
*cannot* catch on its own: a check that requires looking at the mesh as a
whole rather than one entity at a time. **Duplicate node coordinates** --
two distinct node IDs placed at the same physical location -- is exactly
this kind of problem: each node is perfectly valid in isolation (a
finite, uniquely-ID'd point), and each element referencing it is
perfectly valid in isolation, yet the *mesh* is geometrically broken (two
supposedly-different points that are actually the same point).

:func:`validate_mesh` also re-checks element geometry (area, node
references) as defense-in-depth. For meshes built through
:mod:`femtoolkit.mesh.generator` or any other code path that only uses
:class:`~femtoolkit.mesh.mesh.Mesh` and the constructors above, every one
of these checks is expected to always pass -- this function exists to
give a clear, direct error on a hand-built or externally loaded (see
:mod:`femtoolkit.mesh.serialization`) mesh that somehow bypassed those
guarantees, rather than an obscure failure deep inside the solver.
"""

from __future__ import annotations

import math

from femtoolkit.exceptions import DuplicateNodeCoordinatesError, ValidationError
from femtoolkit.mesh.mesh import Mesh

_COORDINATE_TOLERANCE_DECIMALS = 9
"""Number of decimal places used to compare node coordinates for equality.

Coordinates are SI meters, so 1e-9 m (1 nanometer) resolution is far
finer than any physically meaningful mesh spacing while still tolerating
ordinary floating-point round-off.
"""


def validate_mesh(mesh: Mesh) -> None:
    """Validate a mesh as a whole, beyond what construction already guarantees.

    Args:
        mesh: The mesh to validate.

    Raises:
        DuplicateNodeCoordinatesError: If two distinct nodes occupy the
            same physical ``(x, y, z)`` location (within
            :data:`_COORDINATE_TOLERANCE_DECIMALS`).
        ValidationError: If an element references a node not present in
            the mesh, or (defense-in-depth; should be unreachable for any
            element type validated at construction) has non-positive area.
    """
    _validate_no_duplicate_coordinates(mesh)
    _validate_element_references_and_area(mesh)


def _validate_no_duplicate_coordinates(mesh: Mesh) -> None:
    seen_coordinates: dict[tuple[float, float, float], int] = {}
    for node in mesh.nodes:
        key = (
            round(node.x, _COORDINATE_TOLERANCE_DECIMALS),
            round(node.y, _COORDINATE_TOLERANCE_DECIMALS),
            round(node.z, _COORDINATE_TOLERANCE_DECIMALS),
        )
        if key in seen_coordinates:
            raise DuplicateNodeCoordinatesError(
                f"Node {node.id} occupies the same location as node "
                f"{seen_coordinates[key]}: ({node.x}, {node.y}, {node.z})."
            )
        seen_coordinates[key] = node.id


def _validate_element_references_and_area(mesh: Mesh) -> None:
    node_ids = {node.id for node in mesh.nodes}
    for element in mesh.elements:
        for node in element.nodes:
            if node.id not in node_ids:
                raise ValidationError(
                    f"Element {element.id} references node {node.id}, which is not "
                    "present in the mesh."
                )

        area = getattr(element, "area", None)
        if area is not None and (not math.isfinite(area) or area <= 0):
            raise ValidationError(f"Element {element.id} has non-positive area ({area}).")
