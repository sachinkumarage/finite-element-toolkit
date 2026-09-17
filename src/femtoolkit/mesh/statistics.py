"""Whole-mesh descriptive statistics (Version 25, spec section 17).

Separate from :mod:`femtoolkit.mesh.quality` (shape quality) and
:mod:`femtoolkit.mesh.validation` (structural validity):
:func:`compute_mesh_statistics` answers purely descriptive questions --
how big is this mesh, what element types does it contain, what physical
space does it occupy -- useful for a mesh-inspection dashboard
regardless of whether the mesh is "good" or "valid".
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from femtoolkit.exceptions import ValidationError
from femtoolkit.mesh.cst_element import CSTElement2D
from femtoolkit.mesh.hex8_element import Hex8Element3D
from femtoolkit.mesh.quad_element import QuadElement2D
from femtoolkit.mesh.tet4_element import Tet4Element3D

if TYPE_CHECKING:
    from femtoolkit.mesh.mesh import Mesh

_AREA_ELEMENT_TYPES = (CSTElement2D, QuadElement2D)
_VOLUME_ELEMENT_TYPES = (Tet4Element3D, Hex8Element3D)


@dataclass(frozen=True)
class MeshStatistics:
    """Descriptive, whole-mesh statistics.

    Attributes:
        num_nodes: Total node count.
        num_elements: Total element count.
        element_type_counts: Maps each element class name (e.g.
            ``"QuadElement2D"``) to how many elements of that type the
            mesh contains.
        dimension: ``"2D"`` if the mesh contains only 2D continuum
            elements (CST/Q4), ``"3D"`` if it contains only 3D solid
            elements (TET4/HEX8), ``"mixed"`` if it contains both (or a
            mesh of only line elements, which has no inherent
            dimensionality of this kind).
        bounding_box_min: The ``(x, y, z)`` minimum over every node, in meters.
        bounding_box_max: The ``(x, y, z)`` maximum over every node, in meters.
        characteristic_size: A representative element size, in meters
            (the bounding box's diagonal length divided by the number of
            elements raised to ``1/dim``, where ``dim`` is 2 or 3;
            ``None`` for a mesh with no continuum/solid elements, since
            "characteristic element size" has no clear meaning for a
            mesh of only line elements).
        num_boundary_edges: The number of boundary edges (see
            :func:`~femtoolkit.mesh.edges.find_boundary_edges`), for a
            2D continuum mesh; ``None`` if the mesh has no 2D continuum
            elements.
    """

    num_nodes: int
    num_elements: int
    element_type_counts: dict[str, int] = field(default_factory=dict)
    dimension: str = "mixed"
    bounding_box_min: tuple[float, float, float] = (0.0, 0.0, 0.0)
    bounding_box_max: tuple[float, float, float] = (0.0, 0.0, 0.0)
    characteristic_size: float | None = None
    num_boundary_edges: int | None = None


def compute_mesh_statistics(mesh: Mesh) -> MeshStatistics:
    """Compute descriptive statistics for a mesh.

    Args:
        mesh: The mesh to summarize.

    Returns:
        The mesh's :class:`MeshStatistics`.

    Raises:
        ValidationError: If the mesh has no nodes.
    """
    if not mesh.nodes:
        raise ValidationError("Cannot compute statistics for a mesh with no nodes.")

    element_type_counts: dict[str, int] = {}
    for element in mesh.elements:
        type_name = type(element).__name__
        element_type_counts[type_name] = element_type_counts.get(type_name, 0) + 1

    has_2d = any(isinstance(element, _AREA_ELEMENT_TYPES) for element in mesh.elements)
    has_3d = any(isinstance(element, _VOLUME_ELEMENT_TYPES) for element in mesh.elements)
    if has_2d and not has_3d:
        dimension = "2D"
    elif has_3d and not has_2d:
        dimension = "3D"
    else:
        dimension = "mixed"

    xs = [node.x for node in mesh.nodes]
    ys = [node.y for node in mesh.nodes]
    zs = [node.z for node in mesh.nodes]
    bounding_box_min = (min(xs), min(ys), min(zs))
    bounding_box_max = (max(xs), max(ys), max(zs))
    diagonal = math.dist(bounding_box_min, bounding_box_max)

    continuum_elements = [
        element
        for element in mesh.elements
        if isinstance(element, _AREA_ELEMENT_TYPES + _VOLUME_ELEMENT_TYPES)
    ]
    characteristic_size = None
    if continuum_elements and diagonal > 0:
        dim = 3 if dimension in ("3D", "mixed") and has_3d else 2
        characteristic_size = diagonal / (len(continuum_elements) ** (1.0 / dim))

    num_boundary_edges = None
    if has_2d:
        from femtoolkit.mesh.edges import find_boundary_edges

        num_boundary_edges = len(find_boundary_edges(mesh))

    return MeshStatistics(
        num_nodes=len(mesh.nodes),
        num_elements=len(mesh.elements),
        element_type_counts=element_type_counts,
        dimension=dimension,
        bounding_box_min=bounding_box_min,
        bounding_box_max=bounding_box_max,
        characteristic_size=characteristic_size,
        num_boundary_edges=num_boundary_edges,
    )


__all__ = ["MeshStatistics", "compute_mesh_statistics"]
