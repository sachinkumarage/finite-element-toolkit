"""Element edge detection.

A 2D continuum element (CST or Q4) is bounded by straight edges between
consecutive nodes -- 3 edges for a triangle, 4 for a quadrilateral. This
module extracts those edges generically (it does not care which element
type it is looking at, only that the element has an ordered polygon of
nodes) and classifies each one as **interior** or **boundary**:

    An edge that belongs to only one element is a boundary edge.
    An edge shared by two elements is an interior edge.

This is a purely topological definition -- it works for any mesh shape,
not just rectangles -- which is what makes it possible to find the mesh
edges lying along a named geometric boundary
(:mod:`femtoolkit.geometry.boundary`) without hard-coding anything about
rectangular domains here. See :mod:`femtoolkit.analysis.distributed_load`
for how boundary edges and named boundaries come together to convert a
distributed traction into equivalent nodal forces.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from femtoolkit.mesh.cst_element import CSTElement2D
from femtoolkit.mesh.mesh import Mesh
from femtoolkit.mesh.quad_element import QuadElement2D

_EDGED_ELEMENT_TYPES = (CSTElement2D, QuadElement2D)


@dataclass(frozen=True)
class ElementEdge:
    """One straight edge of one continuum element.

    Attributes:
        element_id: ID of the element this edge belongs to.
        node_ids: The edge's two endpoint node IDs, in the element's own
            local node order (so ``node_ids[0] -> node_ids[1]`` follows
            the element's counter-clockwise winding).
    """

    element_id: int
    node_ids: tuple[int, int]


def element_edges(element: CSTElement2D | QuadElement2D) -> list[ElementEdge]:
    """Return every edge of a single CST or Q4 element.

    Args:
        element: The element to extract edges from.

    Returns:
        3 edges for a :class:`~femtoolkit.mesh.cst_element.CSTElement2D`,
        4 for a :class:`~femtoolkit.mesh.quad_element.QuadElement2D`, each
        connecting consecutive nodes (wrapping from the last node back to
        the first).
    """
    node_ids = [node.id for node in element.nodes]
    n = len(node_ids)
    return [ElementEdge(element.id, (node_ids[i], node_ids[(i + 1) % n])) for i in range(n)]


def find_boundary_edges(mesh: Mesh) -> list[ElementEdge]:
    """Find every boundary edge (belonging to exactly one element) in a mesh.

    Non-continuum elements (bar, truss, frame) have no polygon edges in
    this sense and are skipped.

    Args:
        mesh: The mesh to search.

    Returns:
        One :class:`ElementEdge` per boundary edge, in a deterministic
        order (following element insertion order, then local edge order
        within each element).
    """
    edges_by_node_pair: dict[frozenset[int], list[ElementEdge]] = defaultdict(list)
    for element in mesh.elements:
        if not isinstance(element, _EDGED_ELEMENT_TYPES):
            continue
        for edge in element_edges(element):
            edges_by_node_pair[frozenset(edge.node_ids)].append(edge)

    return [edges[0] for edges in edges_by_node_pair.values() if len(edges) == 1]
