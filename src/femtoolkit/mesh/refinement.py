"""Mesh refinement (Version 25, spec section 8).

Only :class:`~femtoolkit.mesh.cst_element.CSTElement2D` and
:class:`~femtoolkit.mesh.quad_element.QuadElement2D` are refined --
these are the two element types with a mathematically reliable, uniform
subdivision rule (spec section 8: "Support refinement only for element
types for which reliable refinement rules can be implemented"). TET4
and HEX8 subdivision templates are geometrically more involved (a
tetrahedron's central-octahedron split has several valid diagonal
choices that affect resulting element quality; a hexahedron's 1-to-8
split needs face and volume-center nodes in addition to edge
midpoints), so they are deliberately left for a future version rather
than shipped as an unreliable approximation -- see the Version 26
preview.

**Uniform refinement** (``elements=None``, the default) applies the
standard "1-to-4" quadrisection to every CST/Q4 element in the mesh:

.. code-block:: text

    CST:  T -> 4T                    Q4:  Q -> 4Q
                                            (+ 1 center node)
         v3                              v4 --- e34 --- v3
         /\\                              |          |
       e13 e23                          e14   c    e23
       /    \\                            |          |
     v1--e12--v2                        v1 --- e12 --- v2

    (edge midpoints e_ij; the CST case also connects
    e12-e23-e13 into one "upside-down" center triangle)

**Local (selective) refinement** (``elements={id, ...}``) refines only
the named elements, leaving the rest of the mesh untouched. This can
produce a **non-conforming mesh** (a "hanging node": a midpoint node on
a shared edge that only one side's element actually uses) -- a known,
explicitly documented limitation, not a defect; a fully conformal
local/adaptive refinement scheme is future-version scope (see the
Version 26 preview, and spec section 8's explicit "Do not implement
adaptive error-based refinement yet").

**Shared edges never get a duplicate midpoint node**: refining two
neighboring elements that share an edge reuses the same midpoint node
for both, keyed by the edge's (unordered) node-ID pair -- the same
"resolve shared topology by node-ID identity, not proximity" pattern
:mod:`femtoolkit.mesh.edges` already uses for boundary-edge detection.

Every refined mesh is validated automatically before being returned
(spec section 9), and refined elements inherit their parent's material
and thickness unchanged (spec section 9's "preserve material/region
associations").
"""

from __future__ import annotations

from collections.abc import Iterable

from femtoolkit.exceptions import UnsupportedRefinementError
from femtoolkit.mesh.cst_element import CSTElement2D
from femtoolkit.mesh.mesh import Mesh
from femtoolkit.mesh.node import Node
from femtoolkit.mesh.quad_element import QuadElement2D
from femtoolkit.mesh.validation import validate_mesh

_REFINABLE_ELEMENT_TYPES = (CSTElement2D, QuadElement2D)


class _IdAllocator:
    """Hands out the next unused integer ID, starting just above the current maximum."""

    def __init__(self, existing_ids: Iterable[int]) -> None:
        self._next_id = max(existing_ids, default=0) + 1

    def take(self) -> int:
        allocated = self._next_id
        self._next_id += 1
        return allocated


class _MidpointCache:
    """Ensures each shared edge gets exactly one new midpoint node."""

    def __init__(self, mesh: Mesh, node_ids: _IdAllocator) -> None:
        self._mesh = mesh
        self._node_ids = node_ids
        self._midpoints: dict[frozenset[int], Node] = {}

    def get(self, node_a: Node, node_b: Node) -> Node:
        key = frozenset((node_a.id, node_b.id))
        if key not in self._midpoints:
            midpoint = Node(
                id=self._node_ids.take(),
                x=(node_a.x + node_b.x) / 2.0,
                y=(node_a.y + node_b.y) / 2.0,
                z=(node_a.z + node_b.z) / 2.0,
            )
            self._mesh.add_node(midpoint)
            self._midpoints[key] = midpoint
        return self._midpoints[key]


def _refine_cst(
    element: CSTElement2D, mesh: Mesh, midpoints: _MidpointCache, element_ids: _IdAllocator
) -> None:
    v1, v2, v3 = (mesh.get_node(node.id) for node in element.nodes)
    e12 = midpoints.get(v1, v2)
    e23 = midpoints.get(v2, v3)
    e13 = midpoints.get(v1, v3)

    for corner_nodes in ((v1, e12, e13), (e12, v2, e23), (e13, e23, v3), (e12, e23, e13)):
        mesh.add_element(
            CSTElement2D(
                id=element_ids.take(),
                nodes=corner_nodes,
                material=element.material,
                thickness=element.thickness,
            )
        )


def refine_uniform(mesh: Mesh, elements: Iterable[int] | None = None) -> Mesh:
    """Refine a mesh's CST/Q4 elements by one level of edge-midpoint quadrisection.

    Args:
        mesh: The mesh to refine. Never mutated -- a new
            :class:`~femtoolkit.mesh.mesh.Mesh` is returned.
        elements: If given, only these element IDs are refined (local
            refinement -- see the module docstring's hanging-node
            caveat); every other element is copied through unchanged.
            If ``None`` (the default), every CST/Q4 element is refined
            (uniform refinement).

    Returns:
        The refined, already-validated :class:`~femtoolkit.mesh.mesh.Mesh`.

    Raises:
        UnsupportedRefinementError: If ``elements`` names an element
            that is not a CST or Q4 element, or does not exist.
        ValidationError: If the refined mesh fails validation.
    """
    target_ids = set(elements) if elements is not None else None
    if target_ids is not None:
        for element_id in target_ids:
            element = mesh.get_element(element_id)
            if not isinstance(element, _REFINABLE_ELEMENT_TYPES):
                raise UnsupportedRefinementError(
                    f"Element {element_id} ({type(element).__name__}) has no reliable "
                    "refinement rule; only CSTElement2D and QuadElement2D are refinable."
                )

    node_ids = _IdAllocator(node.id for node in mesh.nodes)
    element_ids = _IdAllocator(element.id for element in mesh.elements)

    refined = Mesh()
    for node in mesh.nodes:
        refined.add_node(Node(id=node.id, x=node.x, y=node.y, z=node.z))

    midpoints = _MidpointCache(refined, node_ids)

    for element in mesh.elements:
        should_refine = target_ids is None or element.id in target_ids
        if not should_refine:
            _copy_element_unchanged(element, refined)
            continue
        if isinstance(element, CSTElement2D):
            _refine_cst(element, refined, midpoints, element_ids)
        elif isinstance(element, QuadElement2D):
            _refine_quad_element(element, refined, midpoints, element_ids, node_ids)
        else:
            raise UnsupportedRefinementError(
                f"Element {element.id} ({type(element).__name__}) has no reliable "
                "refinement rule; only CSTElement2D and QuadElement2D are refinable."
            )

    validate_mesh(refined)
    return refined


def _copy_element_unchanged(element: CSTElement2D | QuadElement2D, mesh: Mesh) -> None:
    nodes = tuple(mesh.get_node(node.id) for node in element.nodes)
    if isinstance(element, CSTElement2D):
        mesh.add_element(
            CSTElement2D(
                id=element.id, nodes=nodes, material=element.material, thickness=element.thickness
            )
        )
    elif isinstance(element, QuadElement2D):
        mesh.add_element(
            QuadElement2D(
                id=element.id, nodes=nodes, material=element.material, thickness=element.thickness
            )
        )
    else:
        raise UnsupportedRefinementError(
            f"Element {element.id} ({type(element).__name__}) cannot be copied unchanged "
            "by the refiner."
        )


def _refine_quad_element(
    element: QuadElement2D,
    mesh: Mesh,
    midpoints: _MidpointCache,
    element_ids: _IdAllocator,
    node_ids: _IdAllocator,
) -> None:
    v1, v2, v3, v4 = (mesh.get_node(node.id) for node in element.nodes)
    e12 = midpoints.get(v1, v2)
    e23 = midpoints.get(v2, v3)
    e34 = midpoints.get(v3, v4)
    e41 = midpoints.get(v4, v1)

    center = Node(
        id=node_ids.take(),
        x=(v1.x + v2.x + v3.x + v4.x) / 4.0,
        y=(v1.y + v2.y + v3.y + v4.y) / 4.0,
        z=(v1.z + v2.z + v3.z + v4.z) / 4.0,
    )
    mesh.add_node(center)

    for corner_nodes in (
        (v1, e12, center, e41),
        (e12, v2, e23, center),
        (center, e23, v3, e34),
        (e41, center, e34, v4),
    ):
        mesh.add_element(
            QuadElement2D(
                id=element_ids.take(),
                nodes=corner_nodes,
                material=element.material,
                thickness=element.thickness,
            )
        )


__all__ = ["refine_uniform"]
