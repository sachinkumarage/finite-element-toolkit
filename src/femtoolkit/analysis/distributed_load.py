"""Distributed boundary tractions, converted into equivalent nodal loads.

A :class:`DistributedLoad` describes a traction (force per unit area, in
pascals) acting along a named boundary region -- not a specific list of
nodes. :func:`distributed_load_to_nodal_loads` does the actual work of
turning that into something the existing solver understands: it finds
every boundary edge (see :mod:`femtoolkit.mesh.edges`) lying on the given
region, computes each edge's equivalent nodal force (see
:mod:`femtoolkit.continuum.edge`), and returns plain
:class:`~femtoolkit.analysis.loads.NodalLoad` instances -- the exact same
type used for a hand-specified concentrated force.

No new load-carrying machinery is introduced at the solver level: two
adjacent edges sharing a node each contribute their own
:class:`~femtoolkit.analysis.loads.NodalLoad` at that shared node, and
:func:`~femtoolkit.analysis.system.build_force_vector` (unchanged since
Version 2) already sums multiple loads on the same DOF, so the shared
node correctly receives the total of both edges' contributions without
this module needing to pre-aggregate anything.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from femtoolkit.analysis.dof import TranslationDOF
from femtoolkit.analysis.loads import NodalLoad
from femtoolkit.exceptions import ValidationError
from femtoolkit.geometry.boundary import BoundaryRegion
from femtoolkit.mesh.edges import find_boundary_edges
from femtoolkit.mesh.mesh import Mesh

TractionDirection = Literal["normal", "tangential", "global_x", "global_y", "global"]


@dataclass
class DistributedLoad:
    """A distributed traction acting along a named boundary region.

    Attributes:
        boundary: The boundary region the traction acts on.
        magnitude: The traction's magnitude, in pascals (N/m^2). A plain
            ``float`` for ``"normal"``, ``"tangential"``, ``"global_x"``,
            or ``"global_y"`` directions; a ``(tx, ty)`` tuple of global
            traction components for ``"global"``.
        direction: How ``magnitude`` maps to global ``(tx, ty)``
            components:

            * ``"normal"`` (default) -- along the boundary's outward
              normal (see :attr:`~femtoolkit.geometry.boundary.BoundaryRegion.outward_normal`);
              positive magnitude pulls away from the domain (tension).
            * ``"tangential"`` -- perpendicular to the outward normal
              (rotated 90 degrees counter-clockwise).
            * ``"global_x"`` / ``"global_y"`` -- purely along the global
              X or Y axis.
            * ``"global"`` -- ``magnitude`` is already the ``(tx, ty)``
              vector; no resolution needed.

    Raises:
        ValidationError: If ``direction`` is not one of the supported
            values, or if ``magnitude``'s type does not match ``direction``
            (a 2-tuple for ``"global"``, a plain number otherwise).

    Example:
        >>> domain = Rectangle(width=2.0, height=1.0)
        >>> traction = DistributedLoad(domain.boundary("right"), magnitude=1000.0)
        >>> traction.traction_vector()
        (1000.0, 0.0)
    """

    boundary: BoundaryRegion
    magnitude: float | tuple[float, float]
    direction: TractionDirection = "normal"

    def __post_init__(self) -> None:
        """Validate the distributed load immediately after construction.

        Raises:
            ValidationError: If ``direction`` is invalid or ``magnitude``'s
                type does not match it.
        """
        if self.direction not in ("normal", "tangential", "global_x", "global_y", "global"):
            raise ValidationError(
                f"direction must be one of normal/tangential/global_x/global_y/global, "
                f"got {self.direction!r}."
            )
        if self.direction == "global":
            if not (isinstance(self.magnitude, tuple) and len(self.magnitude) == 2):
                raise ValidationError(
                    f'magnitude must be a (tx, ty) tuple when direction="global", '
                    f"got {self.magnitude!r}."
                )
        elif not isinstance(self.magnitude, (int, float)) or isinstance(self.magnitude, bool):
            raise ValidationError(
                f"magnitude must be a plain number when direction={self.direction!r}, "
                f"got {self.magnitude!r}."
            )

    def traction_vector(self) -> tuple[float, float]:
        """Resolve this load into global traction components ``(tx, ty)``, in pascals."""
        if self.direction == "global":
            return self.magnitude  # type: ignore[return-value]

        magnitude = float(self.magnitude)  # type: ignore[arg-type]
        if self.direction == "global_x":
            return (magnitude, 0.0)
        if self.direction == "global_y":
            return (0.0, magnitude)

        nx, ny = self.boundary.outward_normal
        if self.direction == "normal":
            return (magnitude * nx, magnitude * ny)
        # "tangential": the normal rotated 90 degrees counter-clockwise.
        return (-magnitude * ny, magnitude * nx)


def distributed_load_to_nodal_loads(
    mesh: Mesh, load: DistributedLoad, tolerance: float = 1e-9
) -> list[NodalLoad]:
    """Convert a distributed boundary traction into equivalent nodal loads.

    Finds every boundary edge (see
    :func:`~femtoolkit.mesh.edges.find_boundary_edges`) whose both
    endpoint nodes lie on ``load.boundary``, and computes each edge's
    equivalent nodal force contribution via
    :func:`~femtoolkit.continuum.edge.edge_equivalent_nodal_force`, using
    that edge's own owning element's thickness.

    Args:
        mesh: The mesh to search for boundary edges.
        load: The distributed load to convert.
        tolerance: Maximum distance, in meters, for an edge's nodes to be
            considered on ``load.boundary``.

    Returns:
        One :class:`~femtoolkit.analysis.loads.NodalLoad` per constrained
        DOF contribution (up to 4 per boundary edge: X and Y at each of
        its two nodes). Nodes shared by two boundary edges on the same
        region naturally receive two entries, which
        :func:`~femtoolkit.analysis.system.build_force_vector` sums.
    """
    # Imported locally to avoid a module-level circular import: this
    # module needs the pure edge-integration math, kept in `continuum`.
    from femtoolkit.continuum.edge import edge_equivalent_nodal_force

    tx, ty = load.traction_vector()

    nodal_loads: list[NodalLoad] = []
    for edge in find_boundary_edges(mesh):
        node_a = mesh.get_node(edge.node_ids[0])
        node_b = mesh.get_node(edge.node_ids[1])
        if not (
            load.boundary.contains_point(node_a.x, node_a.y, tolerance)
            and load.boundary.contains_point(node_b.x, node_b.y, tolerance)
        ):
            continue

        element = mesh.get_element(edge.element_id)
        thickness = element.thickness  # type: ignore[attr-defined]
        fx1, fy1, fx2, fy2 = edge_equivalent_nodal_force(
            (node_a.x, node_a.y), (node_b.x, node_b.y), (tx, ty), thickness
        )
        nodal_loads.append(NodalLoad(node_a.id, TranslationDOF.X, fx1))
        nodal_loads.append(NodalLoad(node_a.id, TranslationDOF.Y, fy1))
        nodal_loads.append(NodalLoad(node_b.id, TranslationDOF.X, fx2))
        nodal_loads.append(NodalLoad(node_b.id, TranslationDOF.Y, fy2))

    return nodal_loads
