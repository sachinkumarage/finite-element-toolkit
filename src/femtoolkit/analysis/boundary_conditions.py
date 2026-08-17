"""Boundary condition representation.

A **boundary condition** prescribes a known value for a degree of freedom,
most commonly a fixed (zero) or prescribed displacement. Without at least
one boundary condition, the global stiffness matrix is singular and the
structural system ``[K]{u} = {F}`` has no unique solution.

:func:`boundary_conditions_for_region` (Version 9) extends this to
**essential boundary conditions applied by named geometric region**
rather than by manually enumerated node IDs: given a mesh and a
:class:`~femtoolkit.geometry.boundary.BoundaryRegion` (e.g.
``Rectangle(width=2.0, height=1.0).boundary("left")``), it finds every
node on that boundary and returns one :class:`BoundaryCondition` per
constrained DOF at each -- still ordinary :class:`BoundaryCondition`
instances, fed to :class:`~femtoolkit.analysis.static_linear.StaticLinearAnalysis`
exactly like any hand-specified one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from femtoolkit.analysis.dof import TranslationDOF, validate_dof
from femtoolkit.exceptions import ValidationError
from femtoolkit.geometry.boundary import BoundaryRegion
from femtoolkit.mesh.mesh import Mesh


@dataclass
class BoundaryCondition:
    """A prescribed displacement value for one degree of freedom of one node.

    Attributes:
        node_id: Positive integer ID of the constrained node.
        dof: DOF direction being constrained, see
            :class:`~femtoolkit.analysis.dof.TranslationDOF`.
        value: Prescribed displacement value for this DOF, in meters. Use
            ``0.0`` for a fixed (fully restrained) support.

    Raises:
        ValidationError: If ``node_id`` is not a positive integer, ``dof``
            is not a valid DOF direction, or ``value`` is not finite.

    Example:
        >>> boundary_condition = BoundaryCondition(node_id=1, dof=0, value=0.0)
    """

    node_id: int
    dof: int
    value: float

    def __post_init__(self) -> None:
        """Validate the boundary condition immediately after construction.

        Raises:
            ValidationError: If ``node_id`` is not a positive integer,
                ``dof`` is not a valid DOF direction, or ``value`` is not a
                finite number.
        """
        if not isinstance(self.node_id, int) or isinstance(self.node_id, bool) or self.node_id <= 0:
            raise ValidationError(
                f"BoundaryCondition node_id must be a positive integer, got {self.node_id!r}."
            )

        self.dof = validate_dof(self.dof)

        if not math.isfinite(self.value):
            raise ValidationError(
                f"BoundaryCondition value must be a finite number, got {self.value}."
            )


def boundary_conditions_for_region(
    mesh: Mesh,
    boundary: BoundaryRegion,
    *,
    ux: float | None = None,
    uy: float | None = None,
    tolerance: float = 1e-9,
) -> list[BoundaryCondition]:
    """Build essential boundary conditions for every node on a named boundary region.

    Args:
        mesh: The mesh to select nodes from.
        boundary: The boundary region to constrain (e.g.
            ``Rectangle(width=2.0, height=1.0).boundary("left")``).
        ux: Prescribed X displacement, in meters, applied to every
            matching node. ``None`` (the default) leaves X unconstrained.
        uy: Prescribed Y displacement, in meters, applied to every
            matching node. ``None`` (the default) leaves Y unconstrained.
        tolerance: Maximum distance, in meters, for a node to be
            considered on the boundary.

    Returns:
        One :class:`BoundaryCondition` per constrained DOF at each
        matching node (so ``len(result) == n_nodes * n_constrained_dofs``).
        Empty if neither ``ux`` nor ``uy`` is given.

    Example:
        >>> domain = Rectangle(width=2.0, height=1.0)
        >>> fixed = boundary_conditions_for_region(mesh, domain.boundary("left"), ux=0.0, uy=0.0)
        >>> roller = boundary_conditions_for_region(mesh, domain.boundary("right"), uy=0.0)
    """
    conditions: list[BoundaryCondition] = []
    for node in mesh.nodes_on_boundary(boundary, tolerance):
        if ux is not None:
            conditions.append(BoundaryCondition(node.id, TranslationDOF.X, ux))
        if uy is not None:
            conditions.append(BoundaryCondition(node.id, TranslationDOF.Y, uy))
    return conditions
