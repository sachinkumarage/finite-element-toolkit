"""Degree-of-freedom (DOF) representation.

A **degree of freedom** is an independent unknown associated with a node.
For a 1D axial problem a node has a single translational DOF (its axial
displacement). A 2D truss node activates two translational DOFs (X, Y).
Version 5 introduces a third DOF for 2D frame nodes: the rotation about
the global Z axis, ``rz``, needed alongside ``ux`` and ``uy`` to resist
bending moment.

This module defines:

* :class:`TranslationDOF` — the translational directions a DOF can
  represent (``X``, ``Y``, ``Z``).
* :class:`RotationDOF` — the rotational direction a DOF can represent
  (``RZ``, rotation about the global Z axis).
* :class:`DOFMap` — a mapping from ``(node_id, dof)`` pairs to a single
  global DOF index, used to assemble element contributions into a global
  system of equations.

``DOFMap`` numbers DOF slots purely by position (0, 1, 2, ...) and has no
opinion on what a slot means; that meaning is defined entirely by the
owning element. A 2D frame element (``dofs_per_node=3``) uses slot 2 for
``RotationDOF.RZ``, the same numeric value as ``TranslationDOF.Z``. This
is safe because a single analysis only ever contains elements that share
one ``dofs_per_node`` value (see :class:`~femtoolkit.analysis.static_linear.StaticLinearAnalysis`),
so slot 2 never represents both a Z-translation and a Z-rotation within
the same analysis.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import IntEnum

from femtoolkit.exceptions import EntityNotFoundError, ValidationError


class TranslationDOF(IntEnum):
    """A translational degree-of-freedom direction at a node.

    ``X`` represents axial displacement for the 1D bar element and the
    global X displacement for 2D truss and frame elements. ``Y`` is the
    global Y displacement, activated by 2D truss and frame elements.
    ``Z`` is reserved for a future 3D translational direction.
    """

    X = 0
    Y = 1
    Z = 2


class RotationDOF(IntEnum):
    """A rotational degree-of-freedom direction at a node.

    ``RZ`` is the rotation about the global Z axis, activated by 2D frame
    elements (:class:`~femtoolkit.mesh.frame_element.FrameElement2D`)
    alongside ``TranslationDOF.X`` and ``TranslationDOF.Y``. See the
    module docstring for why sharing numeric value 2 with
    ``TranslationDOF.Z`` is safe.
    """

    RZ = 2


def validate_dof(dof: int) -> int:
    """Validate that ``dof`` identifies a known DOF direction.

    Args:
        dof: Candidate DOF index, expected to match one of the
            :class:`TranslationDOF` values (0, 1, or 2) -- which also
            covers :class:`RotationDOF.RZ` (2).

    Returns:
        The validated DOF index, as a plain ``int``.

    Raises:
        ValidationError: If ``dof`` is not a valid DOF direction.
    """
    if isinstance(dof, bool) or not isinstance(dof, int):
        raise ValidationError(f"DOF must be an integer, got {dof!r}.")
    if dof not in (TranslationDOF.X, TranslationDOF.Y, TranslationDOF.Z):
        raise ValidationError(f"DOF must be one of {[d.value for d in TranslationDOF]}, got {dof}.")
    return int(dof)


class DOFMap:
    """Maps ``(node_id, dof)`` pairs to global DOF indices.

    The map fixes an ordering of node IDs and a number of active DOFs per
    node, then assigns consecutive global indices in that order. For
    example, with two nodes and one DOF per node, node 1's DOF occupies
    global index 0 and node 2's DOF occupies global index 1.

    Attributes:
        node_ids: The ordered, unique node IDs covered by this map.
        dofs_per_node: Number of active translational DOFs at each node
            (1 for the Version 2 axial bar problem).

    Raises:
        ValidationError: If ``node_ids`` is empty, contains duplicates, or
            ``dofs_per_node`` is not in ``{1, 2, 3}``.

    Example:
        >>> dof_map = DOFMap(node_ids=[1, 2], dofs_per_node=1)
        >>> dof_map.global_index(node_id=2, dof=TranslationDOF.X)
        1
    """

    def __init__(self, node_ids: Sequence[int], dofs_per_node: int = 1) -> None:
        """Build a DOF map for the given nodes.

        Args:
            node_ids: Ordered sequence of unique node IDs.
            dofs_per_node: Number of active translational DOFs per node.
                Must be 1, 2, or 3.

        Raises:
            ValidationError: If ``node_ids`` is empty or has duplicates, or
                ``dofs_per_node`` is outside ``{1, 2, 3}``.
        """
        if not node_ids:
            raise ValidationError("DOFMap requires at least one node id.")
        if len(set(node_ids)) != len(node_ids):
            raise ValidationError("DOFMap node ids must be unique.")
        if dofs_per_node not in (1, 2, 3):
            raise ValidationError(f"dofs_per_node must be 1, 2, or 3, got {dofs_per_node}.")

        self.node_ids: tuple[int, ...] = tuple(node_ids)
        self.dofs_per_node = dofs_per_node
        self._node_position = {node_id: position for position, node_id in enumerate(self.node_ids)}

    @property
    def total_dofs(self) -> int:
        """Total number of global DOFs represented by this map."""
        return len(self.node_ids) * self.dofs_per_node

    def global_index(self, node_id: int, dof: int) -> int:
        """Return the global DOF index for a given node and DOF direction.

        Args:
            node_id: ID of the node, must be part of this map.
            dof: DOF direction, validated against :func:`validate_dof` and
                the number of active DOFs per node.

        Returns:
            The global DOF index, in ``range(0, self.total_dofs)``.

        Raises:
            EntityNotFoundError: If ``node_id`` is not part of this map.
            ValidationError: If ``dof`` is invalid or not active for this
                map's ``dofs_per_node``.
        """
        if node_id not in self._node_position:
            raise EntityNotFoundError(f"Node id {node_id} is not part of this DOF map.")

        validated_dof = validate_dof(dof)
        if validated_dof >= self.dofs_per_node:
            raise ValidationError(
                f"DOF {validated_dof} is not active for a map with "
                f"dofs_per_node={self.dofs_per_node}."
            )

        return self._node_position[node_id] * self.dofs_per_node + validated_dof
