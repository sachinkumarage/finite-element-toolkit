"""Node data model.

This module defines :class:`Node`, a point in 3D space identified by an
integer ID. Version 1 stores geometry only; degrees of freedom,
displacements, and boundary conditions belong to future versions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from femtoolkit.exceptions import ValidationError


@dataclass
class Node:
    """A point in the finite element model, defined by its coordinates.

    Version 1 represents a node purely as geometry. It does not carry
    displacements, velocities, loads, degrees of freedom, or boundary
    conditions; those belong to future versions of the toolkit.

    Attributes:
        id: Positive integer identifying the node uniquely within a mesh.
        x: X coordinate, in meters (SI units).
        y: Y coordinate, in meters (SI units).
        z: Z coordinate, in meters (SI units).

    Raises:
        ValidationError: If ``id`` is not a positive integer, or if any
            coordinate is not a finite number.

    Example:
        >>> node = Node(id=1, x=0.0, y=0.0, z=0.0)
    """

    id: int
    x: float
    y: float
    z: float

    def __post_init__(self) -> None:
        """Validate the node ID and coordinates immediately after construction.

        Raises:
            ValidationError: If ``id`` is not a positive integer, or any of
                ``x``, ``y``, ``z`` is not finite (e.g. NaN or infinity).
        """
        if not isinstance(self.id, int) or isinstance(self.id, bool) or self.id <= 0:
            raise ValidationError(f"Node id must be a positive integer, got {self.id!r}.")

        for axis_name, value in (("x", self.x), ("y", self.y), ("z", self.z)):
            if not math.isfinite(value):
                raise ValidationError(
                    f"Node coordinate '{axis_name}' must be a finite number, got {value}."
                )
