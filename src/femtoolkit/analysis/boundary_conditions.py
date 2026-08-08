"""Boundary condition representation.

A **boundary condition** prescribes a known value for a degree of freedom,
most commonly a fixed (zero) or prescribed displacement. Without at least
one boundary condition, the global stiffness matrix is singular and the
structural system ``[K]{u} = {F}`` has no unique solution.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from femtoolkit.analysis.dof import validate_dof
from femtoolkit.exceptions import ValidationError


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
