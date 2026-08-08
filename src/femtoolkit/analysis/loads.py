"""Nodal load representation.

A **nodal load** is an externally applied force acting directly on a
node's degree of freedom. Version 2 supports only this simplest load
type; distributed loads, pressures, thermal loads, and body forces
belong to future versions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from femtoolkit.analysis.dof import validate_dof
from femtoolkit.exceptions import ValidationError


@dataclass
class NodalLoad:
    """An externally applied force on one degree of freedom of one node.

    Attributes:
        node_id: Positive integer ID of the loaded node.
        dof: DOF direction the load acts on, see
            :class:`~femtoolkit.analysis.dof.TranslationDOF`.
        value: Applied force value, in newtons (SI units).

    Raises:
        ValidationError: If ``node_id`` is not a positive integer, ``dof``
            is not a valid DOF direction, or ``value`` is not finite.

    Example:
        >>> load = NodalLoad(node_id=2, dof=0, value=1000.0)
    """

    node_id: int
    dof: int
    value: float

    def __post_init__(self) -> None:
        """Validate the nodal load immediately after construction.

        Raises:
            ValidationError: If ``node_id`` is not a positive integer,
                ``dof`` is not a valid DOF direction, or ``value`` is not a
                finite number.
        """
        if not isinstance(self.node_id, int) or isinstance(self.node_id, bool) or self.node_id <= 0:
            raise ValidationError(
                f"NodalLoad node_id must be a positive integer, got {self.node_id!r}."
            )

        self.dof = validate_dof(self.dof)

        if not math.isfinite(self.value):
            raise ValidationError(f"NodalLoad value must be a finite number, got {self.value}.")
