"""Multi-point constraints (MPCs): equal-displacement ties between two DOFs.

A **multi-point constraint** relates two degrees of freedom together,
rather than prescribing a single DOF's value directly (that is what
:class:`~femtoolkit.analysis.boundary_conditions.BoundaryCondition`
does). :class:`MultiPointConstraint` supports the simplest and most
common case -- **equal displacement** at the same DOF direction on two
different nodes, e.g. ``ux(node_a) = ux(node_b)`` -- letting two nodes
(from the same mesh, or from two independently meshed regions sharing a
coincident location) move together as one.

This is deliberately minimal: no rigid-body constraints (fixed offset,
rotation-coupled motion) and no contact (inequality, inequality-switching
constraints). Both are out of scope for this version.

**Enforcement: the penalty method.** Rather than eliminating a DOF from
the system (which would require reworking :class:`~femtoolkit.analysis.dof.DOFMap`
and the assembly/reduction pipeline), a constraint is enforced by adding
a very stiff fictitious spring between the two DOFs directly to the
assembled global stiffness matrix, before boundary conditions are
applied:

.. code-block:: text

    K[a,a] += k_p      K[b,b] += k_p
    K[a,b] -= k_p      K[b,a] -= k_p

This adds the energy term ``0.5 * k_p * (u_a - u_b)^2`` to the system,
which is minimized (driving ``u_a`` toward ``u_b``) as ``k_p`` grows large
relative to the structure's own stiffness. This is an **approximate**,
standard finite-element technique (see e.g. Cook, Malkus, Plesha &
Witt, *Concepts and Applications of Finite Element Analysis*): the
constraint is satisfied to within a small residual, not bit-for-bit
exactly, controlled by :data:`PENALTY_FACTOR`. A very large hardcoded
absolute penalty could produce an ill-conditioned matrix that overwhelms
the structure's own stiffness in floating-point terms, so the penalty is
instead scaled relative to the assembled stiffness matrix's own
magnitude, keeping the technique consistent across problems of very
different physical stiffness or unit scale.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from femtoolkit.analysis.dof import DOFMap, validate_dof
from femtoolkit.exceptions import ValidationError

PENALTY_FACTOR: float = 1.0e7
"""Multiplier applied to the assembled stiffness matrix's largest diagonal
entry to obtain the constraint's penalty stiffness ``k_p``. Large enough
that the constraint is satisfied to a very close approximation for
typical engineering stiffness/load magnitudes, while remaining small
enough (relative to double-precision floating point) that the reduced
system stays well-conditioned.
"""


@dataclass
class MultiPointConstraint:
    """An equal-displacement tie between the same DOF direction at two nodes.

    Enforces ``u(node_id_a, dof) == u(node_id_b, dof)``, e.g.
    ``ux(node1) = ux(node2)``.

    Attributes:
        node_id_a: Positive integer ID of the first tied node.
        node_id_b: Positive integer ID of the second tied node. Must
            differ from ``node_id_a``.
        dof: DOF direction tied between the two nodes, see
            :class:`~femtoolkit.analysis.dof.TranslationDOF`.

    Raises:
        ValidationError: If ``node_id_a``/``node_id_b`` are not distinct
            positive integers, or ``dof`` is not a valid DOF direction.

    Example:
        >>> tie = MultiPointConstraint(node_id_a=1, node_id_b=2, dof=TranslationDOF.X)
    """

    node_id_a: int
    node_id_b: int
    dof: int

    def __post_init__(self) -> None:
        """Validate the constraint immediately after construction.

        Raises:
            ValidationError: If ``node_id_a``/``node_id_b`` are not
                distinct positive integers, or ``dof`` is invalid.
        """
        for name, node_id in (("node_id_a", self.node_id_a), ("node_id_b", self.node_id_b)):
            if not isinstance(node_id, int) or isinstance(node_id, bool) or node_id <= 0:
                raise ValidationError(
                    f"MultiPointConstraint {name} must be a positive integer, got {node_id!r}."
                )
        if self.node_id_a == self.node_id_b:
            raise ValidationError(
                "MultiPointConstraint requires two distinct nodes, got "
                f"node_id_a == node_id_b == {self.node_id_a}."
            )
        self.dof = validate_dof(self.dof)


def apply_multi_point_constraints(
    dof_map: DOFMap,
    stiffness: np.ndarray,
    constraints: Sequence[MultiPointConstraint],
) -> np.ndarray:
    """Augment a global stiffness matrix with penalty stiffness for each constraint.

    Args:
        dof_map: DOF map defining the global DOF numbering ``stiffness``
            is expressed in.
        stiffness: The assembled global stiffness matrix, before boundary
            conditions are applied.
        constraints: Multi-point constraints to enforce.

    Returns:
        A new stiffness matrix (``stiffness`` is not modified in place)
        with penalty terms added for every constraint. Identical to
        ``stiffness`` (same object) if ``constraints`` is empty.

    Raises:
        EntityNotFoundError: If a constraint references a node not in
            ``dof_map``.
        ValidationError: If a constraint's ``dof`` is not active for
            ``dof_map``.
    """
    if not constraints:
        return stiffness

    augmented = stiffness.copy()
    max_diagonal = float(np.max(np.abs(np.diag(stiffness))))
    penalty = PENALTY_FACTOR * (max_diagonal if max_diagonal > 0.0 else 1.0)

    for constraint in constraints:
        index_a = dof_map.global_index(constraint.node_id_a, constraint.dof)
        index_b = dof_map.global_index(constraint.node_id_b, constraint.dof)
        augmented[index_a, index_a] += penalty
        augmented[index_b, index_b] += penalty
        augmented[index_a, index_b] -= penalty
        augmented[index_b, index_a] -= penalty

    return augmented
