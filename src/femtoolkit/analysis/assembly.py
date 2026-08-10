"""Global stiffness matrix assembly.

A structure is modeled as several elements connected at shared nodes.
Each element only "knows" about its own local stiffness matrix; the
**assembly** step maps each element's local degrees of freedom onto the
structure's global degrees of freedom and sums the contributions into a
single global stiffness matrix.

Assembly is deliberately independent of any specific element type: a
contribution is just a local stiffness matrix tagged with the
``(node_id, dof)`` pair each row/column corresponds to. This lets the
same assembly logic serve a 1D bar element (two axial DOFs) and a 2D
truss element (four X/Y DOFs) without duplicating the scatter loop.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

import numpy as np

from femtoolkit.analysis.dof import DOFMap
from femtoolkit.exceptions import ValidationError


class ElementStiffnessContribution(NamedTuple):
    """One element's local stiffness matrix, tagged with its global DOF keys.

    Attributes:
        dof_keys: One ``(node_id, dof)`` pair per row/column of
            ``stiffness``, in matching order. For example, a 1D bar
            element contributes two keys (one axial DOF per node); a 2D
            truss element contributes four (X and Y per node).
        stiffness: The element's local stiffness matrix, square with size
            ``len(dof_keys)``.
    """

    dof_keys: tuple[tuple[int, int], ...]
    stiffness: np.ndarray


def assemble_global_stiffness(
    dof_map: DOFMap,
    contributions: Sequence[ElementStiffnessContribution],
) -> np.ndarray:
    """Assemble a global stiffness matrix from element contributions.

    Each contribution's local stiffness matrix is scattered into the
    global stiffness matrix at the rows/columns given by mapping its
    ``dof_keys`` through ``dof_map``, and overlapping contributions from
    elements that share a DOF are summed.

    Args:
        dof_map: DOF map describing the global DOF numbering for all
            nodes involved.
        contributions: One :class:`ElementStiffnessContribution` per
            element in the model.

    Returns:
        The assembled global stiffness matrix, of shape
        ``(dof_map.total_dofs, dof_map.total_dofs)``.

    Raises:
        ValidationError: If a contribution's stiffness matrix shape does
            not match its number of ``dof_keys``.
        EntityNotFoundError: If a contribution references a node ID that
            is not part of ``dof_map``.

    Example:
        >>> dof_map = DOFMap(node_ids=[1, 2, 3], dofs_per_node=1)
        >>> k_1 = bar_element_stiffness(200e9, 0.01, 1.0)
        >>> k_2 = bar_element_stiffness(200e9, 0.01, 1.0)
        >>> assemble_global_stiffness(
        ...     dof_map,
        ...     [
        ...         ElementStiffnessContribution(((1, 0), (2, 0)), k_1),
        ...         ElementStiffnessContribution(((2, 0), (3, 0)), k_2),
        ...     ],
        ... )
    """
    global_stiffness = np.zeros((dof_map.total_dofs, dof_map.total_dofs))

    for dof_keys, local_stiffness in contributions:
        expected_shape = (len(dof_keys), len(dof_keys))
        if local_stiffness.shape != expected_shape:
            raise ValidationError(
                f"Element stiffness matrix must have shape {expected_shape} to "
                f"match {len(dof_keys)} dof_keys, got {local_stiffness.shape}."
            )

        global_indices = [dof_map.global_index(node_id, dof) for node_id, dof in dof_keys]

        for local_row, global_row in enumerate(global_indices):
            for local_col, global_col in enumerate(global_indices):
                global_stiffness[global_row, global_col] += local_stiffness[local_row, local_col]

    return global_stiffness
