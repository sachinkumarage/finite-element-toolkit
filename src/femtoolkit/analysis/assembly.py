"""Global stiffness matrix assembly.

A structure is modeled as several elements connected at shared nodes.
Each element only "knows" about its own local stiffness matrix; the
**assembly** step maps each element's local degrees of freedom onto the
structure's global degrees of freedom and sums the contributions into a
single global stiffness matrix.

Version 2 limits assembly to two-node 1D axial bar elements, each
contributing a single axial DOF per node (see
:func:`~femtoolkit.analysis.stiffness.bar_element_stiffness`).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

import numpy as np

from femtoolkit.analysis.dof import DOFMap, TranslationDOF
from femtoolkit.exceptions import ValidationError

_BAR_ELEMENT_SHAPE = (2, 2)


class ElementStiffnessContribution(NamedTuple):
    """A single element's local stiffness matrix, tagged with its node IDs.

    Attributes:
        node_ids: The two node IDs the bar element connects, in the same
            order used to build ``stiffness`` (row/column 0 corresponds to
            ``node_ids[0]``, row/column 1 to ``node_ids[1]``).
        stiffness: The element's local 2x2 stiffness matrix, as produced
            by :func:`~femtoolkit.analysis.stiffness.bar_element_stiffness`.
    """

    node_ids: tuple[int, int]
    stiffness: np.ndarray


def assemble_global_stiffness(
    dof_map: DOFMap,
    contributions: Sequence[ElementStiffnessContribution],
) -> np.ndarray:
    """Assemble a global stiffness matrix from 1D bar element contributions.

    Each contribution's 2x2 local stiffness matrix is scattered into the
    global stiffness matrix at the rows/columns given by mapping its node
    IDs through ``dof_map`` (using the axial ``TranslationDOF.X`` DOF),
    and overlapping contributions from elements that share a node are
    summed.

    Args:
        dof_map: DOF map describing the global DOF numbering for all
            nodes involved. Must have been built with ``dofs_per_node=1``,
            since bar elements have a single axial DOF per node.
        contributions: One :class:`ElementStiffnessContribution` per bar
            element in the model.

    Returns:
        The assembled global stiffness matrix, of shape
        ``(dof_map.total_dofs, dof_map.total_dofs)``.

    Raises:
        ValidationError: If any contribution's stiffness matrix is not
            2x2, or if ``dof_map`` was not built with a single DOF per
            node.
        EntityNotFoundError: If a contribution references a node ID that
            is not part of ``dof_map``.

    Example:
        >>> dof_map = DOFMap(node_ids=[1, 2, 3], dofs_per_node=1)
        >>> k_1 = bar_element_stiffness(200e9, 0.01, 1.0)
        >>> k_2 = bar_element_stiffness(200e9, 0.01, 1.0)
        >>> assemble_global_stiffness(
        ...     dof_map,
        ...     [
        ...         ElementStiffnessContribution((1, 2), k_1),
        ...         ElementStiffnessContribution((2, 3), k_2),
        ...     ],
        ... )
    """
    if dof_map.dofs_per_node != 1:
        raise ValidationError(
            "assemble_global_stiffness requires a DOF map with dofs_per_node=1 for 1D bar elements."
        )

    global_stiffness = np.zeros((dof_map.total_dofs, dof_map.total_dofs))

    for node_ids, local_stiffness in contributions:
        if local_stiffness.shape != _BAR_ELEMENT_SHAPE:
            raise ValidationError(
                f"Bar element stiffness matrices must have shape {_BAR_ELEMENT_SHAPE}, "
                f"got {local_stiffness.shape}."
            )

        global_indices = [dof_map.global_index(node_id, TranslationDOF.X) for node_id in node_ids]

        for local_row, global_row in enumerate(global_indices):
            for local_col, global_col in enumerate(global_indices):
                global_stiffness[global_row, global_col] += local_stiffness[local_row, local_col]

    return global_stiffness
