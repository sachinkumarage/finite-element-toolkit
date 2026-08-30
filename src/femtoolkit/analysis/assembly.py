"""Global stiffness and mass matrix assembly.

A structure is modeled as several elements connected at shared nodes.
Each element only "knows" about its own local stiffness (or mass)
matrix; the **assembly** step maps each element's local degrees of
freedom onto the structure's global degrees of freedom and sums the
contributions into a single global matrix.

Assembly is deliberately independent of any specific element type: a
contribution is just a local square matrix tagged with the
``(node_id, dof)`` pair each row/column corresponds to. This lets the
same scatter logic serve a 1D bar element (two axial DOFs) and a 2D
truss element (four X/Y DOFs) without duplicating the loop --
:func:`assemble_global_stiffness` (Version 2) and
:func:`assemble_global_mass` (Version 11) are both thin wrappers over
the same private :func:`_assemble_global_matrix`, so the global mass
matrix is guaranteed to use the exact same global DOF numbering as the
global stiffness matrix, built from the same ``dof_map``.

Nonlinear analysis (Version 13) needs one more assembled quantity: the
global **internal force** vector, ``F_int`` -- an element's resisting
force, not a matrix. :func:`assemble_global_internal_force` scatter-adds
a *vector* the same way, via the analogous private
:func:`_assemble_global_vector`.
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


class ElementMassContribution(NamedTuple):
    """One element's local mass matrix, tagged with its global DOF keys.

    Attributes:
        dof_keys: One ``(node_id, dof)`` pair per row/column of ``mass``,
            in matching order -- the same convention as
            :class:`ElementStiffnessContribution`, typically built from
            the same element's ``dof_keys()``.
        mass: The element's local mass matrix (see
            :mod:`femtoolkit.analysis.mass`), square with size
            ``len(dof_keys)``.
    """

    dof_keys: tuple[tuple[int, int], ...]
    mass: np.ndarray


class ElementForceContribution(NamedTuple):
    """One element's local internal-force vector, tagged with its global DOF keys.

    Attributes:
        dof_keys: One ``(node_id, dof)`` pair per entry of ``force``, in
            matching order -- the same convention as
            :class:`ElementStiffnessContribution`, typically built from
            the same element's ``dof_keys()``.
        force: The element's local internal-force vector (see
            :mod:`femtoolkit.analysis.nonlinear_elements`), length
            ``len(dof_keys)``.
    """

    dof_keys: tuple[tuple[int, int], ...]
    force: np.ndarray


def _assemble_global_matrix(
    dof_map: DOFMap,
    contributions: Sequence[tuple[tuple[tuple[int, int], ...], np.ndarray]],
    matrix_label: str,
) -> np.ndarray:
    """Scatter-add a sequence of ``(dof_keys, local_matrix)`` pairs into a global matrix.

    Shared by :func:`assemble_global_stiffness` and
    :func:`assemble_global_mass`; ``matrix_label`` only affects error
    messages.
    """
    global_matrix = np.zeros((dof_map.total_dofs, dof_map.total_dofs))

    for dof_keys, local_matrix in contributions:
        expected_shape = (len(dof_keys), len(dof_keys))
        if local_matrix.shape != expected_shape:
            raise ValidationError(
                f"Element {matrix_label} matrix must have shape {expected_shape} to "
                f"match {len(dof_keys)} dof_keys, got {local_matrix.shape}."
            )

        global_indices = [dof_map.global_index(node_id, dof) for node_id, dof in dof_keys]

        for local_row, global_row in enumerate(global_indices):
            for local_col, global_col in enumerate(global_indices):
                global_matrix[global_row, global_col] += local_matrix[local_row, local_col]

    return global_matrix


def _assemble_global_vector(
    dof_map: DOFMap,
    contributions: Sequence[tuple[tuple[tuple[int, int], ...], np.ndarray]],
    vector_label: str,
) -> np.ndarray:
    """Scatter-add a sequence of ``(dof_keys, local_vector)`` pairs into a global vector.

    The vector analogue of :func:`_assemble_global_matrix`; contributions
    from elements that share a DOF are summed, exactly like shared
    stiffness/mass entries.
    """
    global_vector = np.zeros(dof_map.total_dofs)

    for dof_keys, local_vector in contributions:
        expected_shape = (len(dof_keys),)
        if local_vector.shape != expected_shape:
            raise ValidationError(
                f"Element {vector_label} vector must have shape {expected_shape} to "
                f"match {len(dof_keys)} dof_keys, got {local_vector.shape}."
            )

        for local_index, (node_id, dof) in enumerate(dof_keys):
            global_index = dof_map.global_index(node_id, dof)
            global_vector[global_index] += local_vector[local_index]

    return global_vector


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
    return _assemble_global_matrix(dof_map, contributions, matrix_label="stiffness")


def assemble_global_mass(
    dof_map: DOFMap,
    contributions: Sequence[ElementMassContribution],
) -> np.ndarray:
    """Assemble a global mass matrix from element contributions.

    Identical scatter-add logic to :func:`assemble_global_stiffness`,
    over the same ``dof_map`` -- the global mass matrix this produces is
    guaranteed to line up DOF-for-DOF with a global stiffness matrix
    assembled from the same ``dof_map``, which is what lets
    :class:`~femtoolkit.analysis.dynamic_system.DynamicSystem` combine
    them directly.

    Args:
        dof_map: DOF map describing the global DOF numbering for all
            nodes involved (the same one used for stiffness assembly).
        contributions: One :class:`ElementMassContribution` per element
            in the model.

    Returns:
        The assembled global mass matrix, of shape
        ``(dof_map.total_dofs, dof_map.total_dofs)``.

    Raises:
        ValidationError: If a contribution's mass matrix shape does not
            match its number of ``dof_keys``.
        EntityNotFoundError: If a contribution references a node ID that
            is not part of ``dof_map``.
    """
    return _assemble_global_matrix(dof_map, contributions, matrix_label="mass")


def assemble_global_internal_force(
    dof_map: DOFMap,
    contributions: Sequence[ElementForceContribution],
) -> np.ndarray:
    """Assemble a global internal-force vector from element contributions.

    Each contribution's local internal-force vector is scattered into
    the global vector at the entries given by mapping its ``dof_keys``
    through ``dof_map``; elements that share a DOF (e.g. two elements
    meeting at a shared node) have their contributions summed, exactly
    like shared stiffness or mass entries.

    Args:
        dof_map: DOF map describing the global DOF numbering for all
            nodes involved (the same one used for stiffness assembly).
        contributions: One :class:`ElementForceContribution` per element
            in the model.

    Returns:
        The assembled global internal-force vector ``F_int``, of shape
        ``(dof_map.total_dofs,)``.

    Raises:
        ValidationError: If a contribution's force vector shape does not
            match its number of ``dof_keys``.
        EntityNotFoundError: If a contribution references a node ID that
            is not part of ``dof_map``.
    """
    return _assemble_global_vector(dof_map, contributions, vector_label="internal-force")
