"""Linear system representation and basic solving.

The structural problem solved by finite element analysis reduces to a
single linear system:

.. code-block:: text

    [K]{u} = {F}

where ``[K]`` is the global stiffness matrix, ``{u}`` is the vector of
unknown nodal displacements, and ``{F}`` is the vector of applied nodal
forces. Without boundary conditions ``[K]`` is singular (the structure is
free to translate as a rigid body), so at least one DOF must have a
prescribed displacement before the system can be solved.

This module keeps the data (:class:`LinearSystem`) separate from the
numerical solving step (:func:`solve`), which uses the standard
elimination approach: partition DOFs into "free" and "constrained",
solve the reduced free-free system with NumPy, then reassemble the full
displacement vector.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from femtoolkit.analysis.boundary_conditions import BoundaryCondition
from femtoolkit.analysis.dof import DOFMap
from femtoolkit.analysis.loads import NodalLoad
from femtoolkit.exceptions import ValidationError


@dataclass
class LinearSystem:
    """The structural linear system ``[K]{u} = {F}``, plus its boundary conditions.

    Attributes:
        dof_map: DOF map defining the global DOF numbering that
            ``stiffness`` and ``forces`` are expressed in.
        stiffness: Global stiffness matrix ``[K]``, of shape
            ``(dof_map.total_dofs, dof_map.total_dofs)``.
        forces: Global applied force vector ``{F}``, of shape
            ``(dof_map.total_dofs,)``.
        boundary_conditions: Prescribed-displacement boundary conditions
            applied to this system. At least one is required for the
            system to be solvable.

    Raises:
        ValidationError: If ``stiffness`` or ``forces`` do not match the
            shape implied by ``dof_map``, or if ``boundary_conditions`` is
            empty, or if multiple boundary conditions target the same DOF.

    Example:
        >>> system = LinearSystem(
        ...     dof_map=dof_map,
        ...     stiffness=global_stiffness,
        ...     forces=force_vector,
        ...     boundary_conditions=[BoundaryCondition(node_id=1, dof=0, value=0.0)],
        ... )
    """

    dof_map: DOFMap
    stiffness: np.ndarray
    forces: np.ndarray
    boundary_conditions: Sequence[BoundaryCondition]

    def __post_init__(self) -> None:
        """Validate matrix/vector shapes and boundary conditions.

        Raises:
            ValidationError: If ``stiffness`` is not square with size
                ``dof_map.total_dofs``, ``forces`` does not have length
                ``dof_map.total_dofs``, no boundary conditions were given,
                or two boundary conditions target the same global DOF.
        """
        n = self.dof_map.total_dofs
        if self.stiffness.shape != (n, n):
            raise ValidationError(
                f"LinearSystem stiffness must have shape ({n}, {n}), got {self.stiffness.shape}."
            )
        if self.forces.shape != (n,):
            raise ValidationError(
                f"LinearSystem forces must have shape ({n},), got {self.forces.shape}."
            )
        if not self.boundary_conditions:
            raise ValidationError(
                "LinearSystem requires at least one boundary condition to be solvable."
            )

        seen_global_indices: set[int] = set()
        for boundary_condition in self.boundary_conditions:
            global_index = self.dof_map.global_index(
                boundary_condition.node_id, boundary_condition.dof
            )
            if global_index in seen_global_indices:
                raise ValidationError(
                    f"Multiple boundary conditions target the same DOF "
                    f"(node_id={boundary_condition.node_id}, dof={boundary_condition.dof})."
                )
            seen_global_indices.add(global_index)


def build_force_vector(dof_map: DOFMap, loads: Sequence[NodalLoad]) -> np.ndarray:
    """Assemble a global force vector from a list of nodal loads.

    Args:
        dof_map: DOF map defining the global DOF numbering.
        loads: Nodal loads to place into the vector. Loads on the same
            DOF are summed.

    Returns:
        The global force vector ``{F}``, of shape ``(dof_map.total_dofs,)``.

    Raises:
        EntityNotFoundError: If a load references a node not in ``dof_map``.
        ValidationError: If a load references an invalid or inactive DOF.
    """
    forces = np.zeros(dof_map.total_dofs)
    for load in loads:
        global_index = dof_map.global_index(load.node_id, load.dof)
        forces[global_index] += load.value
    return forces


def solve(system: LinearSystem) -> np.ndarray:
    """Solve the linear system ``[K]{u} = {F}`` for the displacement vector.

    Prescribed DOFs (from ``system.boundary_conditions``) are eliminated
    from the system before solving: the global DOFs are partitioned into
    "free" (unknown) and "constrained" (prescribed) sets, the reduced
    free-free system is solved with :func:`numpy.linalg.solve`, and the
    constrained values are written directly into the result.

    Args:
        system: The linear system to solve, including its boundary
            conditions.

    Returns:
        The full displacement vector ``{u}``, of shape
        ``(system.dof_map.total_dofs,)``, in global DOF order.

    Raises:
        numpy.linalg.LinAlgError: If the reduced free-free stiffness
            matrix is singular (e.g. the free DOFs still form a
            mechanism).
    """
    n = system.dof_map.total_dofs
    displacements = np.zeros(n)

    constrained_indices = []
    for boundary_condition in system.boundary_conditions:
        global_index = system.dof_map.global_index(
            boundary_condition.node_id, boundary_condition.dof
        )
        constrained_indices.append(global_index)
        displacements[global_index] = boundary_condition.value

    constrained = np.array(constrained_indices, dtype=int)
    free = np.array([i for i in range(n) if i not in set(constrained_indices)], dtype=int)

    if free.size == 0:
        return displacements

    k_free_free = system.stiffness[np.ix_(free, free)]
    k_free_constrained = system.stiffness[np.ix_(free, constrained)]
    reduced_forces = system.forces[free] - k_free_constrained @ displacements[constrained]

    displacements[free] = np.linalg.solve(k_free_free, reduced_forces)
    return displacements
