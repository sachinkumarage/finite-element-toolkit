"""The dynamic structural system: ``M u'' + C u' + K u = F(t)``, plus boundary conditions.

:class:`DynamicSystem` mirrors :class:`~femtoolkit.analysis.system.LinearSystem`
(Version 2): a plain data container for the assembled matrices and
boundary conditions a solve step needs, kept separate from the
orchestration logic that builds it
(:func:`build_dynamic_system`/:class:`~femtoolkit.analysis.dynamic_analysis.DynamicAnalysis`,
mirroring how :class:`~femtoolkit.analysis.static_linear.StaticLinearAnalysis`
builds a :class:`~femtoolkit.analysis.system.LinearSystem`) and the
numerical solve step itself
(:mod:`femtoolkit.analysis.newmark`/:mod:`femtoolkit.analysis.modal`,
mirroring :func:`~femtoolkit.analysis.system.solve`).

**Why there is no ``forces`` field, unlike ``LinearSystem``.** A static
analysis's force vector ``{F}`` is a single fixed vector -- exactly what
``LinearSystem.forces`` stores. A dynamic analysis's force is
inherently time-varying, ``F(t)``, evaluated fresh at every time step
from the registered :class:`~femtoolkit.analysis.dynamic_loads.TimeDependentNodalLoad`
objects (see :class:`~femtoolkit.analysis.dynamic_analysis.DynamicAnalysis`).
``DynamicSystem`` therefore holds only the parts of the dynamic equation
that do *not* change from one time step to the next: mass, damping,
stiffness, DOF numbering, and boundary conditions.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from femtoolkit.analysis.assembly import (
    ElementMassContribution,
    ElementStiffnessContribution,
    assemble_global_mass,
    assemble_global_stiffness,
)
from femtoolkit.analysis.boundary_conditions import BoundaryCondition
from femtoolkit.analysis.damping import RayleighDamping
from femtoolkit.analysis.dof import DOFMap
from femtoolkit.analysis.mass import MassMatrixType, element_mass_matrix
from femtoolkit.exceptions import InvalidAnalysisError, InvalidElementError, ValidationError
from femtoolkit.mesh.cst_element import CSTElement2D
from femtoolkit.mesh.quad_element import QuadElement2D

if TYPE_CHECKING:
    from femtoolkit.mesh.mesh import Mesh

_MASS_CAPABLE_ELEMENT_TYPES = (CSTElement2D, QuadElement2D)


@dataclass
class DynamicSystem:
    """The time-invariant part of the dynamic system ``M u'' + C u' + K u = F(t)``.

    Attributes:
        dof_map: DOF map defining the global DOF numbering that
            ``mass``, ``damping``, and ``stiffness`` are expressed in.
        mass: Global mass matrix ``[M]``, of shape
            ``(dof_map.total_dofs, dof_map.total_dofs)``.
        damping: Global damping matrix ``[C]``, the same shape as ``mass``.
        stiffness: Global stiffness matrix ``[K]``, the same shape as ``mass``.
        boundary_conditions: Prescribed-displacement boundary conditions.
            At least one is required for the system to be solvable (the
            same requirement as :class:`~femtoolkit.analysis.system.LinearSystem`).

    Raises:
        ValidationError: If ``mass``, ``damping``, or ``stiffness`` does
            not match the shape implied by ``dof_map``, if
            ``boundary_conditions`` is empty, or if multiple boundary
            conditions target the same DOF.
    """

    dof_map: DOFMap
    mass: np.ndarray
    damping: np.ndarray
    stiffness: np.ndarray
    boundary_conditions: Sequence[BoundaryCondition]

    def __post_init__(self) -> None:
        """Validate matrix shapes and boundary conditions.

        Raises:
            ValidationError: If ``mass``, ``damping``, or ``stiffness``
                is not square with size ``dof_map.total_dofs``, if
                ``boundary_conditions`` is empty, or if two boundary
                conditions target the same global DOF.
        """
        n = self.dof_map.total_dofs
        expected_shape = (n, n)
        named_matrices = (
            ("mass", self.mass),
            ("damping", self.damping),
            ("stiffness", self.stiffness),
        )
        for name, matrix in named_matrices:
            if matrix.shape != expected_shape:
                raise ValidationError(
                    f"DynamicSystem {name} must have shape {expected_shape}, got {matrix.shape}."
                )

        if not self.boundary_conditions:
            raise ValidationError(
                "DynamicSystem requires at least one boundary condition to be solvable."
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


def build_dynamic_system(
    mesh: Mesh,
    boundary_conditions: Sequence[BoundaryCondition],
    *,
    mass_matrix_type: MassMatrixType = "consistent",
    damping: RayleighDamping | None = None,
) -> DynamicSystem:
    """Assemble a :class:`DynamicSystem` from a mesh of mass-capable elements.

    Reuses the exact same DOF mapping and stiffness assembly path as
    :class:`~femtoolkit.analysis.static_linear.StaticLinearAnalysis`
    (:func:`~femtoolkit.analysis.assembly.assemble_global_stiffness`),
    and the new :func:`~femtoolkit.analysis.assembly.assemble_global_mass`
    for mass, guaranteeing ``mass`` and ``stiffness`` share the same
    global DOF numbering.

    Args:
        mesh: The mesh to build a dynamic system for. Every element must
            be a :class:`~femtoolkit.mesh.cst_element.CSTElement2D` or
            :class:`~femtoolkit.mesh.quad_element.QuadElement2D` (the
            only element types with a mass matrix in this version -- see
            :mod:`femtoolkit.analysis.mass`).
        boundary_conditions: Prescribed-displacement boundary conditions.
            At least one is required.
        mass_matrix_type: ``"consistent"`` (default) or ``"lumped"``, see
            :mod:`femtoolkit.analysis.mass`.
        damping: Rayleigh damping to apply, or ``None`` (the default) for
            an undamped system (``C = 0``).

    Returns:
        The assembled :class:`DynamicSystem`.

    Raises:
        InvalidAnalysisError: If the mesh has no nodes or no elements.
        InvalidElementError: If the mesh contains an element that is not
            mass-capable, or elements with inconsistent ``dofs_per_node``.
        ValidationError: If ``boundary_conditions`` is empty, or an
            element's material has no density set.
    """
    nodes = mesh.nodes
    elements = mesh.elements

    if not nodes:
        raise InvalidAnalysisError("Cannot build a dynamic system for a mesh with no nodes.")
    if not elements:
        raise InvalidAnalysisError("Cannot build a dynamic system for a mesh with no elements.")
    for element in elements:
        if not isinstance(element, _MASS_CAPABLE_ELEMENT_TYPES):
            raise InvalidElementError(
                "DynamicSystem only supports mass-capable elements "
                f"(CSTElement2D, QuadElement2D), got {type(element).__name__} "
                f"(id={element.id})."
            )

    dofs_per_node_values = {element.dofs_per_node for element in elements}
    if len(dofs_per_node_values) > 1:
        raise InvalidElementError(
            "build_dynamic_system requires all elements in a mesh to use the same "
            f"number of DOFs per node, got: {sorted(dofs_per_node_values)}."
        )

    dof_map = DOFMap(node_ids=[node.id for node in nodes], dofs_per_node=dofs_per_node_values.pop())

    stiffness_contributions = [
        ElementStiffnessContribution(element.dof_keys(), element.stiffness_matrix)
        for element in elements
    ]
    mass_contributions = [
        ElementMassContribution(element.dof_keys(), element_mass_matrix(element, mass_matrix_type))
        for element in elements
    ]

    stiffness = assemble_global_stiffness(dof_map, stiffness_contributions)
    mass = assemble_global_mass(dof_map, mass_contributions)
    damping_matrix = (
        damping.damping_matrix(mass, stiffness) if damping is not None else np.zeros_like(mass)
    )

    return DynamicSystem(
        dof_map=dof_map,
        mass=mass,
        damping=damping_matrix,
        stiffness=stiffness,
        boundary_conditions=boundary_conditions,
    )


def free_and_constrained_indices(
    system: DynamicSystem,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Partition a system's global DOFs into free and constrained sets.

    The single shared implementation of the free/constrained DOF
    partition every dynamic solve step needs (Newmark time integration,
    modal reduction, harmonic response) -- mirroring the same strategy
    :func:`~femtoolkit.analysis.system.solve` uses for the static case,
    kept in one place so every dynamic analysis path partitions DOFs
    identically.

    Args:
        system: The dynamic system to partition.

    Returns:
        ``(free, constrained, constrained_values)``: ``free`` and
        ``constrained`` are integer arrays of global DOF indices (in
        ascending order for ``free``, in
        ``system.boundary_conditions`` order for ``constrained``);
        ``constrained_values`` holds each constrained DOF's prescribed
        displacement value, in the same order as ``constrained``.

    Raises:
        ValidationError: If every DOF is constrained (no free DOFs remain).
    """
    total_dofs = system.dof_map.total_dofs
    constrained_indices: list[int] = []
    constrained_values: list[float] = []
    for bc in system.boundary_conditions:
        constrained_indices.append(system.dof_map.global_index(bc.node_id, bc.dof))
        constrained_values.append(bc.value)

    constrained = np.array(constrained_indices, dtype=int)
    constrained_set = set(constrained_indices)
    free = np.array([i for i in range(total_dofs) if i not in constrained_set], dtype=int)

    if free.size == 0:
        raise ValidationError("Every DOF is constrained; there is nothing to analyze.")

    return free, constrained, np.array(constrained_values, dtype=float)
