"""Static linear structural analysis workflow.

This module defines :class:`StaticLinearAnalysis`, which orchestrates the
reusable mathematical foundation (DOF mapping, stiffness assembly, the
linear system, and the solver) against a :class:`~femtoolkit.mesh.mesh.Mesh`
of elements. The analysis itself performs no element-specific
computation; it works against any element that satisfies the
:class:`~femtoolkit.analysis.element.AssemblableElement` protocol --
the minimal interface needed to assemble and solve, satisfied by every
element type in the toolkit (:class:`~femtoolkit.mesh.bar_element.BarElement`,
:class:`~femtoolkit.mesh.truss_element.TrussElement2D`,
:class:`~femtoolkit.mesh.frame_element.FrameElement2D`, and
:class:`~femtoolkit.mesh.cst_element.CSTElement2D`) without this module
importing any of them by name. Richer post-processing capabilities
(scalar axial results, vector continuum strain/stress) are handled by
:class:`~femtoolkit.results.analysis_result.AnalysisResult`, not here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from femtoolkit.analysis.assembly import ElementStiffnessContribution, assemble_global_stiffness
from femtoolkit.analysis.boundary_conditions import BoundaryCondition
from femtoolkit.analysis.dof import DOFMap
from femtoolkit.analysis.element import AssemblableElement
from femtoolkit.analysis.loads import NodalLoad
from femtoolkit.analysis.system import LinearSystem, build_force_vector, solve
from femtoolkit.exceptions import (
    InsufficientConstraintsError,
    InvalidAnalysisError,
    InvalidElementError,
    SingularSystemError,
)

if TYPE_CHECKING:
    # Imported only for type checking: femtoolkit.mesh and
    # femtoolkit.results both interact with femtoolkit.analysis, so a
    # module-level import here would create a circular import. The real
    # imports happen inside solve(), once every package has finished
    # loading.
    from femtoolkit.mesh.mesh import Mesh
    from femtoolkit.results.analysis_result import AnalysisResult


class StaticLinearAnalysis:
    """A linear static structural analysis over a mesh of structural elements.

    The analysis assumes linear elastic material behavior, small
    deformations, and static (time-invariant) loading. It solves the
    equilibrium equation ``[K]{u} = {F}`` for the mesh's elements and
    reports displacements, reactions, and per-element strain, stress, and
    axial force through the returned :class:`AnalysisResult`.

    All elements in the mesh must activate the same number of DOFs per
    node (for example, a mesh of :class:`~femtoolkit.mesh.bar_element.BarElement`
    instances, or a mesh of :class:`~femtoolkit.mesh.truss_element.TrussElement2D`
    instances, but not a mix of both).

    Example:
        >>> analysis = StaticLinearAnalysis(mesh)
        >>> analysis.add_boundary_condition(BoundaryCondition(node_id=1, dof=0, value=0.0))
        >>> analysis.add_load(NodalLoad(node_id=2, dof=0, value=1000.0))
        >>> result = analysis.solve()
    """

    def __init__(self, mesh: Mesh) -> None:
        """Create an analysis for the given mesh.

        Args:
            mesh: The mesh to analyze. Its elements must all satisfy the
                :class:`~femtoolkit.analysis.element.AssemblableElement`
                protocol and share the same ``dofs_per_node``; this is
                checked when :meth:`solve` is called.
        """
        self._mesh = mesh
        self._loads: list[NodalLoad] = []
        self._boundary_conditions: list[BoundaryCondition] = []

    def add_load(self, load: NodalLoad) -> None:
        """Add a nodal load to the analysis.

        Args:
            load: The nodal load to apply.
        """
        self._loads.append(load)

    def add_boundary_condition(self, boundary_condition: BoundaryCondition) -> None:
        """Add a prescribed-displacement boundary condition to the analysis.

        Args:
            boundary_condition: The boundary condition to apply.
        """
        self._boundary_conditions.append(boundary_condition)

    def solve(self) -> AnalysisResult:
        """Assemble and solve the structural system for this analysis.

        Returns:
            The :class:`~femtoolkit.results.analysis_result.AnalysisResult`
            containing displacements, reactions, and per-element results.

        Raises:
            InvalidAnalysisError: If the mesh has no nodes or no elements.
            InvalidElementError: If the mesh contains an element that does
                not satisfy the
                :class:`~femtoolkit.analysis.element.AssemblableElement`
                protocol, or elements with inconsistent ``dofs_per_node``.
            InsufficientConstraintsError: If no boundary conditions have
                been added.
            SingularSystemError: If the structure is insufficiently
                constrained for the reduced stiffness matrix to be solved
                (e.g. an unsupported mechanism).
        """
        # Imported locally to avoid the circular import described above.
        from femtoolkit.results.analysis_result import AnalysisResult

        nodes = self._mesh.nodes
        elements = self._mesh.elements

        if not nodes:
            raise InvalidAnalysisError("Cannot solve an analysis whose mesh has no nodes.")
        if not elements:
            raise InvalidAnalysisError("Cannot solve an analysis whose mesh has no elements.")
        for element in elements:
            if not isinstance(element, AssemblableElement):
                raise InvalidElementError(
                    "StaticLinearAnalysis only supports elements implementing the "
                    f"assemblable element interface, got {type(element).__name__} "
                    f"(id={element.id})."
                )

        dofs_per_node_values = {element.dofs_per_node for element in elements}
        if len(dofs_per_node_values) > 1:
            raise InvalidElementError(
                "StaticLinearAnalysis requires all elements in a mesh to use the "
                f"same number of DOFs per node, got: {sorted(dofs_per_node_values)}."
            )
        dofs_per_node = dofs_per_node_values.pop()

        if not self._boundary_conditions:
            raise InsufficientConstraintsError(
                "StaticLinearAnalysis requires at least one boundary condition."
            )

        dof_map = DOFMap(node_ids=[node.id for node in nodes], dofs_per_node=dofs_per_node)

        contributions = [
            ElementStiffnessContribution(element.dof_keys(), element.stiffness_matrix)
            for element in elements
        ]
        global_stiffness = assemble_global_stiffness(dof_map, contributions)
        forces = build_force_vector(dof_map, self._loads)

        system = LinearSystem(
            dof_map=dof_map,
            stiffness=global_stiffness,
            forces=forces,
            boundary_conditions=self._boundary_conditions,
        )

        try:
            displacements = solve(system)
        except np.linalg.LinAlgError as error:
            raise SingularSystemError(
                "The global stiffness matrix is singular: the structure is "
                "insufficiently constrained (e.g. a free mechanism) even "
                "though boundary conditions were provided."
            ) from error

        reactions = global_stiffness @ displacements - forces

        return AnalysisResult(
            dof_map=dof_map,
            elements=tuple(elements),
            displacements=displacements,
            reactions=reactions,
        )
