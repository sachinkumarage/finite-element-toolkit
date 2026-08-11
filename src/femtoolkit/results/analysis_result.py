"""Structural analysis result representation.

This module defines :class:`AnalysisResult`, a read-only view over the
outcome of a :class:`~femtoolkit.analysis.static_linear.StaticLinearAnalysis`
solve. It exposes nodal displacements, reaction forces, and per-element
strain, stress, and axial force, computed on demand from the raw solved
displacement vector. For frame elements, it additionally exposes per-end
shear force, bending moment, and extreme-fiber bending stress. The result
object performs no matrix assembly or solving of its own, and works with
any element satisfying the :class:`~femtoolkit.analysis.element.StructuralElement`
protocol (or, for frame-specific results,
:class:`~femtoolkit.analysis.element.FrameStructuralElement`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from femtoolkit.analysis.dof import DOFMap, TranslationDOF
from femtoolkit.analysis.element import FrameStructuralElement, StructuralElement
from femtoolkit.exceptions import EntityNotFoundError, InvalidElementError, ValidationError


@dataclass(frozen=True)
class AnalysisResult:
    """Read-only results of a solved static linear analysis.

    Instances are produced by
    :meth:`~femtoolkit.analysis.static_linear.StaticLinearAnalysis.solve`
    and should not be constructed directly by application code.

    Attributes:
        dof_map: DOF map used for the analysis, defining the global DOF
            numbering that ``displacements`` and ``reactions`` are
            expressed in.
        elements: The structural elements that were analyzed.
        displacements: Global displacement vector ``{u}``, in global DOF
            order.
        reactions: Global reaction force vector ``R = [K]{u} - {F}``, in
            global DOF order. Nonzero only at constrained DOFs, up to
            floating-point precision.
    """

    dof_map: DOFMap
    elements: tuple[StructuralElement, ...]
    displacements: np.ndarray
    reactions: np.ndarray

    def displacement(self, node_id: int, dof: int = TranslationDOF.X) -> float:
        """Return the displacement at a node for a given DOF direction.

        Args:
            node_id: ID of the node to query.
            dof: DOF direction to query. Defaults to ``TranslationDOF.X``,
                which is the only active direction for a 1D bar analysis
                and preserves the Version 3 single-argument call form.

        Returns:
            Displacement in meters.

        Raises:
            EntityNotFoundError: If ``node_id`` was not part of the analysis.
            ValidationError: If ``dof`` is not active for this analysis.
        """
        index = self.dof_map.global_index(node_id, dof)
        return float(self.displacements[index])

    def reaction(self, node_id: int, dof: int = TranslationDOF.X) -> float:
        """Return the reaction force at a node for a given DOF direction.

        The reaction is computed as ``R = [K]{u} - {F}``. It is only
        physically meaningful at nodes with a boundary condition on that
        DOF; at unconstrained DOFs it is approximately zero.

        Args:
            node_id: ID of the node to query.
            dof: DOF direction to query. Defaults to ``TranslationDOF.X``,
                which is the only active direction for a 1D bar analysis
                and preserves the Version 3 single-argument call form.

        Returns:
            Reaction force in newtons. The sign is such that an applied
            load and its resulting reaction sum to approximately zero
            (global equilibrium).

        Raises:
            EntityNotFoundError: If ``node_id`` was not part of the analysis.
            ValidationError: If ``dof`` is not active for this analysis.
        """
        index = self.dof_map.global_index(node_id, dof)
        return float(self.reactions[index])

    def node_displacement(self, node_id: int) -> tuple[float, ...]:
        """Return every active displacement component of a node.

        The number of components matches this analysis's
        ``dofs_per_node``: ``(ux, uy)`` for a 2D truss analysis, or
        ``(ux, uy, rz)`` for a 2D frame analysis.

        Args:
            node_id: ID of the node to query.

        Returns:
            A tuple of length ``dof_map.dofs_per_node``, in meters (and,
            for the trailing ``rz`` component of a frame analysis,
            radians).

        Raises:
            EntityNotFoundError: If ``node_id`` was not part of the analysis.
        """
        return tuple(
            self.displacement(node_id, dof) for dof in range(self.dof_map.dofs_per_node)
        )

    def node_reaction(self, node_id: int) -> tuple[float, ...]:
        """Return every active reaction component of a node.

        The number of components matches this analysis's
        ``dofs_per_node``: ``(Rx, Ry)`` for a 2D truss analysis, or
        ``(Rx, Ry, Mz)`` for a 2D frame analysis.

        Args:
            node_id: ID of the node to query.

        Returns:
            A tuple of length ``dof_map.dofs_per_node``, in newtons (and,
            for the trailing ``Mz`` component of a frame analysis,
            newton-meters).

        Raises:
            EntityNotFoundError: If ``node_id`` was not part of the analysis.
        """
        return tuple(self.reaction(node_id, dof) for dof in range(self.dof_map.dofs_per_node))

    def element_strain(self, element_id: int) -> float:
        """Return the axial strain of an element.

        Args:
            element_id: ID of the element to query.

        Returns:
            Strain (dimensionless). Positive is tension.

        Raises:
            EntityNotFoundError: If ``element_id`` was not part of the analysis.
        """
        element = self._get_element(element_id)
        return element.strain_from_dofs(self._element_dof_values(element))

    def element_stress(self, element_id: int) -> float:
        """Return the axial stress of an element, ``sigma = E * epsilon``.

        Args:
            element_id: ID of the element to query.

        Returns:
            Stress in pascals. Positive is tension, negative is compression.

        Raises:
            EntityNotFoundError: If ``element_id`` was not part of the analysis.
        """
        element = self._get_element(element_id)
        return element.stress_from_dofs(self._element_dof_values(element))

    def element_axial_force(self, element_id: int) -> float:
        """Return the internal axial force of an element, ``N = sigma * A``.

        Args:
            element_id: ID of the element to query.

        Returns:
            Axial force in newtons. Positive is tension, negative is
            compression.

        Raises:
            EntityNotFoundError: If ``element_id`` was not part of the analysis.
        """
        element = self._get_element(element_id)
        return element.axial_force_from_dofs(self._element_dof_values(element))

    def element_end_forces(self, element_id: int):
        """Return the axial force, shear force, and bending moment at both ends of a frame element.

        Args:
            element_id: ID of the frame element to query.

        Returns:
            A :class:`~femtoolkit.results.element_results.FrameElementForces`
            with the forces at ``node_1`` and ``node_2``.

        Raises:
            EntityNotFoundError: If ``element_id`` was not part of the analysis.
            InvalidElementError: If the element does not carry shear force
                or bending moment (e.g. a bar or truss element).
        """
        element = self._get_frame_element(element_id)
        return element.end_forces_from_dofs(self._element_dof_values(element))

    def element_shear_force(
        self, element_id: int, end: Literal["node_1", "node_2"] = "node_1"
    ) -> float:
        """Return the shear force at one end of a frame element.

        Args:
            element_id: ID of the frame element to query.
            end: Which end to report, ``"node_1"`` (default) or ``"node_2"``.

        Returns:
            Shear force in newtons. See the sign convention documented in
            :mod:`femtoolkit.mesh.frame_element`.

        Raises:
            EntityNotFoundError: If ``element_id`` was not part of the analysis.
            InvalidElementError: If the element does not carry shear force.
        """
        end_forces = self.element_end_forces(element_id)
        return getattr(end_forces, end).shear_force

    def element_bending_moment(
        self, element_id: int, end: Literal["node_1", "node_2"] = "node_1"
    ) -> float:
        """Return the bending moment at one end of a frame element.

        Args:
            element_id: ID of the frame element to query.
            end: Which end to report, ``"node_1"`` (default) or ``"node_2"``.

        Returns:
            Bending moment in newton-meters. See the sign convention
            documented in :mod:`femtoolkit.mesh.frame_element`.

        Raises:
            EntityNotFoundError: If ``element_id`` was not part of the analysis.
            InvalidElementError: If the element does not carry bending moment.
        """
        end_forces = self.element_end_forces(element_id)
        return getattr(end_forces, end).bending_moment

    def element_bending_stress(
        self, element_id: int, end: Literal["node_1", "node_2"] = "node_1"
    ) -> float:
        """Return the extreme-fiber bending stress at one end of a frame element.

        Computed as ``sigma = M * c / I``, where ``M`` is the bending
        moment at the requested end, ``c`` is
        ``cross_section.extreme_fiber_distance``, and ``I`` is
        ``cross_section.second_moment_of_area``.

        Args:
            element_id: ID of the frame element to query.
            end: Which end to report, ``"node_1"`` (default) or ``"node_2"``.

        Returns:
            Bending stress in pascals.

        Raises:
            EntityNotFoundError: If ``element_id`` was not part of the analysis.
            InvalidElementError: If the element does not carry bending moment.
            ValidationError: If the element's cross-section does not
                specify ``extreme_fiber_distance``.
        """
        element = self._get_frame_element(element_id)
        extreme_fiber_distance = getattr(element.cross_section, "extreme_fiber_distance", None)
        if extreme_fiber_distance is None:
            raise ValidationError(
                f"Element {element_id}'s cross_section has no extreme_fiber_distance; "
                "bending stress cannot be computed."
            )

        bending_moment = self.element_bending_moment(element_id, end=end)
        return bending_moment * extreme_fiber_distance / element.cross_section.second_moment_of_area

    def _get_element(self, element_id: int) -> StructuralElement:
        for element in self.elements:
            if element.id == element_id:
                return element
        raise EntityNotFoundError(f"No element with id {element_id} found in this result.")

    def _get_frame_element(self, element_id: int) -> FrameStructuralElement:
        element = self._get_element(element_id)
        if not isinstance(element, FrameStructuralElement):
            raise InvalidElementError(
                f"Element {element_id} ({type(element).__name__}) does not carry shear "
                "force or bending moment; only frame elements do."
            )
        return element

    def _element_dof_values(self, element: StructuralElement) -> list[float]:
        """Displacement values for an element's DOFs, ordered per ``dof_keys()``."""
        return [self.displacement(node_id, dof) for node_id, dof in element.dof_keys()]
