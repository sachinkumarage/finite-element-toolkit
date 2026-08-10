"""Structural analysis result representation.

This module defines :class:`AnalysisResult`, a read-only view over the
outcome of a :class:`~femtoolkit.analysis.static_linear.StaticLinearAnalysis`
solve. It exposes nodal displacements, reaction forces, and per-element
strain, stress, and axial force, computed on demand from the raw solved
displacement vector. The result object performs no matrix assembly or
solving of its own, and works with any element satisfying the
:class:`~femtoolkit.analysis.element.StructuralElement` protocol.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from femtoolkit.analysis.dof import DOFMap, TranslationDOF
from femtoolkit.analysis.element import StructuralElement
from femtoolkit.exceptions import EntityNotFoundError


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

    def node_displacement(self, node_id: int) -> tuple[float, float]:
        """Return the ``(ux, uy)`` displacement of a node in a 2D analysis.

        Args:
            node_id: ID of the node to query.

        Returns:
            A ``(ux, uy)`` tuple, in meters.

        Raises:
            EntityNotFoundError: If ``node_id`` was not part of the analysis.
            ValidationError: If this analysis does not activate both X
                and Y DOFs (i.e. it is not a 2D analysis).
        """
        return (
            self.displacement(node_id, TranslationDOF.X),
            self.displacement(node_id, TranslationDOF.Y),
        )

    def node_reaction(self, node_id: int) -> tuple[float, float]:
        """Return the ``(Rx, Ry)`` reaction of a node in a 2D analysis.

        Args:
            node_id: ID of the node to query.

        Returns:
            A ``(Rx, Ry)`` tuple, in newtons.

        Raises:
            EntityNotFoundError: If ``node_id`` was not part of the analysis.
            ValidationError: If this analysis does not activate both X
                and Y DOFs (i.e. it is not a 2D analysis).
        """
        return (
            self.reaction(node_id, TranslationDOF.X),
            self.reaction(node_id, TranslationDOF.Y),
        )

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

    def _get_element(self, element_id: int) -> StructuralElement:
        for element in self.elements:
            if element.id == element_id:
                return element
        raise EntityNotFoundError(f"No element with id {element_id} found in this result.")

    def _element_dof_values(self, element: StructuralElement) -> list[float]:
        """Displacement values for an element's DOFs, ordered per ``dof_keys()``."""
        return [self.displacement(node_id, dof) for node_id, dof in element.dof_keys()]
