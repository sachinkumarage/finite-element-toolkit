"""Structural analysis result representation.

This module defines :class:`AnalysisResult`, a read-only view over the
outcome of a :class:`~femtoolkit.analysis.static_linear.StaticLinearAnalysis`
solve. It exposes nodal displacements, reaction forces, and per-element
strain, stress, and axial force, computed on demand from the raw solved
displacement vector. The result object performs no matrix assembly or
solving of its own.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from femtoolkit.analysis.dof import DOFMap, TranslationDOF
from femtoolkit.exceptions import EntityNotFoundError
from femtoolkit.mesh.bar_element import BarElement


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
        elements: The bar elements that were analyzed.
        displacements: Global displacement vector ``{u}``, in global DOF
            order.
        reactions: Global reaction force vector ``R = [K]{u} - {F}``, in
            global DOF order. Nonzero only at constrained DOFs, up to
            floating-point precision.
    """

    dof_map: DOFMap
    elements: tuple[BarElement, ...]
    displacements: np.ndarray
    reactions: np.ndarray

    def displacement(self, node_id: int) -> float:
        """Return the axial displacement at a node.

        Args:
            node_id: ID of the node to query.

        Returns:
            Displacement in meters.

        Raises:
            EntityNotFoundError: If ``node_id`` was not part of the analysis.
        """
        index = self.dof_map.global_index(node_id, TranslationDOF.X)
        return float(self.displacements[index])

    def reaction(self, node_id: int) -> float:
        """Return the reaction force at a node.

        The reaction is computed as ``R = [K]{u} - {F}``. It is only
        physically meaningful at nodes with a boundary condition; at
        unconstrained nodes it is approximately zero.

        Args:
            node_id: ID of the node to query.

        Returns:
            Reaction force in newtons. The sign is such that an applied
            load and its resulting reaction sum to approximately zero
            (global equilibrium).

        Raises:
            EntityNotFoundError: If ``node_id`` was not part of the analysis.
        """
        index = self.dof_map.global_index(node_id, TranslationDOF.X)
        return float(self.reactions[index])

    def element_strain(self, element_id: int) -> float:
        """Return the axial strain of an element, ``epsilon = (u2 - u1) / L``.

        Args:
            element_id: ID of the element to query.

        Returns:
            Strain (dimensionless). Positive is tension.

        Raises:
            EntityNotFoundError: If ``element_id`` was not part of the analysis.
        """
        element = self._get_element(element_id)
        u1, u2 = self._element_displacements(element)
        return element.strain(u1, u2)

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
        u1, u2 = self._element_displacements(element)
        return element.stress(u1, u2)

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
        u1, u2 = self._element_displacements(element)
        return element.axial_force(u1, u2)

    def _get_element(self, element_id: int) -> BarElement:
        for element in self.elements:
            if element.id == element_id:
                return element
        raise EntityNotFoundError(f"No element with id {element_id} found in this result.")

    def _element_displacements(self, element: BarElement) -> tuple[float, float]:
        return self.displacement(element.nodes[0].id), self.displacement(element.nodes[1].id)
