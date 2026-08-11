"""Structural element interface used by the analysis layer.

:class:`StaticLinearAnalysis` and :class:`~femtoolkit.results.analysis_result.AnalysisResult`
need only a handful of things from an element: its ID, how many DOFs it
activates per node, its local stiffness matrix, and how to turn a
displacement vector into strain/stress/axial force. :class:`StructuralElement`
captures exactly that, as a structural (duck-typed) protocol rather than a
shared base class -- so :class:`~femtoolkit.mesh.bar_element.BarElement`,
:class:`~femtoolkit.mesh.truss_element.TrussElement2D`,
:class:`~femtoolkit.mesh.frame_element.FrameElement2D`, and any future
element type can all be analyzed through the same code path without the
analysis layer importing concrete element classes.

:class:`FrameStructuralElement` extends this with the one capability a
frame element has that a bar or truss element does not: per-end shear
force and bending moment, alongside the axial-only results every
structural element already reports.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from femtoolkit.results.element_results import FrameElementForces


@runtime_checkable
class StructuralElement(Protocol):
    """The minimal interface a finite element must expose to be analyzed.

    Attributes:
        id: Positive integer element ID.
        dofs_per_node: Number of DOFs this element activates at each of
            its nodes (1 for an axial bar, 2 for a 2D truss element, 3
            for a 2D frame element). All elements in a single analysis
            must share the same value.
    """

    id: int
    dofs_per_node: int

    def dof_keys(self) -> tuple[tuple[int, int], ...]:
        """Return one ``(node_id, dof)`` pair per row/column of ``stiffness_matrix``."""
        ...

    @property
    def stiffness_matrix(self) -> np.ndarray:
        """The element's local stiffness matrix, square with size ``len(dof_keys())``."""
        ...

    def strain_from_dofs(self, displacements: Sequence[float]) -> float:
        """Compute axial strain from nodal displacements ordered per ``dof_keys()``."""
        ...

    def stress_from_dofs(self, displacements: Sequence[float]) -> float:
        """Compute axial stress from nodal displacements ordered per ``dof_keys()``."""
        ...

    def axial_force_from_dofs(self, displacements: Sequence[float]) -> float:
        """Compute axial force from nodal displacements ordered per ``dof_keys()``."""
        ...


@runtime_checkable
class FrameStructuralElement(StructuralElement, Protocol):
    """A :class:`StructuralElement` that also reports shear force and bending moment.

    Satisfied by :class:`~femtoolkit.mesh.frame_element.FrameElement2D`;
    not satisfied by :class:`~femtoolkit.mesh.bar_element.BarElement` or
    :class:`~femtoolkit.mesh.truss_element.TrussElement2D`, which have no
    bending stiffness and therefore no shear or moment to report.
    """

    def end_forces_from_dofs(self, displacements: Sequence[float]) -> FrameElementForces:
        """Compute per-end axial force, shear force, and bending moment.

        Args:
            displacements: Nodal displacements ordered per ``dof_keys()``.

        Returns:
            A :class:`~femtoolkit.results.element_results.FrameElementForces`
            with the forces at both ends of the element.
        """
        ...
