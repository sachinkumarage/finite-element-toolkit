"""Structural element interface used by the analysis layer.

:class:`StaticLinearAnalysis` and :class:`~femtoolkit.results.analysis_result.AnalysisResult`
need only a handful of things from an element: its ID, how many DOFs it
activates per node, its local stiffness matrix, and how to turn a
displacement vector into strain/stress/axial force. :class:`StructuralElement`
captures exactly that, as a structural (duck-typed) protocol rather than a
shared base class -- so :class:`~femtoolkit.mesh.bar_element.BarElement`,
:class:`~femtoolkit.mesh.truss_element.TrussElement2D`, and any future
element type can all be analyzed through the same code path without the
analysis layer importing concrete element classes.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class StructuralElement(Protocol):
    """The minimal interface a finite element must expose to be analyzed.

    Attributes:
        id: Positive integer element ID.
        dofs_per_node: Number of DOFs this element activates at each of
            its nodes (1 for an axial bar, 2 for a 2D truss element). All
            elements in a single analysis must share the same value.
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
