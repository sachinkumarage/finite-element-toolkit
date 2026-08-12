"""Structural and continuum element interfaces used by the analysis layer.

:class:`StaticLinearAnalysis` needs only four things from an element to
assemble and solve a system: its ID, how many DOFs it activates per node,
which global DOFs its rows/columns correspond to, and its local stiffness
matrix. :class:`AssemblableElement` captures exactly that, as a
structural (duck-typed) protocol rather than a shared base class -- so
every element type in the toolkit can be assembled and solved through the
same code path without the analysis layer importing concrete element
classes.

Two richer protocols extend it for post-processing, matching the two
different kinds of physical quantity an element can report:

* :class:`StructuralElement` -- one-dimensional members
  (:class:`~femtoolkit.mesh.bar_element.BarElement`,
  :class:`~femtoolkit.mesh.truss_element.TrussElement2D`,
  :class:`~femtoolkit.mesh.frame_element.FrameElement2D`) report a single
  **scalar** axial strain, stress, and force -- there is only one
  direction along the member's own axis for these to act in.
* :class:`ContinuumElement` -- 2D continuum elements
  (:class:`~femtoolkit.mesh.cst_element.CSTElement2D`) report a
  three-component **strain/stress vector** (``[x, y, xy]``) plus von
  Mises and principal stress, since a continuum point's strain and stress
  state is inherently multi-directional, not a single scalar.

These two are siblings, not one a subtype of the other: a continuum
element has no meaningful "axial force," and a structural member has no
meaningful "principal stress." :class:`FrameStructuralElement` further
extends :class:`StructuralElement` with the one capability a frame
element has that a bar or truss element does not: per-end shear force and
bending moment.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from femtoolkit.results.element_results import FrameElementForces


@runtime_checkable
class AssemblableElement(Protocol):
    """The minimal interface a finite element must expose to be assembled and solved.

    Attributes:
        id: Positive integer element ID.
        dofs_per_node: Number of DOFs this element activates at each of
            its nodes (1 for an axial bar, 2 for a 2D truss or CST
            element, 3 for a 2D frame element). All elements in a single
            analysis must share the same value.
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


@runtime_checkable
class StructuralElement(AssemblableElement, Protocol):
    """An :class:`AssemblableElement` that reports scalar axial strain, stress, and force.

    Satisfied by one-dimensional members: bar, truss, and frame elements.
    Not satisfied by :class:`~femtoolkit.mesh.cst_element.CSTElement2D`,
    whose strain and stress are vectors, not scalars -- see
    :class:`ContinuumElement`.
    """

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


@runtime_checkable
class ContinuumElement(AssemblableElement, Protocol):
    """An :class:`AssemblableElement` that reports vector strain and stress.

    Satisfied by :class:`~femtoolkit.mesh.cst_element.CSTElement2D`; not
    satisfied by any one-dimensional structural member (see
    :class:`StructuralElement`).
    """

    def strain_from_dofs(self, displacements: Sequence[float]) -> np.ndarray:
        """Compute strain ``[epsilon_x, epsilon_y, gamma_xy]`` from nodal displacements."""
        ...

    def stress_from_dofs(self, displacements: Sequence[float]) -> np.ndarray:
        """Compute stress ``[sigma_x, sigma_y, tau_xy]`` from nodal displacements."""
        ...

    def von_mises_from_dofs(self, displacements: Sequence[float]) -> float:
        """Compute the von Mises equivalent stress from nodal displacements."""
        ...

    def principal_stresses_from_dofs(self, displacements: Sequence[float]) -> tuple[float, float]:
        """Compute the in-plane principal stresses from nodal displacements."""
        ...
