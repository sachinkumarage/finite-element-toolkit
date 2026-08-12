"""2D constant strain triangle (CST) continuum element.

This module defines :class:`CSTElement2D`, a three-node element
representing a finite *area* of material -- unlike every prior element in
the toolkit (:class:`~femtoolkit.mesh.bar_element.BarElement`,
:class:`~femtoolkit.mesh.truss_element.TrussElement2D`,
:class:`~femtoolkit.mesh.frame_element.FrameElement2D`), which represent
one-dimensional structural *members*. Each node activates two
translational DOFs, ``ux`` and ``uy`` -- no rotational DOF, since a
continuum point has no orientation to rotate.

The element coordinates reusable continuum mathematics
(:mod:`femtoolkit.continuum`) rather than embedding it: geometry, shape
functions, the strain-displacement matrix, the constitutive matrix, and
stress recovery are all independently testable pure functions. This
class's job is to hold the element's three nodes, material, and
thickness, and translate between them and that reusable math.

**Orientation policy.** See :mod:`femtoolkit.continuum.geometry`: nodes
may be listed clockwise or counter-clockwise -- both are accepted and
give identical strain, stress, and stiffness results, because the
element consistently uses the *signed* area inside the strain-
displacement matrix and the *absolute* area everywhere a physical area is
needed.

Sign convention: engineering shear strain (``gamma_xy = du/dy + dv/dx``),
matching :mod:`femtoolkit.continuum.strain`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from femtoolkit.continuum.geometry import MIN_TRIANGLE_AREA, triangle_signed_area
from femtoolkit.exceptions import DegenerateElementError, ValidationError
from femtoolkit.materials import LinearElastic2D
from femtoolkit.mesh.node import Node


@dataclass
class CSTElement2D:
    """A three-node 2D constant strain triangle (CST) continuum element.

    Attributes:
        id: Positive integer identifying the element uniquely within a mesh.
        nodes: The three nodes the element connects, ``(node_1, node_2, node_3)``.
        material: The element's 2D constitutive model (plane stress or
            plane strain).
        thickness: Element thickness, in meters. Must be positive.

    Raises:
        ValidationError: If ``id`` is invalid, ``nodes`` does not contain
            exactly three distinct :class:`Node` instances, ``material``
            is not a :class:`LinearElastic2D`, or ``thickness`` is not a
            positive, finite number.
        DegenerateElementError: If the three nodes are collinear, nearly
            collinear, or coincide (zero or near-zero area).

    Example:
        >>> element = CSTElement2D(
        ...     id=1,
        ...     nodes=(node_1, node_2, node_3),
        ...     material=LinearElastic2D(
        ...         youngs_modulus=210e9, poisson_ratio=0.3, formulation="plane_stress"
        ...     ),
        ...     thickness=0.01,
        ... )
        >>> element.area
        0.5
    """

    id: int
    nodes: tuple[Node, Node, Node]
    material: LinearElastic2D
    thickness: float

    dofs_per_node: ClassVar[int] = 2
    """Number of DOFs this element activates per node: ux, uy (see
    :class:`~femtoolkit.analysis.element.AssemblableElement`). No
    rotational DOF -- a continuum point has no orientation.
    """

    def __post_init__(self) -> None:
        """Validate the CST element immediately after construction.

        Raises:
            ValidationError: If ``id`` is not a positive integer,
                ``nodes`` is not a triple of distinct :class:`Node`
                instances, ``material`` is not a :class:`LinearElastic2D`,
                or ``thickness`` is not a positive, finite number.
            DegenerateElementError: If the resulting triangle area is
                zero or near-zero.
        """
        if not isinstance(self.id, int) or isinstance(self.id, bool) or self.id <= 0:
            raise ValidationError(f"CSTElement2D id must be a positive integer, got {self.id!r}.")

        if len(self.nodes) != 3 or not all(isinstance(node, Node) for node in self.nodes):
            raise ValidationError(
                f"CSTElement2D nodes must be a triple of Node instances, got {self.nodes!r}."
            )
        node_ids = {node.id for node in self.nodes}
        if len(node_ids) != 3:
            raise ValidationError("CSTElement2D requires three distinct nodes.")

        if not isinstance(self.material, LinearElastic2D):
            raise ValidationError(
                f"CSTElement2D material must be a LinearElastic2D instance, "
                f"got {self.material!r}."
            )

        if not math.isfinite(self.thickness) or self.thickness <= 0:
            raise ValidationError(
                f"CSTElement2D thickness must be positive, got {self.thickness}."
            )

        if abs(self.signed_area) < MIN_TRIANGLE_AREA:
            raise DegenerateElementError(
                f"CSTElement2D {self.id} has area {self.signed_area} "
                "(collinear, nearly collinear, or duplicate-coordinate nodes)."
            )

    @property
    def signed_area(self) -> float:
        """The triangle's signed area (positive for CCW node order, negative for CW).

        See :func:`~femtoolkit.continuum.geometry.triangle_signed_area`
        and the module docstring's orientation policy.
        """
        node_1, node_2, node_3 = self.nodes
        return triangle_signed_area(node_1.x, node_1.y, node_2.x, node_2.y, node_3.x, node_3.y)

    @property
    def area(self) -> float:
        """The triangle's physical (always positive) area, in square meters."""
        return abs(self.signed_area)

    def dof_keys(self) -> tuple[tuple[int, int], ...]:
        """The ``(node_id, dof)`` pairs matching ``stiffness_matrix``'s rows/columns.

        A CST element has two DOFs per node, so this returns
        ``((node_1.id, X), (node_1.id, Y), (node_2.id, X), (node_2.id, Y),
        (node_3.id, X), (node_3.id, Y))``.
        """
        # Imported locally: see the note on `stiffness_matrix` below.
        from femtoolkit.analysis.dof import TranslationDOF

        node_1, node_2, node_3 = self.nodes
        return (
            (node_1.id, TranslationDOF.X),
            (node_1.id, TranslationDOF.Y),
            (node_2.id, TranslationDOF.X),
            (node_2.id, TranslationDOF.Y),
            (node_3.id, TranslationDOF.X),
            (node_3.id, TranslationDOF.Y),
        )

    @property
    def b_matrix(self) -> np.ndarray:
        """The element's constant 3x6 strain-displacement matrix.

        See :func:`~femtoolkit.continuum.strain.triangle_strain_displacement_matrix`.
        """
        # Imported locally: `continuum` depends on nothing in `mesh`, but
        # keeping the import lazy here matches the established pattern
        # for every other element's `stiffness_matrix`/math properties in
        # this package, avoiding any import-order sensitivity.
        from femtoolkit.continuum.strain import triangle_strain_displacement_matrix

        node_1, node_2, node_3 = self.nodes
        return triangle_strain_displacement_matrix(
            node_1.x, node_1.y, node_2.x, node_2.y, node_3.x, node_3.y
        )

    @property
    def stiffness_matrix(self) -> np.ndarray:
        """Global 6x6 stiffness matrix, ``Ke = t * A * B^T * D * B``.

        See :func:`~femtoolkit.analysis.stiffness.cst_element_stiffness`.
        No coordinate transformation is needed: ``ux``/``uy`` are already
        expressed in global coordinates.
        """
        # Imported locally: the `analysis` package depends on `mesh` (a
        # StaticLinearAnalysis operates on mesh elements), so a
        # module-level import here would create a circular import between
        # the two packages depending on which one loads first.
        from femtoolkit.analysis.stiffness import cst_element_stiffness

        return cst_element_stiffness(
            thickness=self.thickness,
            area=self.area,
            b_matrix=self.b_matrix,
            d_matrix=self.material.constitutive_matrix,
        )

    def strain_from_dofs(self, displacements: Sequence[float]) -> np.ndarray:
        """Compute the constant strain field from ``[u1,v1,u2,v2,u3,v3]``.

        Args:
            displacements: Nodal displacements, ordered per :meth:`dof_keys`.

        Returns:
            A length-3 NumPy array, ``[epsilon_x, epsilon_y, gamma_xy]``.
        """
        from femtoolkit.continuum.strain import strain_from_displacements

        return strain_from_displacements(self.b_matrix, displacements)

    def stress_from_dofs(self, displacements: Sequence[float]) -> np.ndarray:
        """Compute the constant stress field from ``[u1,v1,u2,v2,u3,v3]``.

        Args:
            displacements: Nodal displacements, ordered per :meth:`dof_keys`.

        Returns:
            A length-3 NumPy array, ``[sigma_x, sigma_y, tau_xy]``.
        """
        from femtoolkit.continuum.stress import stress_from_strain

        return stress_from_strain(
            self.material.constitutive_matrix, self.strain_from_dofs(displacements)
        )

    def von_mises_from_dofs(self, displacements: Sequence[float]) -> float:
        """Compute the von Mises equivalent stress from ``[u1,v1,u2,v2,u3,v3]``.

        Dispatches to :func:`~femtoolkit.continuum.stress.von_mises_plane_stress`
        or :func:`~femtoolkit.continuum.stress.von_mises_plane_strain`
        depending on ``self.material.formulation``.

        Args:
            displacements: Nodal displacements, ordered per :meth:`dof_keys`.

        Returns:
            The von Mises equivalent stress, in pascals.
        """
        from femtoolkit.continuum.stress import von_mises_plane_strain, von_mises_plane_stress

        sigma_x, sigma_y, tau_xy = self.stress_from_dofs(displacements)
        if self.material.formulation == "plane_stress":
            return von_mises_plane_stress(sigma_x, sigma_y, tau_xy)
        return von_mises_plane_strain(sigma_x, sigma_y, tau_xy, self.material.poisson_ratio)

    def principal_stresses_from_dofs(self, displacements: Sequence[float]) -> tuple[float, float]:
        """Compute the in-plane principal stresses from ``[u1,v1,u2,v2,u3,v3]``.

        Args:
            displacements: Nodal displacements, ordered per :meth:`dof_keys`.

        Returns:
            ``(sigma_1, sigma_2)``, in pascals (``sigma_1 >= sigma_2``).
        """
        from femtoolkit.continuum.stress import principal_stresses_2d

        sigma_x, sigma_y, tau_xy = self.stress_from_dofs(displacements)
        return principal_stresses_2d(sigma_x, sigma_y, tau_xy)
