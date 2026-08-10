"""2D truss element.

This module defines :class:`TrussElement2D`, a two-node element for
linear-elastic, small-deformation, static axial behavior in a 2D plane.
Like :class:`~femtoolkit.mesh.bar_element.BarElement`, it carries only
axial force -- it has no bending, shear, or torsional stiffness, matching
a pin-jointed truss member. Unlike a bar element, each node has two
translational DOFs (X and Y), and the element's local axial stiffness is
transformed into global coordinates using its direction cosines.

Sign convention: positive strain, stress, and axial force represent
**tension** (the member is being stretched); negative values represent
**compression**.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from femtoolkit.exceptions import ValidationError
from femtoolkit.materials import Material
from femtoolkit.mesh._element_validation import validate_two_node_element
from femtoolkit.mesh.node import Node
from femtoolkit.sections import CrossSection


@dataclass
class TrussElement2D:
    """A two-node 2D truss (pin-jointed, axial-only) element.

    A truss element connects two nodes in the X/Y plane and carries only
    axial load along its own axis. Its local axial stiffness
    ``k = E * A / L`` is transformed into global X/Y coordinates using the
    element's direction cosines ``c = cos(theta)``, ``s = sin(theta)``,
    where ``theta`` is the angle from the global X axis to the vector
    from node 1 to node 2.

    Only the node X and Y coordinates are used; Z is not part of any
    Version 4 calculation.

    Attributes:
        id: Positive integer identifying the element uniquely within a mesh.
        nodes: The two nodes the element connects, ``(node_1, node_2)``.
        material: Material assigned to the element.
        cross_section: Cross-sectional area assigned to the element.

    Raises:
        ValidationError: If ``id`` is invalid, ``nodes`` does not contain
            exactly two distinct :class:`Node` instances, ``material`` is
            not a :class:`Material`, ``cross_section`` is not a
            :class:`CrossSection`, or the two nodes share the same X and Y
            coordinates (a zero-length element).

    Example:
        >>> element = TrussElement2D(
        ...     id=1,
        ...     nodes=(node_1, node_2),
        ...     material=steel,
        ...     cross_section=section,
        ... )
        >>> element.length
        2.0
    """

    id: int
    nodes: tuple[Node, Node]
    material: Material
    cross_section: CrossSection

    dofs_per_node: ClassVar[int] = 2
    """Number of DOFs this element activates per node (see
    :class:`~femtoolkit.analysis.element.StructuralElement`).
    """

    def __post_init__(self) -> None:
        """Validate the truss element immediately after construction.

        Raises:
            ValidationError: If ``id`` is not a positive integer,
                ``nodes`` is not a pair of distinct :class:`Node`
                instances, ``material`` is not a :class:`Material`,
                ``cross_section`` is not a :class:`CrossSection`, or the
                resulting element length is not positive.
        """
        validate_two_node_element(
            "TrussElement2D", self.id, self.nodes, self.material, self.cross_section
        )

        if not math.isfinite(self.length) or self.length <= 0:
            raise ValidationError(
                "TrussElement2D requires two nodes at different X/Y coordinates "
                f"(got length={self.length})."
            )

    @property
    def length(self) -> float:
        """Element length, ``L = sqrt((x2-x1)^2 + (y2-y1)^2)``."""
        node_1, node_2 = self.nodes
        return math.hypot(node_2.x - node_1.x, node_2.y - node_1.y)

    @property
    def direction_cosines(self) -> tuple[float, float]:
        """The element's ``(c, s) = (cos(theta), sin(theta))`` orientation.

        ``c = (x2 - x1) / L`` and ``s = (y2 - y1) / L``, where ``theta``
        is the angle from the global X axis to the vector from node 1 to
        node 2.
        """
        node_1, node_2 = self.nodes
        length = self.length
        return (node_2.x - node_1.x) / length, (node_2.y - node_1.y) / length

    def dof_keys(self) -> tuple[tuple[int, int], ...]:
        """The ``(node_id, dof)`` pairs matching ``stiffness_matrix``'s rows/columns.

        A 2D truss element has two DOFs per node, so this returns
        ``((node_1.id, X), (node_1.id, Y), (node_2.id, X), (node_2.id, Y))``.
        """
        # Imported locally: see the note on `stiffness_matrix` below.
        from femtoolkit.analysis.dof import TranslationDOF

        node_1, node_2 = self.nodes
        return (
            (node_1.id, TranslationDOF.X),
            (node_1.id, TranslationDOF.Y),
            (node_2.id, TranslationDOF.X),
            (node_2.id, TranslationDOF.Y),
        )

    @property
    def stiffness_matrix(self) -> np.ndarray:
        """Local 4x4 stiffness matrix in global X/Y coordinates.

        See :func:`~femtoolkit.analysis.stiffness.truss_element_stiffness_2d`
        for the underlying formula.
        """
        # Imported locally: the `analysis` package depends on `mesh` (a
        # StaticLinearAnalysis operates on mesh elements), so a
        # module-level import here would create a circular import between
        # the two packages depending on which one loads first.
        from femtoolkit.analysis.stiffness import truss_element_stiffness_2d

        cos_theta, sin_theta = self.direction_cosines
        return truss_element_stiffness_2d(
            youngs_modulus=self.material.youngs_modulus,
            area=self.cross_section.area,
            length=self.length,
            cos_theta=cos_theta,
            sin_theta=sin_theta,
        )

    def strain(self, ux1: float, uy1: float, ux2: float, uy2: float) -> float:
        """Compute axial strain from global nodal displacements.

        The global displacements are first projected onto the element's
        local axis using its direction cosines::

            u1' = c*ux1 + s*uy1
            u2' = c*ux2 + s*uy2
            epsilon = (u2' - u1') / L

        Args:
            ux1: X displacement at node 1, in meters.
            uy1: Y displacement at node 1, in meters.
            ux2: X displacement at node 2, in meters.
            uy2: Y displacement at node 2, in meters.

        Returns:
            Axial strain (dimensionless). Positive is tension (elongation).
        """
        c, s = self.direction_cosines
        local_u1 = c * ux1 + s * uy1
        local_u2 = c * ux2 + s * uy2
        return (local_u2 - local_u1) / self.length

    def stress(self, ux1: float, uy1: float, ux2: float, uy2: float) -> float:
        """Compute axial stress from global nodal displacements.

        Uses Hooke's law for a linear elastic material: ``sigma = E * epsilon``.

        Args:
            ux1: X displacement at node 1, in meters.
            uy1: Y displacement at node 1, in meters.
            ux2: X displacement at node 2, in meters.
            uy2: Y displacement at node 2, in meters.

        Returns:
            Axial stress, in pascals. Positive is tension, negative is
            compression.
        """
        return self.material.youngs_modulus * self.strain(ux1, uy1, ux2, uy2)

    def axial_force(self, ux1: float, uy1: float, ux2: float, uy2: float) -> float:
        """Compute internal axial force from global nodal displacements.

        The axial force is calculated as ``N = sigma * A``.

        Args:
            ux1: X displacement at node 1, in meters.
            uy1: Y displacement at node 1, in meters.
            ux2: X displacement at node 2, in meters.
            uy2: Y displacement at node 2, in meters.

        Returns:
            Axial force, in newtons. Positive is tension, negative is
            compression.
        """
        return self.stress(ux1, uy1, ux2, uy2) * self.cross_section.area

    def strain_from_dofs(self, displacements: Sequence[float]) -> float:
        """Compute axial strain from ``[ux1, uy1, ux2, uy2]``, per :meth:`dof_keys`."""
        return self.strain(*displacements)

    def stress_from_dofs(self, displacements: Sequence[float]) -> float:
        """Compute axial stress from ``[ux1, uy1, ux2, uy2]``, per :meth:`dof_keys`."""
        return self.stress(*displacements)

    def axial_force_from_dofs(self, displacements: Sequence[float]) -> float:
        """Compute axial force from ``[ux1, uy1, ux2, uy2]``, per :meth:`dof_keys`."""
        return self.axial_force(*displacements)
