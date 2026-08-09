"""1D axial bar element.

This module defines :class:`BarElement`, a dedicated two-node element for
linear-elastic, small-deformation, static axial behavior. Unlike the
generic :class:`~femtoolkit.mesh.element.Element` introduced in Version 1
(which stores topology only), a bar element performs the local
engineering calculations that are specific to axial bars: length,
stiffness, strain, stress, and axial force.

Sign convention: positive strain, stress, and axial force represent
**tension** (the bar is being stretched); negative values represent
**compression**.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from femtoolkit.exceptions import ValidationError
from femtoolkit.materials import Material
from femtoolkit.mesh.node import Node
from femtoolkit.sections import CrossSection


@dataclass
class BarElement:
    """A two-node 1D axial bar element.

    A bar element connects two nodes along the structural (X) axis and
    carries only axial load. It behaves as a linear elastic spring with
    stiffness ``k = E * A / L``, where ``E`` is the material's Young's
    modulus, ``A`` is the cross-sectional area, and ``L`` is the element
    length.

    Version 3 is strictly 1D: only the node X coordinates determine the
    element length. Nodes should be laid out along the X axis; Y and Z
    coordinates are not used in any Version 3 calculation.

    Attributes:
        id: Positive integer identifying the element uniquely within a mesh.
        nodes: The two nodes the element connects, ``(node_1, node_2)``.
        material: Material assigned to the element.
        cross_section: Cross-sectional area assigned to the element.

    Raises:
        ValidationError: If ``id`` is invalid, ``nodes`` does not contain
            exactly two distinct :class:`Node` instances, ``material`` is
            not a :class:`Material`, ``cross_section`` is not a
            :class:`CrossSection`, or the two nodes share the same X
            coordinate (a zero-length element).

    Example:
        >>> bar = BarElement(
        ...     id=1,
        ...     nodes=(node_1, node_2),
        ...     material=steel,
        ...     cross_section=section,
        ... )
        >>> bar.length
        2.0
    """

    id: int
    nodes: tuple[Node, Node]
    material: Material
    cross_section: CrossSection

    def __post_init__(self) -> None:
        """Validate the bar element immediately after construction.

        Raises:
            ValidationError: If ``id`` is not a positive integer,
                ``nodes`` is not a pair of distinct :class:`Node`
                instances, ``material`` is not a :class:`Material`,
                ``cross_section`` is not a :class:`CrossSection`, or the
                resulting element length is not positive.
        """
        if not isinstance(self.id, int) or isinstance(self.id, bool) or self.id <= 0:
            raise ValidationError(f"BarElement id must be a positive integer, got {self.id!r}.")

        if len(self.nodes) != 2 or not all(isinstance(node, Node) for node in self.nodes):
            raise ValidationError(
                f"BarElement nodes must be a pair of Node instances, got {self.nodes!r}."
            )
        if self.nodes[0].id == self.nodes[1].id:
            raise ValidationError("BarElement requires two distinct nodes.")

        if not isinstance(self.material, Material):
            raise ValidationError(
                f"BarElement material must be a Material instance, got {self.material!r}."
            )
        if not isinstance(self.cross_section, CrossSection):
            raise ValidationError(
                "BarElement cross_section must be a CrossSection instance, "
                f"got {self.cross_section!r}."
            )

        if not math.isfinite(self.length) or self.length <= 0:
            raise ValidationError(
                "BarElement requires two nodes with different X coordinates "
                f"(got length={self.length})."
            )

    @property
    def length(self) -> float:
        """Element length, ``L = |x2 - x1|``, using only the node X coordinates."""
        return abs(self.nodes[1].x - self.nodes[0].x)

    @property
    def stiffness_matrix(self) -> np.ndarray:
        """Local 2x2 stiffness matrix ``[[k, -k], [-k, k]]`` with ``k = EA/L``."""
        # Imported locally: the `analysis` package depends on `mesh` (a
        # StaticLinearAnalysis operates on BarElement instances), so a
        # module-level import here would create a circular import between
        # the two packages depending on which one loads first.
        from femtoolkit.analysis.stiffness import bar_element_stiffness

        return bar_element_stiffness(
            youngs_modulus=self.material.youngs_modulus,
            area=self.cross_section.area,
            length=self.length,
        )

    def strain(self, u1: float, u2: float) -> float:
        """Compute axial strain from nodal displacements.

        The strain is calculated as::

            epsilon = (u2 - u1) / L

        Args:
            u1: Axial displacement at node 1, in meters.
            u2: Axial displacement at node 2, in meters.

        Returns:
            Axial strain (dimensionless). Positive is tension (elongation).
        """
        return (u2 - u1) / self.length

    def stress(self, u1: float, u2: float) -> float:
        """Compute axial stress from nodal displacements.

        The stress is calculated from Hooke's law for a linear elastic
        material::

            sigma = E * epsilon

        Args:
            u1: Axial displacement at node 1, in meters.
            u2: Axial displacement at node 2, in meters.

        Returns:
            Axial stress, in pascals. Positive is tension, negative is
            compression.
        """
        return self.material.youngs_modulus * self.strain(u1, u2)

    def axial_force(self, u1: float, u2: float) -> float:
        """Compute internal axial force from nodal displacements.

        The axial force is calculated as::

            N = sigma * A

        Args:
            u1: Axial displacement at node 1, in meters.
            u2: Axial displacement at node 2, in meters.

        Returns:
            Axial force, in newtons. Positive is tension, negative is
            compression.
        """
        return self.stress(u1, u2) * self.cross_section.area
