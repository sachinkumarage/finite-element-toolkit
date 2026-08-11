"""2D Euler-Bernoulli beam/frame element.

This module defines :class:`FrameElement2D`, a two-node element for
linear-elastic, small-deformation, static behavior in a 2D plane that
resists **axial force, shear force, and bending moment** -- unlike
:class:`~femtoolkit.mesh.truss_element.TrussElement2D`, which carries
only axial force. Each node activates three DOFs: ``ux``, ``uy``, and
``rz`` (rotation about the out-of-plane Z axis).

Engineering assumptions (see the Version 5 section of the project
README for the full discussion):

* Linear elastic material, small deformation, small strain, static
  loading.
* Euler-Bernoulli beam theory: plane sections remain plane and
  perpendicular to the neutral axis, so shear deformation is neglected.
* Constant, prismatic cross-section per element.

Sign convention:

* Positive axial force represents **tension**; positive axial strain and
  stress represent elongation (matches
  :class:`~femtoolkit.mesh.truss_element.TrussElement2D`).
* Local end forces (:meth:`FrameElement2D.end_forces_from_dofs`) are the
  nodal forces ``{f_local} = [Kl]{u_local}`` recovered directly from the
  local stiffness matrix, in the same local axial/shear/moment DOF order
  documented in :func:`~femtoolkit.analysis.stiffness.frame_element_stiffness_local`.
  These represent the forces each node must apply to the element to hold
  its deformed shape; by construction they satisfy element equilibrium
  (``N1 + N2 = 0``, ``V1 + V2 = 0``, and a moment balance about either
  end), the standard convention for finite element end-force recovery.
* Positive rotation (``rz``, ``theta``) follows the right-hand rule about
  the out-of-plane Z axis (counter-clockwise positive).
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
class FrameElement2D:
    """A two-node 2D Euler-Bernoulli beam/frame element.

    A frame element connects two nodes in the X/Y plane and carries
    axial force, shear force, and bending moment. Its local stiffness
    combines an axial term ``EA/L`` (identical to
    :class:`~femtoolkit.mesh.truss_element.TrussElement2D`) with
    Euler-Bernoulli bending terms built from ``EI``, then transforms the
    result into global X/Y/RZ coordinates using the element's direction
    cosines, exactly as the truss element does for its axial-only case.

    Attributes:
        id: Positive integer identifying the element uniquely within a mesh.
        nodes: The two nodes the element connects, ``(node_1, node_2)``.
        material: Material assigned to the element.
        cross_section: Cross-section assigned to the element. Must
            specify ``second_moment_of_area`` (required for bending
            stiffness); ``extreme_fiber_distance`` is optional and only
            needed for bending-stress post-processing.

    Raises:
        ValidationError: If ``id`` is invalid, ``nodes`` does not contain
            exactly two distinct :class:`Node` instances, ``material`` is
            not a :class:`Material`, ``cross_section`` is not a
            :class:`CrossSection` or lacks ``second_moment_of_area``, or
            the two nodes share the same X and Y coordinates (a
            zero-length element).

    Example:
        >>> element = FrameElement2D(
        ...     id=1,
        ...     nodes=(node_1, node_2),
        ...     material=steel,
        ...     cross_section=CrossSection(area=0.01, second_moment_of_area=8.333e-6),
        ... )
        >>> element.length
        2.0
    """

    id: int
    nodes: tuple[Node, Node]
    material: Material
    cross_section: CrossSection

    dofs_per_node: ClassVar[int] = 3
    """Number of DOFs this element activates per node: ux, uy, rz (see
    :class:`~femtoolkit.analysis.element.StructuralElement`).
    """

    def __post_init__(self) -> None:
        """Validate the frame element immediately after construction.

        Raises:
            ValidationError: If ``id`` is not a positive integer,
                ``nodes`` is not a pair of distinct :class:`Node`
                instances, ``material`` is not a :class:`Material`,
                ``cross_section`` is not a :class:`CrossSection`,
                ``cross_section.second_moment_of_area`` is not set, or
                the resulting element length is not positive.
        """
        validate_two_node_element(
            "FrameElement2D", self.id, self.nodes, self.material, self.cross_section
        )

        if self.cross_section.second_moment_of_area is None:
            raise ValidationError(
                "FrameElement2D requires a CrossSection with second_moment_of_area set "
                "(needed for bending stiffness)."
            )

        if not math.isfinite(self.length) or self.length <= 0:
            raise ValidationError(
                "FrameElement2D requires two nodes at different X/Y coordinates "
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

        A 2D frame element has three DOFs per node, so this returns
        ``((node_1.id, X), (node_1.id, Y), (node_1.id, RZ), (node_2.id, X),
        (node_2.id, Y), (node_2.id, RZ))``.
        """
        # Imported locally: see the note on `stiffness_matrix` below.
        from femtoolkit.analysis.dof import RotationDOF, TranslationDOF

        node_1, node_2 = self.nodes
        return (
            (node_1.id, TranslationDOF.X),
            (node_1.id, TranslationDOF.Y),
            (node_1.id, RotationDOF.RZ),
            (node_2.id, TranslationDOF.X),
            (node_2.id, TranslationDOF.Y),
            (node_2.id, RotationDOF.RZ),
        )

    @property
    def stiffness_matrix(self) -> np.ndarray:
        """Local 6x6 stiffness matrix in global X/Y/RZ coordinates.

        See :func:`~femtoolkit.analysis.stiffness.frame_element_stiffness_2d`
        for the underlying formula.
        """
        # Imported locally: the `analysis` package depends on `mesh` (a
        # StaticLinearAnalysis operates on mesh elements), so a
        # module-level import here would create a circular import between
        # the two packages depending on which one loads first.
        from femtoolkit.analysis.stiffness import frame_element_stiffness_2d

        cos_theta, sin_theta = self.direction_cosines
        return frame_element_stiffness_2d(
            youngs_modulus=self.material.youngs_modulus,
            area=self.cross_section.area,
            second_moment_of_area=self.cross_section.second_moment_of_area,
            length=self.length,
            cos_theta=cos_theta,
            sin_theta=sin_theta,
        )

    def local_displacements(self, displacements: Sequence[float]) -> np.ndarray:
        """Transform global nodal displacements into local element coordinates.

        Args:
            displacements: Global displacements ``[ux1, uy1, rz1, ux2, uy2,
                rz2]``, ordered per :meth:`dof_keys`.

        Returns:
            Local displacements ``[u1, v1, theta1, u2, v2, theta2]``, as a
            length-6 NumPy array.
        """
        from femtoolkit.analysis.transformation import frame_transformation_matrix_2d

        cos_theta, sin_theta = self.direction_cosines
        transformation = frame_transformation_matrix_2d(cos_theta, sin_theta)
        return transformation @ np.asarray(displacements, dtype=float)

    def local_end_forces(self, displacements: Sequence[float]) -> np.ndarray:
        """Compute the local end-force vector ``f_local = [Kl]{u_local}``.

        Args:
            displacements: Global displacements ``[ux1, uy1, rz1, ux2, uy2,
                rz2]``, ordered per :meth:`dof_keys`.

        Returns:
            Local end forces ``[N1, V1, M1, N2, V2, M2]``, as a length-6
            NumPy array. See the module sign-convention docstring.
        """
        from femtoolkit.analysis.stiffness import frame_element_stiffness_local

        local_stiffness = frame_element_stiffness_local(
            youngs_modulus=self.material.youngs_modulus,
            area=self.cross_section.area,
            second_moment_of_area=self.cross_section.second_moment_of_area,
            length=self.length,
        )
        return local_stiffness @ self.local_displacements(displacements)

    def strain(
        self, ux1: float, uy1: float, rz1: float, ux2: float, uy2: float, rz2: float
    ) -> float:
        """Compute axial strain from global nodal displacements.

        Only the translational displacements affect axial strain; ``rz1``
        and ``rz2`` are accepted for a uniform six-argument signature
        (matching :meth:`dof_keys`) but do not enter the calculation, the
        same way :meth:`~femtoolkit.mesh.truss_element.TrussElement2D.strain`
        ignores the component perpendicular to its local axis.

        Args:
            ux1: X displacement at node 1, in meters.
            uy1: Y displacement at node 1, in meters.
            rz1: Rotation at node 1, in radians (unused).
            ux2: X displacement at node 2, in meters.
            uy2: Y displacement at node 2, in meters.
            rz2: Rotation at node 2, in radians (unused).

        Returns:
            Axial strain (dimensionless). Positive is tension (elongation).
        """
        del rz1, rz2
        c, s = self.direction_cosines
        local_u1 = c * ux1 + s * uy1
        local_u2 = c * ux2 + s * uy2
        return (local_u2 - local_u1) / self.length

    def stress(
        self, ux1: float, uy1: float, rz1: float, ux2: float, uy2: float, rz2: float
    ) -> float:
        """Compute axial stress from global nodal displacements, ``sigma = E * epsilon``.

        Returns:
            Axial stress, in pascals. Positive is tension, negative is
            compression.
        """
        return self.material.youngs_modulus * self.strain(ux1, uy1, rz1, ux2, uy2, rz2)

    def axial_force(
        self, ux1: float, uy1: float, rz1: float, ux2: float, uy2: float, rz2: float
    ) -> float:
        """Compute internal axial force from global nodal displacements, ``N = sigma * A``.

        Returns:
            Axial force, in newtons. Positive is tension, negative is
            compression.
        """
        return self.stress(ux1, uy1, rz1, ux2, uy2, rz2) * self.cross_section.area

    def strain_from_dofs(self, displacements: Sequence[float]) -> float:
        """Compute axial strain from ``[ux1, uy1, rz1, ux2, uy2, rz2]``, per :meth:`dof_keys`."""
        return self.strain(*displacements)

    def stress_from_dofs(self, displacements: Sequence[float]) -> float:
        """Compute axial stress from ``[ux1, uy1, rz1, ux2, uy2, rz2]``, per :meth:`dof_keys`."""
        return self.stress(*displacements)

    def axial_force_from_dofs(self, displacements: Sequence[float]) -> float:
        """Compute axial force from ``[ux1, uy1, rz1, ux2, uy2, rz2]``, per :meth:`dof_keys`."""
        return self.axial_force(*displacements)

    def end_forces_from_dofs(self, displacements: Sequence[float]):
        """Compute structured per-end forces from ``[ux1, uy1, rz1, ux2, uy2, rz2]``.

        Args:
            displacements: Global displacements, ordered per :meth:`dof_keys`.

        Returns:
            A :class:`~femtoolkit.results.element_results.FrameElementForces`
            with the axial force, shear force, and bending moment at each
            end, recovered from :meth:`local_end_forces`.
        """
        # Imported locally: `results` is a higher-level package that
        # reports on `mesh` elements; a module-level import here would
        # create a circular import between the two packages.
        from femtoolkit.results.element_results import FrameElementForces, FrameEndForces

        n1, v1, m1, n2, v2, m2 = self.local_end_forces(displacements)
        return FrameElementForces(
            node_1=FrameEndForces(axial_force=n1, shear_force=v1, bending_moment=m1),
            node_2=FrameEndForces(axial_force=n2, shear_force=v2, bending_moment=m2),
        )
