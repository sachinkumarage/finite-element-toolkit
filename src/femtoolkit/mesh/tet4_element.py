"""4-node linear tetrahedral (TET4) 3D solid element.

This module defines :class:`Tet4Element3D`, the first genuine 3D
continuum element in the toolkit -- representing a finite *volume* of
material, the 3D analogue of :class:`~femtoolkit.mesh.cst_element.CSTElement2D`.
Each node activates three translational DOFs, ``ux``, ``uy``, ``uz`` (no
rotational DOF, matching every other continuum element). Like CST, TET4's
shape functions are linear, so its strain-displacement matrix ``B`` is
*constant* over the element -- the stiffness integral collapses to a
single closed-form multiplication by volume, with no Gauss quadrature
needed (contrast :class:`~femtoolkit.mesh.hex8_element.Hex8Element3D`).

The element coordinates reusable continuum mathematics
(:mod:`femtoolkit.continuum`) rather than embedding it, exactly like
every other element in this toolkit: geometry (via the isoparametric
Jacobian), the strain-displacement matrix, the constitutive matrix, and
stress recovery are all independently testable pure functions. This
class's job is to hold the element's four nodes and material, and
translate between them and that reusable math.

**Orientation policy.** Following :class:`CSTElement2D`'s precedent (see
:mod:`femtoolkit.continuum.geometry`): nodes may be listed in either
orientation. The *signed* volume (equivalently, the signed Jacobian
determinant) is used consistently inside the strain-displacement matrix,
while the *absolute* volume is used everywhere a physical volume is
needed (stiffness, mass) -- so strain, stress, and stiffness are
identical regardless of node winding. This is a deliberate difference
from :class:`~femtoolkit.mesh.hex8_element.Hex8Element3D` (which, like
:class:`~femtoolkit.mesh.quad_element.QuadElement2D`, requires a single
consistent node order): a tetrahedron's signed volume has an unambiguous
sign-flip-on-reorder relationship simple enough to normalize away safely,
the same way a triangle's does, while a general hexahedron does not.

Sign convention: engineering shear strain (Voigt ordering
``[epsilon_xx, epsilon_yy, epsilon_zz, gamma_xy, gamma_yz, gamma_xz]``),
matching :mod:`femtoolkit.continuum.tensor`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from femtoolkit.continuum.jacobian import jacobian_determinant_3d, jacobian_matrix_3d
from femtoolkit.continuum.shape_functions import tet4_shape_function_derivatives
from femtoolkit.continuum.strain import tet4_strain_displacement_matrix
from femtoolkit.continuum.tensor import mean_stress, principal_stresses_3d, voigt_stress_to_tensor
from femtoolkit.exceptions import DegenerateElementError, ValidationError
from femtoolkit.materials.linear_elastic_3d import LinearElastic3D
from femtoolkit.mesh.node import Node

MIN_TETRAHEDRON_VOLUME: float = 1e-12
"""Minimum acceptable absolute tetrahedron volume, in cubic meters.

The 3D analogue of :data:`~femtoolkit.continuum.geometry.MIN_TRIANGLE_AREA`,
guarding against degenerate tetrahedra (coplanar, nearly coplanar, or
duplicate-coordinate nodes), for which the strain-displacement matrix is
undefined (it divides by the Jacobian determinant).
"""


@dataclass
class Tet4Element3D:
    """A four-node linear tetrahedral (TET4) 3D solid continuum element.

    Attributes:
        id: Positive integer identifying the element uniquely within a mesh.
        nodes: The four nodes the element connects, ``(node_1, node_2,
            node_3, node_4)``.
        material: The element's 3D isotropic linear elastic constitutive
            model.

    Raises:
        ValidationError: If ``id`` is invalid, ``nodes`` does not contain
            exactly four distinct :class:`~femtoolkit.mesh.node.Node`
            instances, or ``material`` is not a
            :class:`~femtoolkit.materials.linear_elastic_3d.LinearElastic3D`.
        DegenerateElementError: If the four nodes are coplanar, nearly
            coplanar, or coincide (zero or near-zero volume).

    Example:
        >>> element = Tet4Element3D(
        ...     id=1,
        ...     nodes=(node_1, node_2, node_3, node_4),
        ...     material=LinearElastic3D(youngs_modulus=200e9, poisson_ratio=0.3, density=7850.0),
        ... )
        >>> element.volume
        0.16666666666666666
    """

    id: int
    nodes: tuple[Node, Node, Node, Node]
    material: LinearElastic3D

    dofs_per_node: ClassVar[int] = 3
    """Number of DOFs this element activates per node: ux, uy, uz (see
    :class:`~femtoolkit.analysis.element.AssemblableElement`). No
    rotational DOF -- a continuum point has no orientation.
    """

    def __post_init__(self) -> None:
        """Validate the TET4 element immediately after construction.

        Raises:
            ValidationError: If ``id`` is not a positive integer,
                ``nodes`` is not a quadruple of distinct
                :class:`~femtoolkit.mesh.node.Node` instances, or
                ``material`` is not a
                :class:`~femtoolkit.materials.linear_elastic_3d.LinearElastic3D`.
            DegenerateElementError: If the resulting tetrahedron volume
                is zero or near-zero.
        """
        if not isinstance(self.id, int) or isinstance(self.id, bool) or self.id <= 0:
            raise ValidationError(f"Tet4Element3D id must be a positive integer, got {self.id!r}.")

        if len(self.nodes) != 4 or not all(isinstance(node, Node) for node in self.nodes):
            raise ValidationError(
                f"Tet4Element3D nodes must be a quadruple of Node instances, got {self.nodes!r}."
            )
        node_ids = {node.id for node in self.nodes}
        if len(node_ids) != 4:
            raise ValidationError("Tet4Element3D requires four distinct nodes.")

        if not isinstance(self.material, LinearElastic3D):
            raise ValidationError(
                f"Tet4Element3D material must be a LinearElastic3D instance, "
                f"got {self.material!r}."
            )

        if abs(self.signed_volume) < MIN_TETRAHEDRON_VOLUME:
            raise DegenerateElementError(
                f"Tet4Element3D {self.id} has volume {self.signed_volume} "
                "(coplanar, nearly coplanar, or duplicate-coordinate nodes)."
            )

    @property
    def _x_coords(self) -> tuple[float, float, float, float]:
        return tuple(node.x for node in self.nodes)

    @property
    def _y_coords(self) -> tuple[float, float, float, float]:
        return tuple(node.y for node in self.nodes)

    @property
    def _z_coords(self) -> tuple[float, float, float, float]:
        return tuple(node.z for node in self.nodes)

    def _jacobian(self) -> np.ndarray:
        """The element's constant 3x3 isoparametric Jacobian matrix."""
        dn_dxi, dn_deta, dn_dzeta = tet4_shape_function_derivatives()
        return jacobian_matrix_3d(
            dn_dxi, dn_deta, dn_dzeta, self._x_coords, self._y_coords, self._z_coords
        )

    @property
    def signed_volume(self) -> float:
        """The tetrahedron's signed volume, ``det(J) / 6`` (see the orientation policy above)."""
        return jacobian_determinant_3d(self._jacobian()) / 6.0

    @property
    def volume(self) -> float:
        """The tetrahedron's physical (always positive) volume, in cubic meters."""
        return abs(self.signed_volume)

    def dof_keys(self) -> tuple[tuple[int, int], ...]:
        """The ``(node_id, dof)`` pairs matching ``stiffness_matrix``'s rows/columns.

        A TET4 element has three DOFs per node, so this returns 12
        entries: ``((node_1.id, X), (node_1.id, Y), (node_1.id, Z),
        (node_2.id, X), ...)``.
        """
        from femtoolkit.analysis.dof import TranslationDOF

        keys = []
        for node in self.nodes:
            keys.append((node.id, TranslationDOF.X))
            keys.append((node.id, TranslationDOF.Y))
            keys.append((node.id, TranslationDOF.Z))
        return tuple(keys)

    @property
    def b_matrix(self) -> np.ndarray:
        """The element's constant 6x12 strain-displacement matrix.

        Computed via the isoparametric Jacobian, using its *signed*
        inverse directly (not :func:`~femtoolkit.continuum.jacobian.inverse_jacobian_3d`,
        which requires a positive determinant) -- consistent with the
        orientation policy documented in the module docstring, this
        makes ``B``, and therefore the recovered strain and stress,
        identical regardless of whether the four nodes are listed in
        either valid tetrahedral winding. Degeneracy is already excluded
        by construction (see ``__post_init__``), so the inverse is
        always well-defined here.
        """
        dn_dxi, dn_deta, dn_dzeta = tet4_shape_function_derivatives()
        jacobian_inverse = np.linalg.inv(self._jacobian())
        natural_derivatives = np.array([dn_dxi, dn_deta, dn_dzeta], dtype=float)
        physical_derivatives = jacobian_inverse @ natural_derivatives
        return tet4_strain_displacement_matrix(
            physical_derivatives[0], physical_derivatives[1], physical_derivatives[2]
        )

    @property
    def stiffness_matrix(self) -> np.ndarray:
        """Global 12x12 stiffness matrix, ``Ke = B^T * D * B * V``.

        See :func:`~femtoolkit.analysis.stiffness.tet4_element_stiffness`.
        No coordinate transformation is needed: ``ux``/``uy``/``uz`` are
        already expressed in global coordinates.
        """
        from femtoolkit.analysis.stiffness import tet4_element_stiffness

        return tet4_element_stiffness(
            volume=self.volume, b_matrix=self.b_matrix, d_matrix=self.material.constitutive_matrix
        )

    def strain_from_dofs(self, displacements: Sequence[float]) -> np.ndarray:
        """Compute the constant strain field from 12 nodal displacements.

        Args:
            displacements: Nodal displacements, ordered per :meth:`dof_keys`.

        Returns:
            A length-6 NumPy array, Voigt
            ``[epsilon_xx, epsilon_yy, epsilon_zz, gamma_xy, gamma_yz, gamma_xz]``.
        """
        from femtoolkit.continuum.strain import strain_from_displacements

        return strain_from_displacements(self.b_matrix, displacements)

    def stress_from_dofs(self, displacements: Sequence[float]) -> np.ndarray:
        """Compute the constant stress field from 12 nodal displacements.

        Args:
            displacements: Nodal displacements, ordered per :meth:`dof_keys`.

        Returns:
            A length-6 NumPy array, Voigt
            ``[sigma_xx, sigma_yy, sigma_zz, tau_xy, tau_yz, tau_xz]``.
        """
        from femtoolkit.continuum.stress import stress_from_strain

        return stress_from_strain(
            self.material.constitutive_matrix, self.strain_from_dofs(displacements)
        )

    def von_mises_from_dofs(self, displacements: Sequence[float]) -> float:
        """Compute the von Mises equivalent stress from 12 nodal displacements.

        Args:
            displacements: Nodal displacements, ordered per :meth:`dof_keys`.

        Returns:
            The von Mises equivalent stress, in pascals.
        """
        from femtoolkit.continuum.stress import von_mises_3d

        return von_mises_3d(*self.stress_from_dofs(displacements))

    def hydrostatic_stress_from_dofs(self, displacements: Sequence[float]) -> float:
        """Compute the mean (hydrostatic) normal stress from 12 nodal displacements.

        Args:
            displacements: Nodal displacements, ordered per :meth:`dof_keys`.

        Returns:
            The mean stress ``sigma_m = tr(sigma) / 3``, in pascals.
        """
        stress_tensor = voigt_stress_to_tensor(self.stress_from_dofs(displacements))
        return mean_stress(stress_tensor)

    def principal_stresses_from_dofs(
        self, displacements: Sequence[float]
    ) -> tuple[float, float, float]:
        """Compute the three principal stresses from 12 nodal displacements.

        Args:
            displacements: Nodal displacements, ordered per :meth:`dof_keys`.

        Returns:
            ``(sigma_1, sigma_2, sigma_3)``, in pascals
            (``sigma_1 >= sigma_2 >= sigma_3``).
        """
        stress_tensor = voigt_stress_to_tensor(self.stress_from_dofs(displacements))
        return principal_stresses_3d(stress_tensor)
