"""8-node trilinear hexahedral (HEX8) 3D solid element.

This module defines :class:`Hex8Element3D`, a eight-node isoparametric
element representing a finite *volume* of material -- the second 3D
continuum element in the toolkit, alongside
:class:`~femtoolkit.mesh.tet4_element.Tet4Element3D`. Each node activates
three translational DOFs, ``ux``, ``uy``, ``uz`` (no rotational DOF,
matching every other continuum element).

Unlike TET4, whose linear shape functions give it a *constant*
strain-displacement matrix, HEX8's *trilinear* shape functions mean its
strain-displacement matrix varies from point to point within the
element -- the direct 3D analogue of
:class:`~femtoolkit.mesh.quad_element.QuadElement2D`. This module
coordinates the reusable continuum math (:mod:`femtoolkit.continuum`)
that makes this tractable: natural coordinates and shape functions,
isoparametric mapping and the Jacobian, 2x2x2 Gauss quadrature, and the
point-specific strain-displacement matrix -- the element class itself
performs no continuum mathematics directly.

**Node ordering.** Nodes must be listed per the standard isoparametric
hexahedron convention documented in
:data:`~femtoolkit.continuum.shape_functions._HEX8_NATURAL_COORDS`
(bottom face counter-clockwise, then the top face directly above it, same
winding). Like :class:`QuadElement2D` (and unlike
:class:`~femtoolkit.mesh.tet4_element.Tet4Element3D`), an inconsistent or
inverted node order gives a non-positive Jacobian determinant at one or
more Gauss points, which :mod:`femtoolkit.continuum.jacobian` rejects
outright -- there is no single "signed volume" to normalize a general
hexahedron against, unlike a tetrahedron.

**Representative strain/stress.** A HEX8 element's strain and stress are
generally different at every point (see above), so
:meth:`Hex8Element3D.strain_from_dofs`/:meth:`Hex8Element3D.stress_from_dofs`
report the value at the element's natural-coordinate center
(``xi = eta = zeta = 0``) as a single representative value -- the same
simplified reporting convention :class:`QuadElement2D` uses.

Sign convention: engineering shear strain (Voigt ordering
``[epsilon_xx, epsilon_yy, epsilon_zz, gamma_xy, gamma_yz, gamma_xz]``),
matching :mod:`femtoolkit.continuum.tensor`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from femtoolkit.continuum.gauss import GAUSS_2X2X2_POINTS
from femtoolkit.continuum.jacobian import physical_shape_function_derivatives_3d
from femtoolkit.continuum.shape_functions import hex8_shape_function_derivatives
from femtoolkit.continuum.strain import hex8_strain_displacement_matrix, strain_from_displacements
from femtoolkit.continuum.tensor import mean_stress, principal_stresses_3d, voigt_stress_to_tensor
from femtoolkit.exceptions import ValidationError
from femtoolkit.materials.linear_elastic_3d import LinearElastic3D
from femtoolkit.mesh.node import Node


@dataclass
class Hex8Element3D:
    """An eight-node trilinear hexahedral (HEX8) 3D solid continuum element.

    Attributes:
        id: Positive integer identifying the element uniquely within a mesh.
        nodes: The eight nodes the element connects, ``(node_1, ...,
            node_8)``, ordered per the module docstring's convention.
        material: The element's 3D isotropic linear elastic constitutive
            model.

    Raises:
        ValidationError: If ``id`` is invalid, ``nodes`` does not contain
            exactly eight distinct :class:`~femtoolkit.mesh.node.Node`
            instances, or ``material`` is not a
            :class:`~femtoolkit.materials.linear_elastic_3d.LinearElastic3D`.
        DegenerateElementError: If the Jacobian determinant is not
            positive at any of the eight Gauss points (degenerate,
            self-intersecting, or inverted node order).

    Example:
        >>> element = Hex8Element3D(
        ...     id=1,
        ...     nodes=(node_1, node_2, node_3, node_4, node_5, node_6, node_7, node_8),
        ...     material=LinearElastic3D(youngs_modulus=200e9, poisson_ratio=0.3, density=7850.0),
        ... )
        >>> element.volume
        1.0
    """

    id: int
    nodes: tuple[Node, Node, Node, Node, Node, Node, Node, Node]
    material: LinearElastic3D

    dofs_per_node: ClassVar[int] = 3
    """Number of DOFs this element activates per node: ux, uy, uz (see
    :class:`~femtoolkit.analysis.element.AssemblableElement`). No
    rotational DOF -- a continuum point has no orientation.
    """

    def __post_init__(self) -> None:
        """Validate the HEX8 element immediately after construction.

        Raises:
            ValidationError: If ``id`` is not a positive integer,
                ``nodes`` is not an octuple of distinct
                :class:`~femtoolkit.mesh.node.Node` instances, or
                ``material`` is not a
                :class:`~femtoolkit.materials.linear_elastic_3d.LinearElastic3D`.
            DegenerateElementError: If the Jacobian determinant is not
                positive at any of the eight Gauss points.
        """
        if not isinstance(self.id, int) or isinstance(self.id, bool) or self.id <= 0:
            raise ValidationError(f"Hex8Element3D id must be a positive integer, got {self.id!r}.")

        if len(self.nodes) != 8 or not all(isinstance(node, Node) for node in self.nodes):
            raise ValidationError(
                f"Hex8Element3D nodes must be an octuple of Node instances, got {self.nodes!r}."
            )
        node_ids = {node.id for node in self.nodes}
        if len(node_ids) != 8:
            raise ValidationError("Hex8Element3D requires eight distinct nodes.")

        if not isinstance(self.material, LinearElastic3D):
            raise ValidationError(
                f"Hex8Element3D material must be a LinearElastic3D instance, "
                f"got {self.material!r}."
            )

        # Validates geometry as a side effect: raises DegenerateElementError
        # if any Gauss point has a non-positive Jacobian determinant.
        self._gauss_point_data()

    @property
    def _x_coords(self) -> tuple[float, ...]:
        return tuple(node.x for node in self.nodes)

    @property
    def _y_coords(self) -> tuple[float, ...]:
        return tuple(node.y for node in self.nodes)

    @property
    def _z_coords(self) -> tuple[float, ...]:
        return tuple(node.z for node in self.nodes)

    def _gauss_point_data(self) -> list[tuple[np.ndarray, float]]:
        """B matrix and Jacobian determinant at each of the 8 Gauss points."""
        x_coords, y_coords, z_coords = self._x_coords, self._y_coords, self._z_coords
        data = []
        for point in GAUSS_2X2X2_POINTS:
            dn_dxi, dn_deta, dn_dzeta = hex8_shape_function_derivatives(
                point.xi, point.eta, point.zeta
            )
            dn_dx, dn_dy, dn_dz, det_j = physical_shape_function_derivatives_3d(
                dn_dxi, dn_deta, dn_dzeta, x_coords, y_coords, z_coords
            )
            b_matrix = hex8_strain_displacement_matrix(dn_dx, dn_dy, dn_dz)
            data.append((b_matrix, det_j))
        return data

    @property
    def volume(self) -> float:
        """The element's physical volume, in cubic meters.

        Computed as ``sum(weight * det(J))`` over the 8 Gauss points --
        the same quadrature used for the stiffness matrix, so this is
        exactly the volume this element's own numerical integration
        accounts for.
        """
        return sum(
            point.weight * det_j
            for point, (_, det_j) in zip(GAUSS_2X2X2_POINTS, self._gauss_point_data(), strict=True)
        )

    def dof_keys(self) -> tuple[tuple[int, int], ...]:
        """The ``(node_id, dof)`` pairs matching ``stiffness_matrix``'s rows/columns.

        A HEX8 element has three DOFs per node, so this returns 24
        entries: ``((node_1.id, X), (node_1.id, Y), (node_1.id, Z), ...,
        (node_8.id, Z))``.
        """
        from femtoolkit.analysis.dof import TranslationDOF

        keys = []
        for node in self.nodes:
            keys.append((node.id, TranslationDOF.X))
            keys.append((node.id, TranslationDOF.Y))
            keys.append((node.id, TranslationDOF.Z))
        return tuple(keys)

    @property
    def centroid_b_matrix(self) -> np.ndarray:
        """The element's 6x24 strain-displacement matrix at its natural-coordinate center.

        Evaluated at ``xi = eta = zeta = 0``. See the module docstring
        for why a HEX8 element reports strain/stress at a single
        representative point rather than per Gauss point.
        """
        dn_dxi, dn_deta, dn_dzeta = hex8_shape_function_derivatives(0.0, 0.0, 0.0)
        dn_dx, dn_dy, dn_dz, _ = physical_shape_function_derivatives_3d(
            dn_dxi, dn_deta, dn_dzeta, self._x_coords, self._y_coords, self._z_coords
        )
        return hex8_strain_displacement_matrix(dn_dx, dn_dy, dn_dz)

    @property
    def stiffness_matrix(self) -> np.ndarray:
        """Global 24x24 stiffness matrix, evaluated by 2x2x2 Gauss quadrature.

        See :func:`~femtoolkit.analysis.stiffness.hex8_element_stiffness`.
        No coordinate transformation is needed: ``ux``/``uy``/``uz`` are
        already expressed in global coordinates.
        """
        from femtoolkit.analysis.stiffness import hex8_element_stiffness

        return hex8_element_stiffness(
            self._x_coords, self._y_coords, self._z_coords, self.material.constitutive_matrix
        )

    def strain_from_dofs(self, displacements: Sequence[float]) -> np.ndarray:
        """Compute the representative (element-center) strain field.

        Args:
            displacements: Nodal displacements (24 entries), ordered per
                :meth:`dof_keys`.

        Returns:
            A length-6 NumPy array, Voigt
            ``[epsilon_xx, epsilon_yy, epsilon_zz, gamma_xy, gamma_yz, gamma_xz]``,
            evaluated at the element's natural-coordinate center.
        """
        return strain_from_displacements(self.centroid_b_matrix, displacements)

    def stress_from_dofs(self, displacements: Sequence[float]) -> np.ndarray:
        """Compute the representative (element-center) stress field.

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
        """Compute the representative von Mises equivalent stress.

        Args:
            displacements: Nodal displacements, ordered per :meth:`dof_keys`.

        Returns:
            The von Mises equivalent stress, in pascals.
        """
        from femtoolkit.continuum.stress import von_mises_3d

        return von_mises_3d(*self.stress_from_dofs(displacements))

    def hydrostatic_stress_from_dofs(self, displacements: Sequence[float]) -> float:
        """Compute the representative mean (hydrostatic) normal stress.

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
        """Compute the representative principal stresses.

        Args:
            displacements: Nodal displacements, ordered per :meth:`dof_keys`.

        Returns:
            ``(sigma_1, sigma_2, sigma_3)``, in pascals
            (``sigma_1 >= sigma_2 >= sigma_3``).
        """
        stress_tensor = voigt_stress_to_tensor(self.stress_from_dofs(displacements))
        return principal_stresses_3d(stress_tensor)
