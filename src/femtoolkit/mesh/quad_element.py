"""2D bilinear quadrilateral (Q4) continuum element.

This module defines :class:`QuadElement2D`, a four-node isoparametric
element representing a finite *area* of material -- the second continuum
element in the toolkit, alongside :class:`~femtoolkit.mesh.cst_element.CSTElement2D`
(Version 6). Each node activates two translational DOFs, ``ux`` and
``uy`` -- no rotational DOF, matching every other continuum element.

Unlike the CST element, whose linear shape functions give it a *constant*
strain-displacement matrix, the Q4 element's *bilinear* shape functions
mean its strain-displacement matrix varies from point to point within the
element. This module coordinates the reusable continuum math
(:mod:`femtoolkit.continuum`) that makes this tractable: natural
coordinates and shape functions, isoparametric mapping and the Jacobian,
2x2 Gauss quadrature, and the point-specific strain-displacement matrix --
the element class itself performs no continuum mathematics directly.

**Node ordering.** Nodes must be listed counter-clockwise, matching the
natural-coordinate corners used throughout
:mod:`femtoolkit.continuum.shape_functions`:

.. code-block:: text

    Node 4 -------- Node 3
      |               |
      |               |
    Node 1 -------- Node 2

Unlike :class:`~femtoolkit.mesh.cst_element.CSTElement2D`, which accepts
either winding order via a signed/absolute-area distinction, a Q4
element's node order is fixed: clockwise or otherwise inverted node order
gives a negative Jacobian determinant at every point, which
:mod:`femtoolkit.continuum.jacobian` rejects outright (there is no
single "signed area" for a general quadrilateral to normalize against,
unlike a triangle).

**Representative strain/stress.** A Q4 element's strain and stress are
generally different at every point (see above), so
:meth:`QuadElement2D.strain_from_dofs`/:meth:`QuadElement2D.stress_from_dofs`
report the value at the element's natural-coordinate center
(``xi = eta = 0``) as a single representative value -- consistent with
the :class:`~femtoolkit.analysis.element.ContinuumElement` protocol's
scalar-per-component return shape, and a standard simplified reporting
convention (the alternative, reporting all four Gauss-point values, is
out of scope for this version).

Sign convention: engineering shear strain (``gamma_xy = du/dy + dv/dx``),
matching :mod:`femtoolkit.continuum.strain`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from femtoolkit.continuum.gauss import GAUSS_2X2_POINTS
from femtoolkit.continuum.jacobian import physical_shape_function_derivatives
from femtoolkit.continuum.shape_functions import quad_shape_function_derivatives
from femtoolkit.continuum.strain import quad_strain_displacement_matrix, strain_from_displacements
from femtoolkit.exceptions import ValidationError
from femtoolkit.materials import LinearElastic2D
from femtoolkit.mesh.node import Node


@dataclass
class QuadElement2D:
    """A four-node 2D bilinear quadrilateral (Q4) continuum element.

    Attributes:
        id: Positive integer identifying the element uniquely within a mesh.
        nodes: The four nodes the element connects, ``(node_1, node_2,
            node_3, node_4)``, listed counter-clockwise (see the module
            docstring).
        material: The element's 2D constitutive model (plane stress or
            plane strain).
        thickness: Element thickness, in meters. Must be positive.

    Raises:
        ValidationError: If ``id`` is invalid, ``nodes`` does not contain
            exactly four distinct :class:`Node` instances, ``material``
            is not a :class:`LinearElastic2D`, or ``thickness`` is not a
            positive, finite number.
        DegenerateElementError: If the Jacobian determinant is not
            positive at any of the four Gauss points (degenerate,
            self-intersecting, or clockwise-ordered geometry).

    Example:
        >>> element = QuadElement2D(
        ...     id=1,
        ...     nodes=(node_1, node_2, node_3, node_4),
        ...     material=LinearElastic2D(
        ...         youngs_modulus=210e9, poisson_ratio=0.3, formulation="plane_stress"
        ...     ),
        ...     thickness=0.01,
        ... )
        >>> element.area
        1.0
    """

    id: int
    nodes: tuple[Node, Node, Node, Node]
    material: LinearElastic2D
    thickness: float

    dofs_per_node: ClassVar[int] = 2
    """Number of DOFs this element activates per node: ux, uy (see
    :class:`~femtoolkit.analysis.element.AssemblableElement`). No
    rotational DOF -- a continuum point has no orientation.
    """

    def __post_init__(self) -> None:
        """Validate the Q4 element immediately after construction.

        Raises:
            ValidationError: If ``id`` is not a positive integer,
                ``nodes`` is not a quadruple of distinct :class:`Node`
                instances, ``material`` is not a :class:`LinearElastic2D`,
                or ``thickness`` is not a positive, finite number.
            DegenerateElementError: If the Jacobian determinant is not
                positive at any of the four Gauss points.
        """
        if not isinstance(self.id, int) or isinstance(self.id, bool) or self.id <= 0:
            raise ValidationError(f"QuadElement2D id must be a positive integer, got {self.id!r}.")

        if len(self.nodes) != 4 or not all(isinstance(node, Node) for node in self.nodes):
            raise ValidationError(
                f"QuadElement2D nodes must be a quadruple of Node instances, got {self.nodes!r}."
            )
        node_ids = {node.id for node in self.nodes}
        if len(node_ids) != 4:
            raise ValidationError("QuadElement2D requires four distinct nodes.")

        if not isinstance(self.material, LinearElastic2D):
            raise ValidationError(
                f"QuadElement2D material must be a LinearElastic2D instance, "
                f"got {self.material!r}."
            )

        if not math.isfinite(self.thickness) or self.thickness <= 0:
            raise ValidationError(
                f"QuadElement2D thickness must be positive, got {self.thickness}."
            )

        # Validates geometry as a side effect: raises DegenerateElementError
        # if any Gauss point has a non-positive Jacobian determinant.
        self._gauss_point_data()

    @property
    def _x_coords(self) -> tuple[float, float, float, float]:
        return tuple(node.x for node in self.nodes)

    @property
    def _y_coords(self) -> tuple[float, float, float, float]:
        return tuple(node.y for node in self.nodes)

    def _gauss_point_data(self) -> list[tuple[np.ndarray, float]]:
        """B matrix and Jacobian determinant at each of the 4 Gauss points."""
        x_coords, y_coords = self._x_coords, self._y_coords
        data = []
        for point in GAUSS_2X2_POINTS:
            dn_dxi, dn_deta = quad_shape_function_derivatives(point.xi, point.eta)
            dn_dx, dn_dy, det_j = physical_shape_function_derivatives(
                dn_dxi, dn_deta, x_coords, y_coords
            )
            b_matrix = quad_strain_displacement_matrix(dn_dx, dn_dy)
            data.append((b_matrix, det_j))
        return data

    @property
    def area(self) -> float:
        """The element's physical area, in square meters.

        Computed as ``sum(weight * det(J))`` over the 4 Gauss points --
        the same quadrature used for the stiffness matrix, so this is
        exactly the area this element's own numerical integration
        accounts for.
        """
        x_coords, y_coords = self._x_coords, self._y_coords
        total = 0.0
        for point in GAUSS_2X2_POINTS:
            dn_dxi, dn_deta = quad_shape_function_derivatives(point.xi, point.eta)
            _, _, det_j = physical_shape_function_derivatives(dn_dxi, dn_deta, x_coords, y_coords)
            total += point.weight * det_j
        return total

    def dof_keys(self) -> tuple[tuple[int, int], ...]:
        """The ``(node_id, dof)`` pairs matching ``stiffness_matrix``'s rows/columns.

        A Q4 element has two DOFs per node, so this returns
        ``((node_1.id, X), (node_1.id, Y), ..., (node_4.id, X), (node_4.id, Y))``.
        """
        # Imported locally: see the note on `stiffness_matrix` below.
        from femtoolkit.analysis.dof import TranslationDOF

        keys = []
        for node in self.nodes:
            keys.append((node.id, TranslationDOF.X))
            keys.append((node.id, TranslationDOF.Y))
        return tuple(keys)

    @property
    def centroid_b_matrix(self) -> np.ndarray:
        """The element's 3x8 strain-displacement matrix at its natural-coordinate center.

        Evaluated at ``xi = eta = 0``. See the module docstring for why a
        Q4 element reports strain/stress at a single representative point
        rather than per Gauss point.
        """
        dn_dxi, dn_deta = quad_shape_function_derivatives(0.0, 0.0)
        dn_dx, dn_dy, _ = physical_shape_function_derivatives(
            dn_dxi, dn_deta, self._x_coords, self._y_coords
        )
        return quad_strain_displacement_matrix(dn_dx, dn_dy)

    @property
    def stiffness_matrix(self) -> np.ndarray:
        """Global 8x8 stiffness matrix, evaluated by 2x2 Gauss quadrature.

        See :func:`~femtoolkit.analysis.stiffness.quad_element_stiffness`.
        No coordinate transformation is needed: ``ux``/``uy`` are already
        expressed in global coordinates.
        """
        # Imported locally: the `analysis` package depends on `mesh` (a
        # StaticLinearAnalysis operates on mesh elements), so a
        # module-level import here would create a circular import between
        # the two packages depending on which one loads first.
        from femtoolkit.analysis.stiffness import quad_element_stiffness

        return quad_element_stiffness(
            self._x_coords, self._y_coords, self.thickness, self.material.constitutive_matrix
        )

    def strain_from_dofs(self, displacements: Sequence[float]) -> np.ndarray:
        """Compute the representative (element-center) strain field.

        Args:
            displacements: Nodal displacements ``[u1,v1,u2,v2,u3,v3,u4,v4]``,
                ordered per :meth:`dof_keys`.

        Returns:
            A length-3 NumPy array, ``[epsilon_x, epsilon_y, gamma_xy]``,
            evaluated at the element's natural-coordinate center.
        """
        return strain_from_displacements(self.centroid_b_matrix, displacements)

    def stress_from_dofs(self, displacements: Sequence[float]) -> np.ndarray:
        """Compute the representative (element-center) stress field.

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
        """Compute the representative von Mises equivalent stress.

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
        """Compute the representative in-plane principal stresses.

        Args:
            displacements: Nodal displacements, ordered per :meth:`dof_keys`.

        Returns:
            ``(sigma_1, sigma_2)``, in pascals (``sigma_1 >= sigma_2``).
        """
        from femtoolkit.continuum.stress import principal_stresses_2d

        sigma_x, sigma_y, tau_xy = self.stress_from_dofs(displacements)
        return principal_stresses_2d(sigma_x, sigma_y, tau_xy)
