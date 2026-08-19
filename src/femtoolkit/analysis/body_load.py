"""Body forces (e.g. gravity), converted into equivalent nodal loads.

A body force acts throughout an element's area/volume, rather than along
a boundary (contrast with
:mod:`femtoolkit.analysis.distributed_load`'s surface tractions). Gravity
is the toolkit's first body force: a uniform acceleration field ``g`` in
a fixed direction, producing a distributed force per unit volume
``rho * g`` inside every continuum element.

Equivalent nodal forces are obtained the standard finite-element way, by
integrating the body force against each element's own shape functions:

.. code-block:: text

    fe = integral( N^T * rho * g_vec ) * thickness dA

For a **CST element** (Version 6), the shape functions are the
triangle's area coordinates, whose integral over the triangle is exact
and well known in closed form -- ``integral(Li) dA = A/3`` for each of
the three area coordinates -- so the total element weight
``W = density * area * thickness * g`` splits evenly across its three
nodes, with no numerical integration needed.

For a **Q4 element** (Version 7), the bilinear shape functions do not
split evenly in general (only for a rectangle/parallelogram), so this
module evaluates the same integral with the existing 2x2 Gauss
quadrature (see :mod:`femtoolkit.continuum.gauss`), reusing the
isoparametric machinery already used for the element's stiffness matrix.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from femtoolkit.analysis.dof import TranslationDOF
from femtoolkit.analysis.loads import NodalLoad
from femtoolkit.exceptions import ValidationError
from femtoolkit.mesh.cst_element import CSTElement2D
from femtoolkit.mesh.mesh import Mesh
from femtoolkit.mesh.quad_element import QuadElement2D

_UNIT_VECTOR_TOLERANCE: float = 1e-9


@dataclass
class GravityLoad:
    """A uniform gravitational (or other constant) body-force acceleration.

    Attributes:
        g: Acceleration magnitude, in m/s^2. Must be positive and finite.
        direction: Unit vector ``(dx, dy)`` the acceleration acts along.
            Defaults to ``(0.0, -1.0)`` (downward, i.e. the global -Y
            axis).

    Raises:
        ValidationError: If ``g`` is not positive and finite, or
            ``direction`` is not a unit vector.

    Example:
        >>> gravity = GravityLoad(g=9.81)
        >>> gravity.acceleration_vector()
        (0.0, -9.81)
    """

    g: float = 9.81
    direction: tuple[float, float] = (0.0, -1.0)

    def __post_init__(self) -> None:
        """Validate the gravity load immediately after construction.

        Raises:
            ValidationError: If ``g`` is not positive and finite, or
                ``direction`` is not a unit vector.
        """
        if not math.isfinite(self.g) or self.g <= 0:
            raise ValidationError(f"GravityLoad g must be positive, got {self.g}.")

        dx, dy = self.direction
        if not (math.isfinite(dx) and math.isfinite(dy)):
            raise ValidationError(f"GravityLoad direction must be finite, got {self.direction}.")

        length = math.hypot(dx, dy)
        if abs(length - 1.0) > _UNIT_VECTOR_TOLERANCE:
            raise ValidationError(
                f"GravityLoad direction must be a unit vector, got {self.direction} "
                f"(length {length})."
            )

    def acceleration_vector(self) -> tuple[float, float]:
        """The full acceleration vector, ``g * direction``, in m/s^2."""
        dx, dy = self.direction
        return (self.g * dx, self.g * dy)


def _element_density(element: CSTElement2D | QuadElement2D) -> float:
    density = element.material.density
    if density is None:
        raise ValidationError(
            f"Element {element.id} ({type(element).__name__})'s material has no density "
            "set; density is required for gravity/body-force loading."
        )
    return density


def _cst_gravity_nodal_loads(
    element: CSTElement2D, acceleration: tuple[float, float]
) -> list[NodalLoad]:
    """Exact equal 1/3 split of the element's total weight across its three nodes."""
    density = _element_density(element)
    gx, gy = acceleration
    share = density * element.area * element.thickness / 3.0

    loads: list[NodalLoad] = []
    for node in element.nodes:
        loads.append(NodalLoad(node.id, TranslationDOF.X, share * gx))
        loads.append(NodalLoad(node.id, TranslationDOF.Y, share * gy))
    return loads


def _quad_gravity_nodal_loads(
    element: QuadElement2D, acceleration: tuple[float, float]
) -> list[NodalLoad]:
    """2x2 Gauss quadrature of ``integral(Ni * rho * g) dA`` per node."""
    from femtoolkit.continuum.gauss import GAUSS_2X2_POINTS
    from femtoolkit.continuum.jacobian import physical_shape_function_derivatives
    from femtoolkit.continuum.shape_functions import (
        quad_shape_function_derivatives,
        quad_shape_functions,
    )

    density = _element_density(element)
    gx, gy = acceleration
    x_coords = tuple(node.x for node in element.nodes)
    y_coords = tuple(node.y for node in element.nodes)

    node_force_x = [0.0, 0.0, 0.0, 0.0]
    node_force_y = [0.0, 0.0, 0.0, 0.0]
    for point in GAUSS_2X2_POINTS:
        n_values = quad_shape_functions(point.xi, point.eta)
        dn_dxi, dn_deta = quad_shape_function_derivatives(point.xi, point.eta)
        _, _, det_j = physical_shape_function_derivatives(dn_dxi, dn_deta, x_coords, y_coords)
        weight_factor = point.weight * det_j * density * element.thickness
        for i, n_value in enumerate(n_values):
            node_force_x[i] += weight_factor * n_value * gx
            node_force_y[i] += weight_factor * n_value * gy

    loads: list[NodalLoad] = []
    for i, node in enumerate(element.nodes):
        loads.append(NodalLoad(node.id, TranslationDOF.X, node_force_x[i]))
        loads.append(NodalLoad(node.id, TranslationDOF.Y, node_force_y[i]))
    return loads


def gravity_load_to_nodal_loads(mesh: Mesh, gravity: GravityLoad) -> list[NodalLoad]:
    """Convert a gravity (body-force) load into equivalent nodal loads.

    Every continuum element (CST or Q4) in ``mesh`` contributes its own
    share of nodal force; elements sharing a node each add their own
    contribution, which :func:`~femtoolkit.analysis.system.build_force_vector`
    sums, exactly like a distributed boundary traction.

    Args:
        mesh: The mesh to compute gravity loads for. Only
            :class:`~femtoolkit.mesh.cst_element.CSTElement2D` and
            :class:`~femtoolkit.mesh.quad_element.QuadElement2D` elements
            contribute; other element types (which have no area/volume
            concept) are skipped.
        gravity: The gravity load to convert.

    Returns:
        One :class:`~femtoolkit.analysis.loads.NodalLoad` per DOF
        contribution (2 per node per contributing element).

    Raises:
        ValidationError: If any contributing element's material has no
            density set.

    Example:
        >>> gravity = GravityLoad(g=9.81)
        >>> loads = gravity_load_to_nodal_loads(mesh, gravity)
        >>> sum(load.value for load in loads if load.dof == TranslationDOF.Y)
        -49.05
    """
    acceleration = gravity.acceleration_vector()
    nodal_loads: list[NodalLoad] = []
    for element in mesh.elements:
        if isinstance(element, CSTElement2D):
            nodal_loads.extend(_cst_gravity_nodal_loads(element, acceleration))
        elif isinstance(element, QuadElement2D):
            nodal_loads.extend(_quad_gravity_nodal_loads(element, acceleration))
    return nodal_loads
