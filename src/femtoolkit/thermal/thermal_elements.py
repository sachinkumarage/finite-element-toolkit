"""Thermal element formulations: conductivity/capacity matrices, gradients, and flux (Version 20).

**Weak form and the finite element equations.** Starting from the heat
equation ``rho*c*dT/dt = div(k*grad(T)) + Q``, multiplying by a test
function ``w`` and integrating over the domain, the divergence theorem
turns the second-derivative term into a first-derivative one plus a
boundary term (exactly the same manipulation that turns the mechanical
equilibrium equation into the virtual-work principle behind every
stiffness matrix already in this toolkit):

.. code-block:: text

    integral( w * rho*c*dT/dt ) dV + integral( grad(w) . k*grad(T) ) dV
        = integral( w * Q ) dV + boundary flux terms

Substituting the finite element interpolation ``T = N @ T_nodal``,
``w = N`` (Galerkin), and ``grad(T) = G @ T_nodal`` (``G = grad(N)``,
this module's thermal analogue of the mechanical strain-displacement
matrix ``B``) gives, per element:

.. code-block:: text

    C_T = integral( rho*c * N^T @ N ) dV        (capacity matrix)
    K_T = integral( G^T @ (k*I) @ G ) dV         (conductivity matrix)
    F_T = integral( N^T * Q ) dV + boundary flux terms

``K_T`` is the *exact* scalar-field analogue of the mechanical stiffness
matrix ``Ke = integral(B^T @ D @ B) dV`` (``G`` in place of ``B``, the
scalar ``k`` in place of the elastic constitutive matrix ``D``); ``C_T``
is *literally* the same integral as the mechanical consistent mass matrix
``Me = integral(rho * N^T @ N) dV``
(:mod:`femtoolkit.continuum.mass`), with ``rho*c`` in place of ``rho`` and
*not* replicated across multiple DOFs per node (temperature has exactly
one DOF per node, unlike displacement's 2 or 3). This module reuses that
existing mass-matrix machinery directly wherever possible -- see each
``_capacity_matrix`` function's docstring -- rather than re-deriving the
same integral from scratch.

**Reused, not duplicated.** Every element's raw shape-function gradients
(``dNi/dx``, etc.) come from the exact same
:mod:`femtoolkit.continuum.jacobian`/:mod:`femtoolkit.continuum.shape_functions`
machinery the mechanical elements already use; only the *thin* assembly
into ``G`` (a simple vertical stack, not the mechanical Voigt-strain
layout) and the ``K_T``/``F_T`` integrals themselves are new to this
module.

**Elements supported.** A 1D conduction element (reusing
:class:`~femtoolkit.mesh.bar_element.BarElement`'s geometry), a 2D CST
(triangle) element (reusing
:class:`~femtoolkit.mesh.cst_element.CSTElement2D`'s geometry), and the
3D TET4/HEX8 solids -- all purely as *geometry providers*: a thermal
element function never reads ``element.material`` (that is always a
placeholder mechanical material, following this project's established
"separately supplied advanced material" pattern since Version 13); the
thermal properties always come from a separately supplied
:class:`~femtoolkit.thermal.thermal_material.ThermalMaterial`. This keeps
thermal elements completely uncoupled from mechanical ones.
"""

from __future__ import annotations

import numpy as np

from femtoolkit.analysis.assembly import ElementMassContribution, ElementStiffnessContribution
from femtoolkit.analysis.dof import TranslationDOF
from femtoolkit.continuum.gauss import GAUSS_2X2_POINTS, GAUSS_2X2X2_POINTS
from femtoolkit.continuum.jacobian import (
    physical_shape_function_derivatives,
    physical_shape_function_derivatives_3d,
)
from femtoolkit.continuum.mass import (
    hex8_consistent_mass_matrix,
    quad_consistent_mass_matrix,
    tetrahedron_consistent_mass_matrix,
    triangle_consistent_mass_matrix,
)
from femtoolkit.continuum.shape_functions import (
    hex8_shape_function_derivatives,
    hex8_shape_functions,
    quad_shape_function_derivatives,
    quad_shape_functions,
    tet4_shape_function_derivatives,
    tet4_shape_functions,
    triangle_shape_functions,
)
from femtoolkit.continuum.strain import triangle_strain_displacement_matrix
from femtoolkit.exceptions import ValidationError
from femtoolkit.mesh.bar_element import BarElement
from femtoolkit.mesh.cst_element import CSTElement2D
from femtoolkit.mesh.hex8_element import Hex8Element3D
from femtoolkit.mesh.quad_element import QuadElement2D
from femtoolkit.mesh.tet4_element import Tet4Element3D
from femtoolkit.thermal.thermal_material import ThermalMaterial

ThermalCapableElement = BarElement | CSTElement2D | QuadElement2D | Tet4Element3D | Hex8Element3D
"""Element geometries this module can build thermal matrices for."""

_TEMPERATURE_DOF = TranslationDOF.X
"""The single scalar DOF slot used for temperature at each node -- reusing
:class:`~femtoolkit.analysis.dof.DOFMap`'s existing ``dofs_per_node=1``
convention (already established for :class:`~femtoolkit.mesh.bar_element.BarElement`'s
axial DOF), so the existing, unmodified assembly/solve machinery
(:mod:`femtoolkit.analysis.assembly`, :mod:`femtoolkit.analysis.system`)
can be reused verbatim for the thermal system."""


def thermal_dof_keys(element: ThermalCapableElement) -> tuple[tuple[int, int], ...]:
    """Return one ``(node_id, dof)`` key per node, for thermal assembly.

    Args:
        element: Any thermally-capable element.

    Returns:
        A tuple of ``(node_id, _TEMPERATURE_DOF)`` pairs, one per node,
        in node order.
    """
    return tuple((node.id, _TEMPERATURE_DOF) for node in element.nodes)


# --------------------------------------------------------------------------
# 1D bar (conduction along a rod)
# --------------------------------------------------------------------------


def bar_conductivity_matrix(
    element: BarElement, material: ThermalMaterial, temperature: float
) -> np.ndarray:
    """Return the closed-form 2x2 conductivity matrix of a 1D conduction element.

    For a linear (2-node) element, ``dT/dx`` is constant, giving the
    closed-form result -- the thermal analogue of
    :func:`~femtoolkit.analysis.stiffness.bar_element_stiffness`:

    .. code-block:: text

        K_T = (k * A / L) * [[1, -1], [-1, 1]]

    Args:
        element: The bar element (geometry only).
        material: The thermal material.
        temperature: Temperature to evaluate ``k`` at, in kelvin.

    Returns:
        A symmetric 2x2 NumPy array.
    """
    conductivity = material.conductivity_at(temperature)
    factor = conductivity * element.cross_section.area / element.length
    return factor * np.array([[1.0, -1.0], [-1.0, 1.0]])


def bar_capacity_matrix(
    element: BarElement, material: ThermalMaterial, temperature: float
) -> np.ndarray:
    """Return the closed-form 2x2 consistent capacity matrix of a 1D conduction element.

    .. code-block:: text

        C_T = (rho * c * A * L / 6) * [[2, 1], [1, 2]]

    The direct 1D analogue of :func:`~femtoolkit.continuum.mass.triangle_consistent_mass_matrix`'s
    derivation (linear shape functions, closed-form ``integral(Ni*Nj) dx``).

    Args:
        element: The bar element (geometry only).
        material: The thermal material.
        temperature: Temperature to evaluate ``rho*c`` at, in kelvin.

    Returns:
        A symmetric 2x2 NumPy array.
    """
    volumetric_capacity = material.volumetric_heat_capacity_at(temperature)
    factor = volumetric_capacity * element.cross_section.area * element.length / 6.0
    return factor * np.array([[2.0, 1.0], [1.0, 2.0]])


# --------------------------------------------------------------------------
# CST (2D triangle)
# --------------------------------------------------------------------------


def _cst_gradient_matrix(element: CSTElement2D) -> np.ndarray:
    """Return the constant 2x3 thermal gradient matrix ``G`` for a CST element.

    Extracted directly from the existing (and already tested) mechanical
    strain-displacement matrix: row 0 of that matrix holds each node's
    ``dNi/dx`` (at the even, "u" columns), row 1 holds ``dNi/dy`` (at the
    odd, "v" columns) -- see
    :func:`~femtoolkit.continuum.strain.triangle_strain_displacement_matrix`.
    Reusing it this way avoids re-deriving the same triangle-geometry
    gradient formula a second time.
    """
    x1, y1 = element.nodes[0].x, element.nodes[0].y
    x2, y2 = element.nodes[1].x, element.nodes[1].y
    x3, y3 = element.nodes[2].x, element.nodes[2].y
    b_matrix = triangle_strain_displacement_matrix(x1, y1, x2, y2, x3, y3)
    return np.array([b_matrix[0, 0::2], b_matrix[1, 1::2]])


def cst_conductivity_matrix(
    element: CSTElement2D, material: ThermalMaterial, temperature: float
) -> np.ndarray:
    """Return the closed-form 3x3 conductivity matrix of a CST element.

    ``G`` is constant over a CST element (linear shape functions), so
    ``K_T = t * A * G^T @ (k*I2) @ G`` in closed form -- no quadrature
    needed, exactly like the CST mechanical stiffness matrix.

    Args:
        element: The CST element (geometry only).
        material: The thermal material.
        temperature: Temperature to evaluate ``k`` at, in kelvin.

    Returns:
        A symmetric 3x3 NumPy array.
    """
    conductivity = material.conductivity_at(temperature)
    gradient = _cst_gradient_matrix(element)
    return element.thickness * element.area * conductivity * (gradient.T @ gradient)


def cst_capacity_matrix(
    element: CSTElement2D, material: ThermalMaterial, temperature: float
) -> np.ndarray:
    """Return the closed-form 3x3 consistent capacity matrix of a CST element.

    Reuses :func:`~femtoolkit.continuum.mass.triangle_consistent_mass_matrix`
    directly (passing ``rho*c`` as its ``density`` argument) and extracts
    the un-replicated 3x3 scalar block via ``[0::2, 0::2]`` slicing -- the
    mechanical mass matrix is exactly
    ``kron(this_scalar_matrix, identity(2))``, so this recovers the
    scalar capacity matrix without duplicating the underlying
    closed-form area-coordinate integral.

    Args:
        element: The CST element (geometry only).
        material: The thermal material.
        temperature: Temperature to evaluate ``rho*c`` at, in kelvin.

    Returns:
        A symmetric 3x3 NumPy array.
    """
    volumetric_capacity = material.volumetric_heat_capacity_at(temperature)
    mechanical_mass = triangle_consistent_mass_matrix(
        density=volumetric_capacity, area=element.area, thickness=element.thickness
    )
    return mechanical_mass[0::2, 0::2]


# --------------------------------------------------------------------------
# Q4 (2D quadrilateral)
# --------------------------------------------------------------------------


def quad_conductivity_matrix(
    element: QuadElement2D, material: ThermalMaterial, temperature: float
) -> np.ndarray:
    """Return the 4x4 conductivity matrix of a Q4 element, via 2x2 Gauss quadrature.

    ``G`` varies over a Q4 element (bilinear shape functions), so this
    reuses the same 2x2 Gauss quadrature as the Q4 mechanical stiffness
    matrix.

    Args:
        element: The Q4 element (geometry only).
        material: The thermal material.
        temperature: Temperature to evaluate ``k`` at, in kelvin.

    Returns:
        A symmetric 4x4 NumPy array.
    """
    conductivity = material.conductivity_at(temperature)
    x_coords = tuple(node.x for node in element.nodes)
    y_coords = tuple(node.y for node in element.nodes)

    conductivity_matrix = np.zeros((4, 4))
    for point in GAUSS_2X2_POINTS:
        dn_dxi, dn_deta = quad_shape_function_derivatives(point.xi, point.eta)
        dn_dx, dn_dy, det_j = physical_shape_function_derivatives(
            dn_dxi, dn_deta, x_coords, y_coords
        )
        gradient = np.array([dn_dx, dn_dy])
        conductivity_matrix += (
            point.weight * element.thickness * conductivity * (gradient.T @ gradient) * det_j
        )
    return conductivity_matrix


def quad_capacity_matrix(
    element: QuadElement2D, material: ThermalMaterial, temperature: float
) -> np.ndarray:
    """Return the 4x4 consistent capacity matrix of a Q4 element.

    Reuses :func:`~femtoolkit.continuum.mass.quad_consistent_mass_matrix`
    directly (passing ``rho*c``) and slices out the scalar 4x4 block --
    see :func:`cst_capacity_matrix` for the same technique.

    Args:
        element: The Q4 element (geometry only).
        material: The thermal material.
        temperature: Temperature to evaluate ``rho*c`` at, in kelvin.

    Returns:
        A symmetric 4x4 NumPy array.
    """
    volumetric_capacity = material.volumetric_heat_capacity_at(temperature)
    x_coords = tuple(node.x for node in element.nodes)
    y_coords = tuple(node.y for node in element.nodes)
    mechanical_mass = quad_consistent_mass_matrix(
        x_coords, y_coords, density=volumetric_capacity, thickness=element.thickness
    )
    return mechanical_mass[0::2, 0::2]


# --------------------------------------------------------------------------
# TET4
# --------------------------------------------------------------------------


def _tet4_gradient_matrix(element: Tet4Element3D) -> np.ndarray:
    """Return the constant 3x4 thermal gradient matrix ``G`` for a TET4 element."""
    dn_dxi, dn_deta, dn_dzeta = tet4_shape_function_derivatives()
    x = tuple(node.x for node in element.nodes)
    y = tuple(node.y for node in element.nodes)
    z = tuple(node.z for node in element.nodes)
    dn_dx, dn_dy, dn_dz, _ = physical_shape_function_derivatives_3d(
        dn_dxi, dn_deta, dn_dzeta, x, y, z
    )
    return np.array([dn_dx, dn_dy, dn_dz])


def tet4_conductivity_matrix(
    element: Tet4Element3D, material: ThermalMaterial, temperature: float
) -> np.ndarray:
    """Return the closed-form 4x4 conductivity matrix of a TET4 element.

    ``G`` is constant over a TET4 element, so ``K_T = V * G^T @ (k*I3) @ G``
    in closed form -- no quadrature needed, exactly like the TET4
    mechanical stiffness matrix.

    Args:
        element: The TET4 element (geometry only).
        material: The thermal material.
        temperature: Temperature to evaluate ``k`` at, in kelvin.

    Returns:
        A symmetric 4x4 NumPy array.
    """
    conductivity = material.conductivity_at(temperature)
    gradient = _tet4_gradient_matrix(element)
    return element.volume * conductivity * (gradient.T @ gradient)


def tet4_capacity_matrix(
    element: Tet4Element3D, material: ThermalMaterial, temperature: float
) -> np.ndarray:
    """Return the closed-form 4x4 consistent capacity matrix of a TET4 element.

    Reuses :func:`~femtoolkit.continuum.mass.tetrahedron_consistent_mass_matrix`
    directly and slices out the scalar 4x4 block -- see
    :func:`cst_capacity_matrix` for the same technique.

    Args:
        element: The TET4 element (geometry only).
        material: The thermal material.
        temperature: Temperature to evaluate ``rho*c`` at, in kelvin.

    Returns:
        A symmetric 4x4 NumPy array.
    """
    volumetric_capacity = material.volumetric_heat_capacity_at(temperature)
    mechanical_mass = tetrahedron_consistent_mass_matrix(
        density=volumetric_capacity, volume=element.volume
    )
    return mechanical_mass[0::3, 0::3]


# --------------------------------------------------------------------------
# HEX8
# --------------------------------------------------------------------------


def hex8_conductivity_matrix(
    element: Hex8Element3D, material: ThermalMaterial, temperature: float
) -> np.ndarray:
    """Return the 8x8 conductivity matrix of a HEX8 element, via 2x2x2 Gauss quadrature.

    ``G`` varies over a HEX8 element (trilinear shape functions), so this
    reuses the same 2x2x2 Gauss quadrature as the HEX8 mechanical
    stiffness matrix.

    Args:
        element: The HEX8 element (geometry only).
        material: The thermal material.
        temperature: Temperature to evaluate ``k`` at, in kelvin.

    Returns:
        A symmetric 8x8 NumPy array.
    """
    conductivity = material.conductivity_at(temperature)
    x = tuple(node.x for node in element.nodes)
    y = tuple(node.y for node in element.nodes)
    z = tuple(node.z for node in element.nodes)

    conductivity_matrix = np.zeros((8, 8))
    for point in GAUSS_2X2X2_POINTS:
        dn_dxi, dn_deta, dn_dzeta = hex8_shape_function_derivatives(point.xi, point.eta, point.zeta)
        dn_dx, dn_dy, dn_dz, det_j = physical_shape_function_derivatives_3d(
            dn_dxi, dn_deta, dn_dzeta, x, y, z
        )
        gradient = np.array([dn_dx, dn_dy, dn_dz])
        conductivity_matrix += point.weight * conductivity * (gradient.T @ gradient) * det_j
    return conductivity_matrix


def hex8_capacity_matrix(
    element: Hex8Element3D, material: ThermalMaterial, temperature: float
) -> np.ndarray:
    """Return the 8x8 consistent capacity matrix of a HEX8 element.

    Reuses :func:`~femtoolkit.continuum.mass.hex8_consistent_mass_matrix`
    directly and slices out the scalar 8x8 block -- see
    :func:`cst_capacity_matrix` for the same technique.

    Args:
        element: The HEX8 element (geometry only).
        material: The thermal material.
        temperature: Temperature to evaluate ``rho*c`` at, in kelvin.

    Returns:
        A symmetric 8x8 NumPy array.
    """
    volumetric_capacity = material.volumetric_heat_capacity_at(temperature)
    x = tuple(node.x for node in element.nodes)
    y = tuple(node.y for node in element.nodes)
    z = tuple(node.z for node in element.nodes)
    mechanical_mass = hex8_consistent_mass_matrix(x, y, z, density=volumetric_capacity)
    return mechanical_mass[0::3, 0::3]


# --------------------------------------------------------------------------
# Dispatch
# --------------------------------------------------------------------------


def conductivity_contribution(
    element: ThermalCapableElement, material: ThermalMaterial, temperature: float
) -> ElementStiffnessContribution:
    """Return ``element``'s conductivity matrix as an assembly-ready contribution.

    Args:
        element: Any thermally-capable element.
        material: The thermal material.
        temperature: Temperature to evaluate temperature-dependent
            properties at, in kelvin.

    Returns:
        An :class:`~femtoolkit.analysis.assembly.ElementStiffnessContribution`,
        directly usable with
        :func:`~femtoolkit.analysis.assembly.assemble_global_stiffness`.

    Raises:
        ValidationError: If ``element`` is not a supported type.
    """
    if isinstance(element, BarElement):
        matrix = bar_conductivity_matrix(element, material, temperature)
    elif isinstance(element, CSTElement2D):
        matrix = cst_conductivity_matrix(element, material, temperature)
    elif isinstance(element, QuadElement2D):
        matrix = quad_conductivity_matrix(element, material, temperature)
    elif isinstance(element, Tet4Element3D):
        matrix = tet4_conductivity_matrix(element, material, temperature)
    elif isinstance(element, Hex8Element3D):
        matrix = hex8_conductivity_matrix(element, material, temperature)
    else:
        raise ValidationError(f"Unsupported thermal element type: {type(element).__name__}.")
    return ElementStiffnessContribution(thermal_dof_keys(element), matrix)


def capacity_contribution(
    element: ThermalCapableElement, material: ThermalMaterial, temperature: float
) -> ElementMassContribution:
    """Return ``element``'s capacity matrix as an assembly-ready contribution.

    Args:
        element: Any thermally-capable element.
        material: The thermal material.
        temperature: Temperature to evaluate temperature-dependent
            properties at, in kelvin.

    Returns:
        An :class:`~femtoolkit.analysis.assembly.ElementMassContribution`,
        directly usable with
        :func:`~femtoolkit.analysis.assembly.assemble_global_mass`.

    Raises:
        ValidationError: If ``element`` is not a supported type.
    """
    if isinstance(element, BarElement):
        matrix = bar_capacity_matrix(element, material, temperature)
    elif isinstance(element, CSTElement2D):
        matrix = cst_capacity_matrix(element, material, temperature)
    elif isinstance(element, QuadElement2D):
        matrix = quad_capacity_matrix(element, material, temperature)
    elif isinstance(element, Tet4Element3D):
        matrix = tet4_capacity_matrix(element, material, temperature)
    elif isinstance(element, Hex8Element3D):
        matrix = hex8_capacity_matrix(element, material, temperature)
    else:
        raise ValidationError(f"Unsupported thermal element type: {type(element).__name__}.")
    return ElementMassContribution(thermal_dof_keys(element), matrix)


def element_temperature_gradient(
    element: ThermalCapableElement, nodal_temperatures: np.ndarray
) -> np.ndarray:
    """Return the (constant, or element-averaged) temperature gradient ``grad(T)``.

    For the constant-gradient elements (bar, CST, TET4) this is exact.
    For Q4/HEX8 (whose gradient varies within the element), this returns
    the gradient at the element's natural-coordinate origin (``xi=eta=
    (zeta=)0``, the centroid) as a simple, representative single value --
    a caller wanting the full per-Gauss-point field can call the
    per-point machinery directly (see :func:`hex8_conductivity_matrix`
    for the same Gauss-point loop structure).

    Args:
        element: Any thermally-capable element.
        nodal_temperatures: The element's nodal temperatures, ordered to
            match ``element.nodes``.

    Returns:
        A length-1 (bar), length-2 (CST/Q4), or length-3 (TET4/HEX8)
        NumPy array, ``grad(T)``, in K/m.

    Raises:
        ValidationError: If ``element`` is not a supported type.
    """
    temperatures = np.asarray(nodal_temperatures, dtype=float)
    if isinstance(element, BarElement):
        gradient = np.array([[-1.0 / element.length, 1.0 / element.length]])
    elif isinstance(element, CSTElement2D):
        gradient = _cst_gradient_matrix(element)
    elif isinstance(element, QuadElement2D):
        x_coords = tuple(node.x for node in element.nodes)
        y_coords = tuple(node.y for node in element.nodes)
        dn_dxi, dn_deta = quad_shape_function_derivatives(0.0, 0.0)
        dn_dx, dn_dy, _ = physical_shape_function_derivatives(dn_dxi, dn_deta, x_coords, y_coords)
        gradient = np.array([dn_dx, dn_dy])
    elif isinstance(element, Tet4Element3D):
        gradient = _tet4_gradient_matrix(element)
    elif isinstance(element, Hex8Element3D):
        x = tuple(node.x for node in element.nodes)
        y = tuple(node.y for node in element.nodes)
        z = tuple(node.z for node in element.nodes)
        dn_dxi, dn_deta, dn_dzeta = hex8_shape_function_derivatives(0.0, 0.0, 0.0)
        dn_dx, dn_dy, dn_dz, _ = physical_shape_function_derivatives_3d(
            dn_dxi, dn_deta, dn_dzeta, x, y, z
        )
        gradient = np.array([dn_dx, dn_dy, dn_dz])
    else:
        raise ValidationError(f"Unsupported thermal element type: {type(element).__name__}.")
    return gradient @ temperatures


def element_heat_flux(
    element: ThermalCapableElement,
    material: ThermalMaterial,
    nodal_temperatures: np.ndarray,
    temperature: float,
) -> np.ndarray:
    """Return the heat flux ``q = -k * grad(T)`` (Fourier's law).

    Args:
        element: Any thermally-capable element.
        material: The thermal material.
        nodal_temperatures: The element's nodal temperatures, ordered to
            match ``element.nodes``.
        temperature: Temperature to evaluate ``k`` at, in kelvin.

    Returns:
        The heat flux vector, same shape as
        :func:`element_temperature_gradient`'s result, in W/m^2.
    """
    conductivity = material.conductivity_at(temperature)
    gradient = element_temperature_gradient(element, nodal_temperatures)
    return -conductivity * gradient


def element_temperature_at_centroid(
    element: ThermalCapableElement, nodal_temperatures: np.ndarray
) -> float:
    """Return the temperature interpolated to ``element``'s centroid, ``T = N @ T_nodal``.

    Args:
        element: Any thermally-capable element.
        nodal_temperatures: The element's nodal temperatures, ordered to
            match ``element.nodes``.

    Returns:
        The interpolated temperature, in kelvin.

    Raises:
        ValidationError: If ``element`` is not a supported type.
    """
    temperatures = np.asarray(nodal_temperatures, dtype=float)
    if isinstance(element, BarElement):
        n_values = (0.5, 0.5)
    elif isinstance(element, CSTElement2D):
        x1, y1 = element.nodes[0].x, element.nodes[0].y
        x2, y2 = element.nodes[1].x, element.nodes[1].y
        x3, y3 = element.nodes[2].x, element.nodes[2].y
        centroid_x = (x1 + x2 + x3) / 3.0
        centroid_y = (y1 + y2 + y3) / 3.0
        n_values = triangle_shape_functions(centroid_x, centroid_y, x1, y1, x2, y2, x3, y3)
    elif isinstance(element, QuadElement2D):
        n_values = quad_shape_functions(0.0, 0.0)
    elif isinstance(element, Tet4Element3D):
        n_values = tet4_shape_functions(0.25, 0.25, 0.25)
    elif isinstance(element, Hex8Element3D):
        n_values = hex8_shape_functions(0.0, 0.0, 0.0)
    else:
        raise ValidationError(f"Unsupported thermal element type: {type(element).__name__}.")
    return float(np.array(n_values) @ temperatures)
