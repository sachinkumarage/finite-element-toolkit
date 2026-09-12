"""Identifying and integrating over element surfaces, for convection/radiation (Version 21).

Convection and radiation act on a **surface** -- one edge of a 2D
element, or one face of a 3D element -- not on an element's whole
area/volume. :class:`ThermalSurface` identifies such a surface
unambiguously and safely: rather than trust a caller-supplied, possibly
inconsistently-ordered list of node IDs (risky for a HEX8 face, whose
bilinear surface integral needs a *consistent* winding -- see
:mod:`femtoolkit.continuum.surface`), it names an element and *which* of
that element's own faces/edges is meant, by index. The actual node
coordinates are always looked up from the mesh's own connectivity and
the fixed local face tables, never from arbitrary user input.

Supported host elements: CST and Q4 (each edge is a straight 2D line,
reusing :mod:`femtoolkit.continuum.edge`) and TET4/HEX8 (each face is a
flat triangle or a bilinear quad in 3D, reusing
:mod:`femtoolkit.continuum.surface`). A 1D bar element has no lateral
surface in this toolkit's idealization (it is a line, not a solid of
revolution), so it is intentionally unsupported here -- consistent with
the Version 20 thermal element set, which never modeled a bar's
cross-sectional perimeter either.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from femtoolkit.analysis.assembly import ElementStiffnessContribution
from femtoolkit.analysis.dof import DOFMap, TranslationDOF
from femtoolkit.continuum.edge import (
    edge_consistent_conductance_matrix,
    edge_equivalent_nodal_force,
)
from femtoolkit.continuum.surface import (
    HEX8_FACES,
    TET4_FACES,
    quad_face_area,
    quad_face_consistent_matrix,
    quad_face_load_vector,
    triangle_face_area,
    triangle_face_conductance_matrix,
    triangle_face_load_vector,
)
from femtoolkit.exceptions import ValidationError
from femtoolkit.mesh.cst_element import CSTElement2D
from femtoolkit.mesh.hex8_element import Hex8Element3D
from femtoolkit.mesh.mesh import Mesh
from femtoolkit.mesh.quad_element import QuadElement2D
from femtoolkit.mesh.tet4_element import Tet4Element3D
from femtoolkit.thermal.thermal_loads import ThermalLoad

_TEMPERATURE_DOF = TranslationDOF.X

_CST_EDGES: tuple[tuple[int, int], ...] = ((0, 1), (1, 2), (2, 0))
_QUAD_EDGES: tuple[tuple[int, int], ...] = ((0, 1), (1, 2), (2, 3), (3, 0))

SurfaceCapableElement = CSTElement2D | QuadElement2D | Tet4Element3D | Hex8Element3D
"""Element geometries :class:`ThermalSurface` can identify a face/edge of."""


@dataclass(frozen=True)
class ThermalSurface:
    """A single face (3D) or edge (2D) of one mesh element, identified by index.

    Attributes:
        element_id: The ID of the element the surface belongs to.
        local_face_index: Which of that element's faces/edges is meant:
            ``0, 1, 2`` for a CST element's three edges; ``0, 1, 2, 3``
            for a Q4 element's four edges; ``0-3`` for a TET4 element's
            four faces (see
            :data:`~femtoolkit.continuum.surface.TET4_FACES`); ``0-5``
            for a HEX8 element's six faces (see
            :data:`~femtoolkit.continuum.surface.HEX8_FACES`).

    Raises:
        ValidationError: If ``element_id`` is not a positive integer, or
            ``local_face_index`` is negative.

    Example:
        >>> hot_face = ThermalSurface(element_id=1, local_face_index=0)
    """

    element_id: int
    local_face_index: int

    def __post_init__(self) -> None:
        """Validate the surface's identifying indices immediately after construction.

        Raises:
            ValidationError: If ``element_id`` is not a positive
                integer, or ``local_face_index`` is not a non-negative
                integer.
        """
        if (
            not isinstance(self.element_id, int)
            or isinstance(self.element_id, bool)
            or self.element_id <= 0
        ):
            raise ValidationError(
                f"ThermalSurface element_id must be a positive integer, got {self.element_id!r}."
            )
        if (
            not isinstance(self.local_face_index, int)
            or isinstance(self.local_face_index, bool)
            or self.local_face_index < 0
        ):
            raise ValidationError(
                "ThermalSurface local_face_index must be a non-negative integer, got "
                f"{self.local_face_index!r}."
            )


def _resolve(mesh: Mesh, surface: ThermalSurface) -> SurfaceCapableElement:
    """Look up the element a surface belongs to, checked for a supported type."""
    element = mesh.get_element(surface.element_id)
    if not isinstance(element, SurfaceCapableElement):
        raise ValidationError(
            f"ThermalSurface only supports CST/Q4/TET4/HEX8 elements, got "
            f"{type(element).__name__} (id={surface.element_id})."
        )
    return element


def _edge_node_pair(
    element: CSTElement2D | QuadElement2D, local_face_index: int
) -> tuple[int, int]:
    edges = _CST_EDGES if isinstance(element, CSTElement2D) else _QUAD_EDGES
    if not 0 <= local_face_index < len(edges):
        raise ValidationError(
            f"{type(element).__name__} local_face_index must be in [0, {len(edges) - 1}], "
            f"got {local_face_index}."
        )
    return edges[local_face_index]


def _face_node_indices(
    element: Tet4Element3D | Hex8Element3D, local_face_index: int
) -> tuple[int, ...]:
    faces = TET4_FACES if isinstance(element, Tet4Element3D) else HEX8_FACES
    if not 0 <= local_face_index < len(faces):
        raise ValidationError(
            f"{type(element).__name__} local_face_index must be in [0, {len(faces) - 1}], "
            f"got {local_face_index}."
        )
    return faces[local_face_index]


def surface_node_ids(mesh: Mesh, surface: ThermalSurface) -> tuple[int, ...]:
    """Return the global node IDs making up one surface, in the surface's own local order.

    Args:
        mesh: The mesh ``surface.element_id`` belongs to.
        surface: The surface to resolve.

    Returns:
        A tuple of node IDs: length 2 for a CST/Q4 edge, 3 for a TET4
        face, 4 for a HEX8 face.

    Raises:
        EntityNotFoundError: If ``surface.element_id`` is not in ``mesh``.
        ValidationError: If the element is not a supported type, or
            ``local_face_index`` is out of range for it.
    """
    element = _resolve(mesh, surface)
    if isinstance(element, CSTElement2D | QuadElement2D):
        local_indices = _edge_node_pair(element, surface.local_face_index)
    else:
        local_indices = _face_node_indices(element, surface.local_face_index)
    return tuple(element.nodes[i].id for i in local_indices)


def _node_coords_2d(element: CSTElement2D | QuadElement2D, node_id: int) -> tuple[float, float]:
    node = next(node for node in element.nodes if node.id == node_id)
    return node.x, node.y


def _node_coords_3d(element: Tet4Element3D | Hex8Element3D, node_id: int) -> np.ndarray:
    node = next(node for node in element.nodes if node.id == node_id)
    return np.array([node.x, node.y, node.z])


def surface_area(mesh: Mesh, surface: ThermalSurface) -> float:
    """Return one surface's area (3D face) or length-times-thickness (2D edge).

    For a 2D element's edge, "area" means the physical area exposed to
    convection/radiation: edge length times the element's thickness --
    the same quantity :mod:`femtoolkit.continuum.edge` integrates
    tractions over.

    Args:
        mesh: The mesh ``surface.element_id`` belongs to.
        surface: The surface to measure.

    Returns:
        The surface's exposed area, in square meters.

    Raises:
        EntityNotFoundError: If ``surface.element_id`` is not in ``mesh``.
        ValidationError: If the element is not a supported type, or
            ``local_face_index`` is out of range for it.
        DegenerateElementError: If the resulting area is zero or near-zero.
    """
    element = _resolve(mesh, surface)
    node_ids = surface_node_ids(mesh, surface)
    if isinstance(element, CSTElement2D | QuadElement2D):
        xa, ya = _node_coords_2d(element, node_ids[0])
        xb, yb = _node_coords_2d(element, node_ids[1])
        length = float(np.hypot(xb - xa, yb - ya))
        return length * element.thickness
    coords = [_node_coords_3d(element, node_id) for node_id in node_ids]
    if isinstance(element, Tet4Element3D):
        return triangle_face_area(*coords)
    return quad_face_area(np.array(coords))


def surface_conductance_contribution(
    mesh: Mesh, surface: ThermalSurface, coefficient: float
) -> ElementStiffnessContribution:
    """Return ``coefficient * integral(N^T N) dA`` over a surface, assembly-ready.

    This is a convection surface's contribution to the global
    conductivity matrix, ``K_conv`` (see
    :mod:`femtoolkit.thermal.thermal_boundary_conditions`'s module
    docstring for the weak-form derivation).

    Args:
        mesh: The mesh ``surface.element_id`` belongs to.
        surface: The surface to integrate over.
        coefficient: The (positive) scalar multiplying the integral --
            typically a convection coefficient ``h``.

    Returns:
        An :class:`~femtoolkit.analysis.assembly.ElementStiffnessContribution`
        over just this surface's nodes, directly usable with
        :func:`~femtoolkit.analysis.assembly.assemble_global_stiffness`.
    """
    element = _resolve(mesh, surface)
    node_ids = surface_node_ids(mesh, surface)
    dof_keys = tuple((node_id, _TEMPERATURE_DOF) for node_id in node_ids)

    if isinstance(element, CSTElement2D | QuadElement2D):
        node_a = _node_coords_2d(element, node_ids[0])
        node_b = _node_coords_2d(element, node_ids[1])
        matrix = edge_consistent_conductance_matrix(node_a, node_b, coefficient, element.thickness)
    else:
        coords = [_node_coords_3d(element, node_id) for node_id in node_ids]
        if isinstance(element, Tet4Element3D):
            matrix = triangle_face_conductance_matrix(*coords, coefficient)
        else:
            matrix = quad_face_consistent_matrix(np.array(coords), coefficient)
    return ElementStiffnessContribution(dof_keys, matrix)


def surface_uniform_load(
    mesh: Mesh, surface: ThermalSurface, coefficient: float
) -> list[ThermalLoad]:
    """Return ``coefficient * integral(N^T) dA`` over a surface, as nodal thermal loads.

    A surface's contribution to the thermal load vector for a uniform
    (spatially constant) source term -- e.g. ``h * T_infinity`` for
    convection, or an emissivity/Stefan-Boltzmann term for radiation.

    Args:
        mesh: The mesh ``surface.element_id`` belongs to.
        surface: The surface to integrate over.
        coefficient: The scalar multiplying the integral.

    Returns:
        One :class:`~femtoolkit.thermal.thermal_loads.ThermalLoad` per
        surface node.
    """
    element = _resolve(mesh, surface)
    node_ids = surface_node_ids(mesh, surface)

    if isinstance(element, CSTElement2D | QuadElement2D):
        node_a = _node_coords_2d(element, node_ids[0])
        node_b = _node_coords_2d(element, node_ids[1])
        force = edge_equivalent_nodal_force(
            node_a, node_b, traction=(coefficient, 0.0), thickness=element.thickness
        )
        contributions = force[0::2]
    else:
        coords = [_node_coords_3d(element, node_id) for node_id in node_ids]
        if isinstance(element, Tet4Element3D):
            contributions = triangle_face_load_vector(*coords, coefficient)
        else:
            contributions = quad_face_load_vector(np.array(coords), coefficient)
    return [
        ThermalLoad(node_id, float(contribution))
        for node_id, contribution in zip(node_ids, contributions, strict=True)
    ]


def surface_mean_temperature(
    mesh: Mesh, surface: ThermalSurface, temperatures: np.ndarray, dof_map: DOFMap
) -> float:
    """Return the mean nodal temperature over a surface's nodes.

    The representative surface temperature used to evaluate
    temperature-dependent convection/radiation (see
    :mod:`femtoolkit.thermal.thermal_boundary_conditions`'s module
    docstring for why a single representative value, rather than a
    per-Gauss-point nonlinear evaluation, is used).

    Args:
        mesh: The mesh ``surface.element_id`` belongs to.
        surface: The surface to average over.
        temperatures: The global nodal temperature vector.
        dof_map: The DOF map ``temperatures`` is ordered by.

    Returns:
        The mean temperature over the surface's nodes, in kelvin.
    """
    node_ids = surface_node_ids(mesh, surface)
    values = [temperatures[dof_map.global_index(node_id, _TEMPERATURE_DOF)] for node_id in node_ids]
    return float(np.mean(values))
