"""Simple visual markers for boundary conditions and loads (Version 23).

Spec section 17 is explicit about scope: "simple visual markers... the
goal is engineering clarity, not a full CAD-style annotation system."
Every boundary condition and load type this toolkit has -- prescribed
displacement (:class:`~femtoolkit.analysis.boundary_conditions.BoundaryCondition`),
prescribed temperature/heat flux
(:class:`~femtoolkit.thermal.thermal_boundary_conditions.PrescribedTemperature`/
:class:`~femtoolkit.thermal.thermal_boundary_conditions.PrescribedHeatFlux`),
and an applied mechanical load
(:class:`~femtoolkit.analysis.loads.NodalLoad`) -- shares the same
essential shape for visualization purposes: *a marker at the node it
applies to*. Rather than one bespoke function per type, this module
provides one generic :func:`node_markers` (build a point cloud at a set
of node IDs, whatever kind of condition they came from) and one
generic :func:`surface_markers` for the two boundary conditions that
apply over a *surface* rather than a single node -- convection and
radiation (:class:`~femtoolkit.thermal.thermal_surfaces.ThermalSurface`).
A caller renders either result with
:meth:`~femtoolkit.postprocessing.visualization_3d.viewer.FEAViewer.add_mesh`,
choosing whatever glyph style (points, spheres, arrows) suits the
condition being shown.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

import numpy as np

from femtoolkit.postprocessing.result_model import MeshTopology
from femtoolkit.postprocessing.visualization_3d.mesh_converter import require_pyvista

try:
    import pyvista as pv
except ImportError:  # pragma: no cover - exercised via the no-PyVista regression check
    pv = None

if TYPE_CHECKING:
    import pyvista

    from femtoolkit.mesh.mesh import Mesh
    from femtoolkit.thermal.thermal_surfaces import ThermalSurface


def node_markers(topology: MeshTopology, node_ids: Iterable[int]) -> pyvista.PolyData:
    """Build a point-cloud marker mesh at a set of nodes.

    Works for any node-based boundary condition or load -- a prescribed
    displacement, a prescribed temperature, a prescribed heat flux, or
    an applied nodal load -- since each is, for visualization purposes,
    just "a marker at this node."

    Args:
        topology: The mesh topology providing node coordinates.
        node_ids: The nodes to mark (e.g. ``[bc.node_id for bc in
            boundary_conditions]``). Duplicates are kept as separate
            points (harmless for rendering).

    Returns:
        A :class:`pyvista.PolyData` point cloud, one point per node ID.

    Raises:
        ImportError: If PyVista is not installed.
    """
    require_pyvista()
    points = np.array([topology.node_coordinates[node_id] for node_id in node_ids], dtype=float)
    if points.size == 0:
        points = points.reshape((0, 3))
    return pv.PolyData(points)


def surface_markers(mesh: Mesh, surfaces: Iterable[ThermalSurface]) -> pyvista.PolyData:
    """Build a small polygon-patch marker mesh for a set of convection/radiation surfaces.

    Each :class:`~femtoolkit.thermal.thermal_surfaces.ThermalSurface`
    (a CST/Q4 edge or a TET4/HEX8 face) becomes one polygon, built from
    that face's own node coordinates -- reusing
    :func:`~femtoolkit.thermal.thermal_surfaces.surface_node_ids` for the
    node lookup rather than re-deriving element face geometry.

    Args:
        mesh: The mesh the surfaces belong to.
        surfaces: The convection or radiation surfaces to mark.

    Returns:
        A :class:`pyvista.PolyData` with one polygon face per surface.

    Raises:
        ImportError: If PyVista is not installed.
    """
    require_pyvista()
    from femtoolkit.thermal.thermal_surfaces import surface_node_ids

    all_points: list[tuple[float, float, float]] = []
    faces: list[int] = []
    offset = 0
    for surface in surfaces:
        node_ids = surface_node_ids(mesh, surface)
        for node_id in node_ids:
            node = mesh.get_node(node_id)
            all_points.append((node.x, node.y, node.z))
        count = len(node_ids)
        faces.append(count)
        faces.extend(range(offset, offset + count))
        offset += count

    points_array = np.array(all_points, dtype=float) if all_points else np.zeros((0, 3))
    return pv.PolyData(points_array, faces=np.array(faces, dtype=int) if faces else None)
