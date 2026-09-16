"""Converting a Version 22 result into a PyVista mesh (Version 23).

PyVista represents a finite element mesh as an
:class:`pyvista.UnstructuredGrid`: a flat array of node coordinates, a
flat "cell" array (for each cell, its node count followed by that many
node indices), and a parallel array naming each cell's VTK cell type.
This module is the **only** place in the toolkit that builds one --
everywhere else, geometry is described by this project's own
:class:`~femtoolkit.postprocessing.result_model.MeshTopology` (itself
built from a :class:`~femtoolkit.mesh.mesh.Mesh` or a raw element list,
see :mod:`femtoolkit.postprocessing.result_model`), keeping the
PyVista/VTK dependency confined to this visualization layer.

**Node ordering: verified, not assumed.** A finite element's node order
and VTK's expected cell-node order are not automatically the same thing
-- a mismatch silently produces an inverted or self-intersecting cell,
which is exactly the failure mode this module's tests
(``tests/test_visualization_mesh_converter.py``) guard against by
checking a converted cell's *volume/area* against the closed-form
expected value, not just that conversion "succeeds" without an
exception. For every element type this toolkit supports, the node order
already matches VTK's directly, with no reordering needed:

* **TET4** (``Tet4Element3D``) -> ``VTK_TETRA``: both are simply "four
  points, in the order supplied" (this project's TET4 explicitly
  accepts either winding -- see :mod:`femtoolkit.mesh.tet4_element`'s
  module docstring -- and VTK's tetrahedron cell has no orientation
  requirement for its volume to compute correctly either).
* **HEX8** (``Hex8Element3D``) -> ``VTK_HEXAHEDRON``: both use the
  identical convention -- bottom face counter-clockwise, then the top
  face directly above it in the same winding (see
  :data:`~femtoolkit.continuum.shape_functions._HEX8_NATURAL_COORDS`,
  and VTK's own hexahedron cell documentation).
* **CST** (``CSTElement2D``) -> ``VTK_TRIANGLE`` and **Q4**
  (``QuadElement2D``) -> ``VTK_QUAD``: both are the direct node order
  already used throughout this project's 2D continuum modules.
* **Bar** (``BarElement``) -> ``VTK_LINE``: a 2-node line, included for
  completeness (spec section 5's "existing 2D elements where
  meaningful" extends naturally to this simplest case).

**Fields become VTK arrays, not new data.** A step's nodal fields
become VTK **point data**; its element fields become VTK **cell data**
-- straight copies from the already-computed
:class:`~femtoolkit.postprocessing.result_model.ResultStep`, in
:attr:`~femtoolkit.postprocessing.result_model.MeshTopology.node_ids`/
:attr:`~femtoolkit.postprocessing.result_model.MeshTopology.element_ids`
order. A scalar field becomes a length-``N`` array; a vector field is
padded (or truncated) to exactly 3 components, since VTK vector arrays
are always 3D -- the same zero-padding convention
:func:`~femtoolkit.postprocessing.field_calculator.deformed_coordinates`
already uses for a 2D displacement field.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from femtoolkit.exceptions import ValidationError
from femtoolkit.postprocessing.result_model import FieldValue, MeshTopology, ResultStep

try:
    import pyvista as pv
except ImportError:  # pragma: no cover - exercised via the no-PyVista regression check
    pv = None

if TYPE_CHECKING:
    import pyvista

ELEMENT_CELL_TYPES: dict[str, int] = {
    "BarElement": 3,  # VTK_LINE
    "CSTElement2D": 5,  # VTK_TRIANGLE
    "QuadElement2D": 9,  # VTK_QUAD
    "Tet4Element3D": 10,  # VTK_TETRA
    "Hex8Element3D": 12,  # VTK_HEXAHEDRON
}
"""Maps this project's element class names to their VTK cell type code.

The numeric values are the standard VTK cell type IDs (also available as
``pyvista.CellType.LINE``/``.TRIANGLE``/``.QUAD``/``.TETRA``/``.HEXAHEDRON``
when PyVista is installed) -- given directly here so this mapping is
inspectable without importing PyVista."""


def require_pyvista() -> None:
    """Raise a clear, actionable error if PyVista is not installed.

    Every public entry point in :mod:`femtoolkit.postprocessing.visualization_3d`
    calls this first, so importing this subpackage never fails outright
    (keeping the rest of the toolkit -- materials, elements, solvers,
    every non-visualization test -- usable without PyVista installed),
    but *using* it does, with a message that says exactly what to do.

    Raises:
        ImportError: If PyVista is not installed.
    """
    if pv is None:
        raise ImportError(
            "3D visualization requires PyVista, which is not installed. "
            'Install it with: pip install "femtoolkit[viz3d]" (or `pip install pyvista`).'
        )


def mesh_topology_to_grid(topology: MeshTopology) -> pyvista.UnstructuredGrid:
    """Convert a mesh topology's geometry into an empty (field-free) PyVista grid.

    Args:
        topology: The mesh topology to convert.

    Returns:
        A :class:`pyvista.UnstructuredGrid` with the same nodes and
        elements as ``topology``, and no field data attached yet (see
        :func:`attach_nodal_field`/:func:`attach_element_field`, or
        :func:`result_step_to_grid` for a ready-to-plot grid).

    Raises:
        ImportError: If PyVista is not installed.
        ValidationError: If ``topology`` contains an element type this
            module does not support (see :data:`ELEMENT_CELL_TYPES`).
    """
    require_pyvista()

    node_index = {node_id: index for index, node_id in enumerate(topology.node_ids)}
    points = topology.node_coordinate_array()

    cells: list[int] = []
    cell_types: list[int] = []
    for element_id in topology.element_ids:
        element_type_name = topology.element_types[element_id]
        cell_type = ELEMENT_CELL_TYPES.get(element_type_name)
        if cell_type is None:
            raise ValidationError(
                f"Element {element_id} has type {element_type_name!r}, which is not "
                f"supported for 3D visualization. Supported types: "
                f"{sorted(ELEMENT_CELL_TYPES)}."
            )
        connectivity = topology.element_connectivity[element_id]
        indices = [node_index[node_id] for node_id in connectivity]
        cells.append(len(indices))
        cells.extend(indices)
        cell_types.append(cell_type)

    return pv.UnstructuredGrid(np.array(cells), np.array(cell_types), points)


def _pad_to_three_components(value: FieldValue) -> np.ndarray:
    array = np.atleast_1d(np.asarray(value, dtype=float))
    if array.size >= 3:
        return array[:3]
    padded = np.zeros(3)
    padded[: array.size] = array
    return padded


def attach_nodal_field(
    grid: pyvista.UnstructuredGrid,
    topology: MeshTopology,
    name: str,
    values: dict[int, FieldValue],
) -> None:
    """Attach one nodal field to a grid as VTK point data, in-place.

    Args:
        grid: The :class:`pyvista.UnstructuredGrid` to attach the field to
            (typically from :func:`mesh_topology_to_grid`).
        topology: The mesh topology ``grid`` was built from (defines the
            node ordering the resulting array must follow).
        name: The field's name (becomes the VTK array name).
        values: Maps a subset of ``topology.node_ids`` to their value.
            Nodes not present get ``NaN`` (scalar fields) or zero
            (vector fields).
    """
    sample = next(iter(values.values()))
    is_vector = np.asarray(sample).size > 1

    if is_vector:
        array = np.array(
            [
                _pad_to_three_components(values[node_id]) if node_id in values else np.zeros(3)
                for node_id in topology.node_ids
            ]
        )
    else:
        array = np.array(
            [
                float(values[node_id]) if node_id in values else np.nan
                for node_id in topology.node_ids
            ]
        )
    grid.point_data[name] = array


def attach_element_field(
    grid: pyvista.UnstructuredGrid,
    topology: MeshTopology,
    name: str,
    values: dict[int, FieldValue],
) -> None:
    """Attach one element field to a grid as VTK cell data, in-place.

    Args:
        grid: The :class:`pyvista.UnstructuredGrid` to attach the field to.
        topology: The mesh topology ``grid`` was built from (defines the
            element ordering the resulting array must follow).
        name: The field's name (becomes the VTK array name).
        values: Maps a subset of ``topology.element_ids`` to their value.
            Elements not present get ``NaN`` (scalar fields) or zero
            (vector fields).
    """
    sample = next(iter(values.values()))
    is_vector = np.asarray(sample).size > 1

    if is_vector:
        array = np.array(
            [
                _pad_to_three_components(values[element_id])
                if element_id in values
                else np.zeros(3)
                for element_id in topology.element_ids
            ]
        )
    else:
        array = np.array(
            [
                float(values[element_id]) if element_id in values else np.nan
                for element_id in topology.element_ids
            ]
        )
    grid.cell_data[name] = array


def result_step_to_grid(topology: MeshTopology, step: ResultStep) -> pyvista.UnstructuredGrid:
    """Convert one result step into a PyVista grid with every field attached.

    Args:
        topology: The mesh topology.
        step: The step whose nodal/element fields should be attached.

    Returns:
        A :class:`pyvista.UnstructuredGrid` with every field in ``step``
        attached as point data (nodal fields) or cell data (element fields).

    Raises:
        ImportError: If PyVista is not installed.
        ValidationError: If ``topology`` contains an unsupported element type.
    """
    grid = mesh_topology_to_grid(topology)
    for name, values in step.nodal_fields.items():
        attach_nodal_field(grid, topology, name, values)
    for name, values in step.element_fields.items():
        attach_element_field(grid, topology, name, values)
    return grid
