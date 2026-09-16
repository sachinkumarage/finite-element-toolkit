"""Interactive 3D FEA visualization (Version 23).

Built entirely on top of Version 22's post-processing architecture
(:mod:`femtoolkit.postprocessing.result_model`,
:mod:`femtoolkit.postprocessing.field_calculator`): a
:class:`~femtoolkit.postprocessing.result_model.SimulationResult` is
converted into a PyVista mesh
(:mod:`femtoolkit.postprocessing.visualization_3d.mesh_converter`) and
displayed through :class:`~femtoolkit.postprocessing.visualization_3d.viewer.FEAViewer`.

Importing this package never requires PyVista to be installed -- every
solver, material, element, and non-visualization test keeps working
without it (spec section 26). Only actually *using* a function here (or
constructing an :class:`~femtoolkit.postprocessing.visualization_3d.viewer.FEAViewer`)
raises a clear :class:`ImportError` if PyVista is missing, via
:func:`~femtoolkit.postprocessing.visualization_3d.mesh_converter.require_pyvista`.
Install with ``pip install "femtoolkit[viz3d]"``.
"""

from femtoolkit.postprocessing.visualization_3d.boundary_conditions import (
    node_markers,
    surface_markers,
)
from femtoolkit.postprocessing.visualization_3d.config import CAMERA_VIEWS, ViewerConfig
from femtoolkit.postprocessing.visualization_3d.mesh_converter import (
    ELEMENT_CELL_TYPES,
    attach_element_field,
    attach_nodal_field,
    mesh_topology_to_grid,
    require_pyvista,
    result_step_to_grid,
)
from femtoolkit.postprocessing.visualization_3d.viewer import FEAViewer

__all__ = [
    "CAMERA_VIEWS",
    "ELEMENT_CELL_TYPES",
    "FEAViewer",
    "ViewerConfig",
    "attach_element_field",
    "attach_nodal_field",
    "mesh_topology_to_grid",
    "node_markers",
    "require_pyvista",
    "result_step_to_grid",
    "surface_markers",
]
