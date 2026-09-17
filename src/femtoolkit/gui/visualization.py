"""Version 23 3D visualization integration for the GUI (Version 24, spec section 14;
extended in Version 25 for mesh-quality visualization, spec section 14 of that version).

Reuses :class:`~femtoolkit.postprocessing.visualization_3d.viewer.FEAViewer`
directly rather than duplicating any visualization code -- this module
only adds the two things a Streamlit page needs that the viewer itself
does not provide: a plain ``bool`` check for whether PyVista is even
installed (:func:`is_pyvista_available`, reusing the same
``require_pyvista`` guard every other Version 23 module calls), and a
"render the current state to a PNG file" helper
(:func:`render_result_screenshot`) a page can pass straight to
``st.image`` -- since an embedded interactive PyVista widget is out of
scope here, a re-rendered screenshot on every control change is this
version's practical answer to "interactive 3D viewer inside a Streamlit
page."

:func:`render_mesh_quality_screenshot` extends the same pattern to a
bare :class:`~femtoolkit.mesh.mesh.Mesh` with no solved
:class:`~femtoolkit.postprocessing.result_model.SimulationResult` --
mesh quality is a pre-solve concept, so it cannot go through
``FEAViewer`` (which is always built from a solved result). Instead it
reuses Version 23's own mesh-conversion layer directly
(:mod:`femtoolkit.postprocessing.visualization_3d.mesh_converter`),
attaching a per-element quality metric as a scalar field on the exact
same kind of PyVista grid ``FEAViewer`` itself displays -- no second
visualization framework, per this version's spec section 14.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from femtoolkit.postprocessing.result_model import SimulationResult
from femtoolkit.postprocessing.visualization_3d.mesh_converter import require_pyvista

if TYPE_CHECKING:
    from femtoolkit.mesh.mesh import Mesh
    from femtoolkit.mesh.quality.quality_report import MeshQualityReport

QUALITY_METRIC_FIELDS = ("quality", "aspect_ratio", "jacobian_determinant")
"""Per-element :class:`~femtoolkit.mesh.quality.metrics.ElementQuality`/
:class:`~femtoolkit.mesh.quality.metrics.SolidElementQuality` attribute
names that :func:`render_mesh_quality_screenshot` can color a mesh by --
the subset present (and numeric) on both the 2D and 3D quality
dataclasses."""


def is_pyvista_available() -> bool:
    """Whether PyVista (the ``viz3d`` extra) is installed and usable."""
    try:
        require_pyvista()
    except ImportError:
        return False
    return True


def available_scalar_fields(simulation: SimulationResult) -> list[str]:
    """Return every nodal or element field name available on a result's final step."""
    step = simulation.final_step
    return sorted(set(step.nodal_fields) | set(step.element_fields))


def render_result_screenshot(
    simulation: SimulationResult,
    output_path: str | Path,
    *,
    scalar_field: str | None = None,
    deformed: bool = False,
    deformation_scale: float = 1.0,
    show_edges: bool = True,
    camera_view: str = "isometric",
) -> Path:
    """Render one off-screen screenshot of a result via :class:`FEAViewer`.

    Args:
        simulation: The result to visualize.
        output_path: Destination PNG path.
        scalar_field: The scalar field to color the mesh by, or ``None``.
        deformed: Whether to show the deformed geometry.
        deformation_scale: The cosmetic deformation scale factor, used
            only when ``deformed`` is ``True``.
        show_edges: Whether to draw mesh edges.
        camera_view: A named camera preset (see
            :data:`~femtoolkit.postprocessing.visualization_3d.config.CAMERA_VIEWS`).

    Returns:
        The path written to.

    Raises:
        ImportError: If PyVista is not installed.
    """
    from femtoolkit.postprocessing.visualization_3d import FEAViewer, ViewerConfig

    viewer = FEAViewer(simulation, ViewerConfig(off_screen=True))
    try:
        viewer.show_edges(show_edges)
        if scalar_field:
            viewer.set_scalar_field(scalar_field)
        if deformed:
            viewer.show_deformed()
            viewer.set_deformation_scale(deformation_scale)
        else:
            viewer.show_undeformed()
        viewer.set_camera(camera_view)
        viewer.display()

        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        viewer.screenshot(str(destination))
        return destination
    finally:
        viewer.close()


def render_mesh_quality_screenshot(
    mesh: Mesh,
    quality_report: MeshQualityReport,
    output_path: str | Path,
    *,
    metric: str = "quality",
    show_edges: bool = True,
    colormap: str = "RdYlGn",
    camera_view: str = "isometric",
) -> Path:
    """Render one off-screen screenshot of a mesh colored by a shape-quality metric.

    Args:
        mesh: The mesh to visualize.
        quality_report: The mesh's evaluated
            :class:`~femtoolkit.mesh.quality.quality_report.MeshQualityReport`
            (see :class:`~femtoolkit.mesh.quality.evaluator.QualityEvaluator`).
        output_path: Destination PNG path.
        metric: Which per-element metric to color by -- one of
            :data:`QUALITY_METRIC_FIELDS`.
        show_edges: Whether to draw mesh edges.
        colormap: The colormap name (red-to-green by default, so a low
            "quality" value reads as a visual warning).
        camera_view: A named camera preset (see
            :data:`~femtoolkit.postprocessing.visualization_3d.config.CAMERA_VIEWS`).

    Returns:
        The path written to.

    Raises:
        ImportError: If PyVista is not installed.
        ValueError: If ``metric`` is not one of :data:`QUALITY_METRIC_FIELDS`.
    """
    require_pyvista()
    if metric not in QUALITY_METRIC_FIELDS:
        raise ValueError(f"metric must be one of {QUALITY_METRIC_FIELDS}, got {metric!r}.")

    import pyvista as pv

    from femtoolkit.postprocessing.result_model import MeshTopology
    from femtoolkit.postprocessing.visualization_3d.config import CAMERA_VIEWS
    from femtoolkit.postprocessing.visualization_3d.mesh_converter import (
        attach_element_field,
        mesh_topology_to_grid,
    )

    if camera_view not in CAMERA_VIEWS:
        raise ValueError(f"camera_view must be one of {CAMERA_VIEWS}, got {camera_view!r}.")

    topology = MeshTopology.from_mesh(mesh)
    grid = mesh_topology_to_grid(topology)
    values = {
        element_id: getattr(quality, metric)
        for element_id, quality in quality_report.element_qualities.items()
        if getattr(quality, metric) is not None
    }
    attach_element_field(grid, topology, metric, values)

    # "quality" is defined to lie in (0, 1] by construction (see
    # femtoolkit.mesh.quality.metrics); a fixed color range avoids PyVista's
    # default auto-scaling turning sub-1e-10 floating-point noise on a
    # near-perfect mesh into a misleading full red-to-green gradient.
    color_limits = (0.0, 1.0) if metric == "quality" else None

    plotter = pv.Plotter(off_screen=True)
    try:
        plotter.add_mesh(
            grid, scalars=metric, cmap=colormap, show_edges=show_edges, clim=color_limits
        )
        camera_presets = {
            "isometric": plotter.view_isometric,
            "xy": plotter.view_xy,
            "xz": plotter.view_xz,
            "yz": plotter.view_yz,
        }
        camera_presets[camera_view]()
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        plotter.screenshot(str(destination))
        return destination
    finally:
        plotter.close()
