"""Version 23 3D visualization integration for the GUI (Version 24, spec section 14).

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
"""

from __future__ import annotations

from pathlib import Path

from femtoolkit.postprocessing.result_model import SimulationResult
from femtoolkit.postprocessing.visualization_3d.mesh_converter import require_pyvista


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
