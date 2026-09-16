"""The interactive 3D FEA viewer (Version 23).

:class:`FEAViewer` is the reusable viewer class this version's brief
asks for: a thin, stateful wrapper around a PyVista
:class:`~pyvista.Plotter` that always gets its geometry and field data
from a Version 22 :class:`~femtoolkit.postprocessing.result_model.SimulationResult`
(via :mod:`femtoolkit.postprocessing.visualization_3d.mesh_converter`),
never from solver internals -- exactly the separation spec section 25
requires ("The visualization layer must depend on result data, not on
internal solver implementation").

**A declarative "set state, then display" pattern.** Every ``set_*``/
``show_*`` method (``set_scalar_field``, ``set_vector_field``,
``show_edges``, ``show_deformed``, ``set_deformation_scale``,
``set_step``, ...) only updates the viewer's own internal state; it is
:meth:`display` that turns the current state into an actual rendered
mesh. This mirrors how a real interactive session naturally proceeds
(configure what you want to see, then look at it) and means every
setter is trivially testable on its own, without needing a real
rendering backend to verify it took effect.

**No field calculation happens here.** Selecting a scalar/vector field
by name only ever *looks up* an already-computed array on the mesh
converted from the result (see
:mod:`femtoolkit.postprocessing.visualization_3d.mesh_converter`) --
von Mises stress, equivalent strain, or a vector field's magnitude are
never computed inside this class. A result carrying those fields is
expected to have already been passed through
:func:`~femtoolkit.postprocessing.field_calculator.with_derived_fields`
(Version 22) before being handed to :class:`FEAViewer` -- the same
"Field Calculator computes, Post Processor and Visualization only
consume" separation Version 22 established, now extended to 3D.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from femtoolkit.exceptions import ValidationError
from femtoolkit.postprocessing.field_calculator import deformed_coordinates
from femtoolkit.postprocessing.result_model import SimulationResult
from femtoolkit.postprocessing.visualization_3d.config import ViewerConfig
from femtoolkit.postprocessing.visualization_3d.mesh_converter import (
    require_pyvista,
    result_step_to_grid,
)

try:
    import pyvista as pv
except ImportError:  # pragma: no cover - exercised via the no-PyVista regression check
    pv = None

if TYPE_CHECKING:
    import pyvista

_PRIMARY_MESH_NAME = "fea_mesh"
_VECTOR_GLYPH_NAME = "fea_vectors"


class FEAViewer:
    """An interactive 3D viewer over a Version 22
    :class:`~femtoolkit.postprocessing.result_model.SimulationResult`.

    Attributes:
        result: The wrapped simulation result.
        config: The active :class:`~femtoolkit.postprocessing.visualization_3d.config.ViewerConfig`.

    Raises:
        ImportError: If PyVista is not installed.

    Example:
        >>> viewer = FEAViewer(result, ViewerConfig(off_screen=True))
        >>> viewer.set_scalar_field("temperature")
        >>> viewer.display()
        >>> viewer.screenshot("temperature.png")
        >>> viewer.close()
    """

    def __init__(self, result: SimulationResult, config: ViewerConfig | None = None) -> None:
        """Create a viewer over ``result``.

        Args:
            result: The simulation result to visualize. For a result
                whose scalar fields include derived quantities (von
                Mises stress, equivalent strain, a vector field's
                magnitude), pass it through
                :func:`~femtoolkit.postprocessing.field_calculator.with_derived_fields`
                first -- this class never computes them itself.
            config: Visualization settings. Defaults to
                ``ViewerConfig()`` (an interactive, on-screen window).

        Raises:
            ImportError: If PyVista is not installed.
        """
        require_pyvista()
        self.result = result
        self.config = config or ViewerConfig()

        self._plotter = pv.Plotter(
            off_screen=self.config.off_screen,
            window_size=list(self.config.window_size),
            title=self.config.window_title,
        )
        self._plotter.set_background(self.config.background_color)

        self._current_step = result.num_steps - 1
        self._scalar_field: str | None = None
        self._scalar_component: int | None = None
        self._vector_field: str | None = None
        self._show_edges = self.config.show_edges
        self._show_colorbar = self.config.show_colorbar
        self._deformed = False
        self._displacement_field = "displacement"
        self._deformation_scale = self.config.deformation_scale
        self._actors: dict[str, Any] = {}
        self._current_grid: pyvista.UnstructuredGrid | None = None

        self.set_camera(self.config.camera_view)

    # -- Mesh management (spec section 6) -----------------------------------

    def add_mesh(self, name: str, mesh: pyvista.DataSet, **kwargs: Any) -> None:
        """Add (or replace) a named mesh in the scene.

        Args:
            name: A unique name for this mesh's actor. Adding a mesh
                under a name that already exists replaces it.
            mesh: Any PyVista mesh (an
                :class:`~pyvista.UnstructuredGrid` or :class:`~pyvista.PolyData`).
            **kwargs: Forwarded to :meth:`pyvista.Plotter.add_mesh`
                (``scalars``, ``color``, ``show_edges``, ``cmap``, ...).
        """
        actor = self._plotter.add_mesh(mesh, name=name, **kwargs)
        self._actors[name] = actor

    def remove_mesh(self, name: str) -> None:
        """Remove a named mesh from the scene, if present."""
        self._plotter.remove_actor(name)
        self._actors.pop(name, None)

    # -- Field/step selection (spec sections 6, 14, 15) ----------------------

    def set_scalar_field(self, field_name: str | None, component: int | None = None) -> None:
        """Select which field :meth:`display` colors the mesh by.

        Args:
            field_name: A nodal or element field name present in the
                result (e.g. ``"temperature"``, ``"von_mises_stress"``),
                or ``None`` to show the mesh with no scalar coloring.
            component: For a vector field (e.g. ``"displacement"``),
                which component to show (``0``/``1``/``2`` for X/Y/Z).
                Required for a vector field; ignored for a scalar one.
        """
        self._scalar_field = field_name
        self._scalar_component = component

    def set_vector_field(self, field_name: str | None) -> None:
        """Select which vector field :meth:`display` draws as glyphs (arrows).

        Args:
            field_name: A vector-valued nodal or element field name
                (e.g. ``"heat_flux"``, ``"displacement"``), or ``None``
                to hide vector glyphs.
        """
        self._vector_field = field_name

    def set_step(self, index: int) -> None:
        """Select which result step :meth:`display` shows.

        Args:
            index: The step index (supports negative indexing, e.g.
                ``-1`` for the last/final step).

        Raises:
            ValidationError: If ``index`` is out of range.
        """
        if not -self.result.num_steps <= index < self.result.num_steps:
            raise ValidationError(
                f"step index {index} is out of range for a result with "
                f"{self.result.num_steps} steps."
            )
        self._current_step = index

    @property
    def current_step(self) -> int:
        """The currently selected step index (always resolved to a non-negative index)."""
        return self._current_step % self.result.num_steps

    def step_forward(self) -> None:
        """Advance to the next step (wraps to the first step after the last)."""
        self.set_step((self.current_step + 1) % self.result.num_steps)

    def step_backward(self) -> None:
        """Go back to the previous step (wraps to the last step before the first)."""
        self.set_step((self.current_step - 1) % self.result.num_steps)

    # -- Display appearance (spec sections 6, 7, 10, 11, 20) -----------------

    def show_edges(self, visible: bool = True) -> None:
        """Set whether element edges are drawn on top of the mesh (default: on)."""
        self._show_edges = visible

    def show_colorbar(self, visible: bool = True) -> None:
        """Set whether a colorbar is shown alongside the active scalar field."""
        self._show_colorbar = visible

    def show_undeformed(self) -> None:
        """Display the mesh at its original (undeformed) coordinates."""
        self._deformed = False

    def show_deformed(self, displacement_field: str = "displacement") -> None:
        """Display the mesh at its deformed coordinates, ``x + s*u``.

        Args:
            displacement_field: The nodal vector field to use as the
                displacement ``u`` (default: ``"displacement"``).
        """
        self._deformed = True
        self._displacement_field = displacement_field

    def set_deformation_scale(self, scale: float) -> None:
        """Set the visualization scale factor ``s`` applied when showing the deformed mesh.

        Args:
            scale: The scale factor. Purely cosmetic -- see
                :func:`~femtoolkit.postprocessing.field_calculator.deformed_coordinates`.

        Raises:
            ValidationError: If ``scale`` is not finite.
        """
        if not np.isfinite(scale):
            raise ValidationError(f"deformation scale must be finite, got {scale}.")
        self._deformation_scale = scale

    def set_camera(self, view: str | Sequence[float]) -> None:
        """Set the camera to a named preset or an explicit PyVista camera position.

        Args:
            view: One of
                :data:`~femtoolkit.postprocessing.visualization_3d.config.CAMERA_VIEWS`
                (``"isometric"``, ``"xy"``, ``"xz"``, ``"yz"``), or an
                explicit PyVista ``camera_position`` value.
        """
        if isinstance(view, str):
            preset = {
                "isometric": self._plotter.view_isometric,
                "xy": self._plotter.view_xy,
                "xz": self._plotter.view_xz,
                "yz": self._plotter.view_yz,
            }.get(view)
            if preset is None:
                raise ValidationError(f"Unknown camera view {view!r}.")
            preset()
        else:
            self._plotter.camera_position = view

    def add_labels(
        self, points: Sequence[Sequence[float]], labels: Sequence[str], name: str = "labels"
    ) -> None:
        """Add text labels at a set of 3D points.

        Args:
            points: One ``(x, y, z)`` position per label.
            labels: One text string per point, same length as ``points``.
            name: A unique name for this label actor.
        """
        self._plotter.add_point_labels(points, labels, name=name)

    # -- Rendering the current state (spec sections 6, 8, 9, 11, 12, 13) -----

    def _build_current_grid(self) -> pyvista.UnstructuredGrid:
        step = self.result.step(self._current_step)
        grid = result_step_to_grid(self.result.topology, step)

        if self._deformed:
            if self._displacement_field not in step.nodal_fields:
                raise ValidationError(
                    f"No {self._displacement_field!r} nodal field at step "
                    f"{self._current_step} to deform the mesh with."
                )
            displacement_field = step.nodal_field(self._displacement_field)
            deformed = deformed_coordinates(
                self.result.topology, displacement_field, self._deformation_scale
            )
            grid.points = np.array(
                [deformed[node_id] for node_id in self.result.topology.node_ids]
            )
        return grid

    def _resolve_scalar_name(self, grid: pyvista.UnstructuredGrid) -> str | None:
        if self._scalar_field is None:
            return None
        if self._scalar_field in grid.point_data:
            array = grid.point_data[self._scalar_field]
            is_point = True
        elif self._scalar_field in grid.cell_data:
            array = grid.cell_data[self._scalar_field]
            is_point = False
        else:
            raise ValidationError(
                f"No field named {self._scalar_field!r} at step {self._current_step}."
            )

        if array.ndim == 1:
            return self._scalar_field

        if self._scalar_component is None:
            raise ValidationError(
                f"{self._scalar_field!r} is a vector field; call set_scalar_field(..., "
                "component=0/1/2), or pass a result already enriched with "
                "femtoolkit.postprocessing.field_calculator.with_derived_fields for a "
                "magnitude field."
            )
        component_name = f"{self._scalar_field}[{self._scalar_component}]"
        values = array[:, self._scalar_component]
        (grid.point_data if is_point else grid.cell_data)[component_name] = values
        return component_name

    def _add_vector_glyphs(self, grid: pyvista.UnstructuredGrid) -> None:
        if self._vector_field is None:
            self.remove_mesh(_VECTOR_GLYPH_NAME)
            return

        if self._vector_field in grid.point_data:
            source = grid
        elif self._vector_field in grid.cell_data:
            source = grid.cell_centers()
            source.point_data[self._vector_field] = grid.cell_data[self._vector_field]
        else:
            raise ValidationError(
                f"No vector field named {self._vector_field!r} at step {self._current_step}."
            )

        glyphs = source.glyph(
            orient=self._vector_field, scale=self._vector_field, factor=self.config.vector_scale
        )
        self.add_mesh(_VECTOR_GLYPH_NAME, glyphs, color="black", show_scalar_bar=False)

    def display(self) -> None:
        """(Re)build and render the mesh from the viewer's current state.

        This is the single method every ``set_*``/``show_*`` call
        eventually feeds into: it reads the currently selected step,
        scalar field, vector field, and deformation state, builds the
        corresponding PyVista grid (via
        :mod:`femtoolkit.postprocessing.visualization_3d.mesh_converter`),
        and adds/updates the scene's primary mesh actor (plus, if a
        vector field is selected, a glyph actor).

        Raises:
            ValidationError: If the selected scalar/vector field is not
                present at the current step, or a vector scalar field
                has no component selected.
        """
        grid = self._build_current_grid()
        self._current_grid = grid
        scalars = self._resolve_scalar_name(grid)

        self.add_mesh(
            _PRIMARY_MESH_NAME,
            grid,
            scalars=scalars,
            cmap=self.config.colormap,
            show_edges=self._show_edges,
            show_scalar_bar=self._show_colorbar and scalars is not None,
        )
        self._add_vector_glyphs(grid)

    # -- Inspection: clipping, slicing, thresholding (spec section 18) ------

    def clip(
        self, normal: Sequence[float] = (1.0, 0.0, 0.0), origin: Sequence[float] | None = None
    ) -> None:
        """Clip the currently displayed mesh with a plane, showing only one side.

        Args:
            normal: The clipping plane's normal vector.
            origin: A point on the clipping plane. Defaults to the mesh's center.
        """
        grid = self._require_current_grid()
        clipped = grid.clip(normal=normal, origin=origin)
        self.add_mesh(
            _PRIMARY_MESH_NAME,
            clipped,
            scalars=self._resolve_scalar_name(clipped),
            cmap=self.config.colormap,
            show_edges=self._show_edges,
            show_scalar_bar=self._show_colorbar,
        )

    def slice(
        self, normal: Sequence[float] = (1.0, 0.0, 0.0), origin: Sequence[float] | None = None
    ) -> None:
        """Replace the currently displayed mesh with a planar slice through it.

        Args:
            normal: The slicing plane's normal vector.
            origin: A point on the slicing plane. Defaults to the mesh's center.
        """
        grid = self._require_current_grid()
        sliced = grid.slice(normal=normal, origin=origin)
        self.add_mesh(
            _PRIMARY_MESH_NAME,
            sliced,
            scalars=self._resolve_scalar_name(sliced),
            cmap=self.config.colormap,
            show_scalar_bar=self._show_colorbar,
        )

    def threshold(self, field_name: str, value_range: tuple[float, float]) -> None:
        """Replace the currently displayed mesh with only the cells/points within a value range.

        Args:
            field_name: The field to threshold on.
            value_range: ``(minimum, maximum)`` inclusive range to keep.
        """
        grid = self._require_current_grid()
        thresholded = grid.threshold(value_range, scalars=field_name)
        self.add_mesh(
            _PRIMARY_MESH_NAME,
            thresholded,
            scalars=field_name,
            cmap=self.config.colormap,
            show_scalar_bar=self._show_colorbar,
        )

    def _require_current_grid(self) -> pyvista.UnstructuredGrid:
        if self._current_grid is None:
            self.display()
        return self._current_grid

    # -- Probing (spec section 19) -------------------------------------------

    def probe_nearest_node(self, point: Sequence[float], field_name: str) -> tuple[int, Any]:
        """Find the node nearest a 3D point and return its ID and field value.

        Uses plain nearest-node lookup (no spatial interpolation -- see
        the module docstring's scope note).

        Args:
            point: The ``(x, y, z)`` query point.
            field_name: The nodal field to look up.

        Returns:
            ``(node_id, value)`` for the nearest node, at the current step.
        """
        topology = self.result.topology
        coordinates = topology.node_coordinate_array()
        distances = np.linalg.norm(coordinates - np.asarray(point, dtype=float), axis=1)
        nearest_index = int(np.argmin(distances))
        node_id = topology.node_ids[nearest_index]
        value = self.result.step(self._current_step).nodal_value(field_name, node_id)
        return node_id, value

    def probe_element(self, element_id: int, field_name: str) -> Any:
        """Return one element's value for a field at the current step.

        Args:
            element_id: The element to look up.
            field_name: The element field to look up.

        Returns:
            The field value at ``element_id``, at the current step.
        """
        return self.result.step(self._current_step).element_value(field_name, element_id)

    # -- Animation (spec section 16) -----------------------------------------

    def animate(
        self,
        field_name: str | None = None,
        steps: Sequence[int] | None = None,
        filename: str | None = None,
        fps: float = 5.0,
    ) -> None:
        """Step through a sequence of result steps, updating the display at each one.

        Args:
            field_name: If given, selected as the scalar field for every frame.
            steps: Which step indices to visit, in order. Defaults to
                every step, in order.
            filename: If given, a GIF is recorded to this path (one
                frame per step) instead of (or as well as) just updating
                the live display.
            fps: Playback speed, in frames per second, for the recorded GIF.

        Raises:
            ValidationError: If ``fps`` is not positive and finite.
        """
        if not np.isfinite(fps) or fps <= 0:
            raise ValidationError(f"fps must be positive, got {fps}.")
        if field_name is not None:
            self.set_scalar_field(field_name)

        step_indices = steps if steps is not None else range(self.result.num_steps)

        recording = filename is not None
        if recording:
            self._plotter.open_gif(filename, fps=fps)

        for index in step_indices:
            self.set_step(index)
            self.display()
            if recording:
                self._plotter.write_frame()

        if recording:
            self._plotter.close()

    # -- Export (spec section 21) --------------------------------------------

    def screenshot(self, path: str) -> None:
        """Save a screenshot of the current scene to an image file.

        Args:
            path: Destination file path (format inferred from extension,
                e.g. ``.png``).
        """
        self._plotter.screenshot(path)

    def export_mesh(self, path: str) -> None:
        """Save the currently displayed mesh (geometry and every attached field) to a file.

        Args:
            path: Destination file path (e.g. ``.vtu``), reopenable in
                PyVista or ParaView -- the "saved visualization state"
                this version's brief asks for.
        """
        grid = self._require_current_grid()
        grid.save(path)

    # -- Lifecycle -------------------------------------------------------------

    def show(self) -> None:
        """Open the interactive window (or, if ``config.off_screen``, just render off-screen)."""
        self._plotter.show()

    def close(self) -> None:
        """Close the underlying plotter and release its resources."""
        self._plotter.close()
