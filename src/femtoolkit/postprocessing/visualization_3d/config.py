"""Visualization settings for
:class:`~femtoolkit.postprocessing.visualization_3d.viewer.FEAViewer` (Version 23).

Every visual setting the viewer exposes (window title, background,
mesh/edge visibility, colormap, colorbar visibility, deformation/vector
scale, camera preset, window size, off-screen rendering) lives in one
place, :class:`ViewerConfig`, rather than being hard-coded at each call
site inside :mod:`femtoolkit.postprocessing.visualization_3d.viewer` --
the same "one configuration object, not scattered constants" pattern
this project already uses for solver settings
(:class:`~femtoolkit.analysis.nonlinear_analysis.NonlinearSolverSettings`).
This module has no PyVista dependency of its own (it is plain data), so
it can be imported and validated even where PyVista is not installed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from femtoolkit.exceptions import ValidationError

CAMERA_VIEWS: tuple[str, ...] = ("isometric", "xy", "xz", "yz")
"""The named camera view presets :attr:`ViewerConfig.camera_view` accepts."""


@dataclass(frozen=True)
class ViewerConfig:
    """Configuration for an :class:`~femtoolkit.postprocessing.visualization_3d.viewer.FEAViewer`.

    Attributes:
        window_title: The plotter window's title.
        background_color: PyVista/VTK-recognized color name or hex
            string for the scene background.
        show_edges: Whether element edges are drawn on top of the
            surface/solid representation by default.
        colormap: The Matplotlib/VTK colormap name used for scalar
            field visualization.
        show_colorbar: Whether a colorbar is shown alongside a scalar field.
        deformation_scale: The default visualization scale factor ``s``
            applied to a displacement field (``x_deformed = x + s*u``).
            Purely cosmetic -- see
            :func:`~femtoolkit.postprocessing.field_calculator.deformed_coordinates`.
            Must be finite (may be negative, e.g. to visualize a mode
            shape's opposite phase).
        vector_scale: The default glyph scale factor for vector field
            visualization (heat flux, displacement arrows). Must be positive.
        window_size: The plotter window's ``(width, height)`` in pixels.
            Both must be positive.
        camera_view: One of :data:`CAMERA_VIEWS` -- the initial camera
            preset.
        off_screen: If ``True``, the viewer renders without opening an
            interactive window (for automated scripts, examples, and
            tests). If ``False`` (the default), :meth:`FEAViewer.show`
            opens a normal interactive window with camera controls.

    Raises:
        ValidationError: If any field fails validation.

    Example:
        >>> config = ViewerConfig(colormap="plasma", deformation_scale=50.0, off_screen=True)
    """

    window_title: str = "Finite Element Toolkit -- 3D Viewer"
    background_color: str = "white"
    show_edges: bool = True
    colormap: str = "viridis"
    show_colorbar: bool = True
    deformation_scale: float = 1.0
    vector_scale: float = 1.0
    window_size: tuple[int, int] = field(default=(1024, 768))
    camera_view: str = "isometric"
    off_screen: bool = False

    def __post_init__(self) -> None:
        """Validate every setting immediately after construction.

        Raises:
            ValidationError: If ``deformation_scale`` is not finite,
                ``vector_scale`` is not positive and finite,
                ``window_size`` is not a pair of positive integers, or
                ``camera_view`` is not one of :data:`CAMERA_VIEWS`.
        """
        if not math.isfinite(self.deformation_scale):
            raise ValidationError(
                f"ViewerConfig deformation_scale must be finite, got {self.deformation_scale}."
            )
        if not math.isfinite(self.vector_scale) or self.vector_scale <= 0:
            raise ValidationError(
                f"ViewerConfig vector_scale must be positive, got {self.vector_scale}."
            )
        width, height = self.window_size
        if width <= 0 or height <= 0:
            raise ValidationError(
                f"ViewerConfig window_size must be a pair of positive integers, got "
                f"{self.window_size}."
            )
        if self.camera_view not in CAMERA_VIEWS:
            raise ValidationError(
                f"ViewerConfig camera_view must be one of {CAMERA_VIEWS}, got "
                f"{self.camera_view!r}."
            )
