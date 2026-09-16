"""Visualization page: Version 23 3D viewer controls (Version 24, spec section 14)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

from femtoolkit.application.exceptions_display import describe_visualization_error
from femtoolkit.gui.state import AppState
from femtoolkit.gui.visualization import (
    available_scalar_fields,
    is_pyvista_available,
    render_result_screenshot,
)


def render(state: AppState) -> None:
    """Render the 3D Visualization page."""
    st.header("3D Visualization")
    st.caption("Interactive controls over the Version 23 PyVista-based 3D viewer.")

    if not state.has_results():
        st.warning("Run a simulation first, on the Run page.")
        return

    if not is_pyvista_available():
        st.info(
            '3D visualization requires PyVista. Install it with: pip install "femtoolkit[viz3d]"'
        )
        st.caption("2D result plots remain available on the Results page without this extra.")
        return

    simulation = state.last_run.simulation
    settings = state.visualization_settings

    fields = available_scalar_fields(simulation)
    scalar_field = st.selectbox(
        "Scalar field",
        options=[None, *fields],
        index=(fields.index(settings.scalar_field) + 1) if settings.scalar_field in fields else 0,
        format_func=lambda name: "(none)" if name is None else name,
    )

    col_deformed, col_edges = st.columns(2)
    with col_deformed:
        deformed = st.checkbox("Show deformed geometry", value=settings.deformed)
    with col_edges:
        show_edges = st.checkbox("Show mesh edges", value=settings.show_edges)

    deformation_scale = settings.deformation_scale
    if deformed:
        deformation_scale = st.slider(
            "Deformation scale (cosmetic only)", 1.0, 5000.0, float(settings.deformation_scale)
        )

    settings.scalar_field = scalar_field
    settings.deformed = deformed
    settings.show_edges = show_edges
    settings.deformation_scale = deformation_scale
    state.visualization_settings = settings

    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            screenshot_path = Path(tmp_dir) / "viewer.png"
            render_result_screenshot(
                simulation,
                screenshot_path,
                scalar_field=scalar_field,
                deformed=deformed,
                deformation_scale=deformation_scale,
                show_edges=show_edges,
            )
            st.image(str(screenshot_path), caption="Current 3D view", use_container_width=True)
    except Exception as exc:  # noqa: BLE001 - visualization failures must never crash the page
        st.error(describe_visualization_error(exc))
