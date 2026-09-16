"""Mesh page: configure and inspect the generated mesh (Version 24, spec section 8)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import streamlit as st

from femtoolkit.application.exceptions_display import describe_error
from femtoolkit.application.model_service import ModelService
from femtoolkit.application.validation import validate_mesh
from femtoolkit.exceptions import FiniteElementToolkitError
from femtoolkit.gui.components import error_banner, require_project
from femtoolkit.gui.state import AppState

_model_service = ModelService()


def render(state: AppState) -> None:
    """Render the Mesh configuration and inspection page."""
    st.header("Mesh")
    st.caption("Configure the structured 2D mesh and inspect its statistics and quality.")

    if not require_project(state):
        return

    project = state.project
    is_mechanical = project.analysis_type in (
        "linear_static",
        "nonlinear_static",
        "thermomechanical",
    )

    col_w, col_h = st.columns(2)
    with col_w:
        width = st.number_input("Width (m)", value=project.mesh.width, min_value=0.01)
    with col_h:
        height = st.number_input("Height (m)", value=project.mesh.height, min_value=0.01)

    col_nx, col_ny = st.columns(2)
    with col_nx:
        nx = st.number_input("Subdivisions (X)", value=project.mesh.nx, min_value=1, step=1)
    with col_ny:
        ny = st.number_input("Subdivisions (Y)", value=project.mesh.ny, min_value=1, step=1)

    element_type = st.selectbox(
        "Element type",
        options=["quad", "cst"],
        index=["quad", "cst"].index(project.mesh.element_type),
        format_func=(
            lambda key: "Q4 (bilinear quadrilateral)" if key == "quad" else "CST (triangle)"
        ),
    )

    thickness = project.mesh.thickness
    if is_mechanical:
        thickness = st.number_input("Thickness (m)", value=project.mesh.thickness, min_value=1e-6)

    project.mesh.width = width
    project.mesh.height = height
    project.mesh.nx = int(nx)
    project.mesh.ny = int(ny)
    project.mesh.element_type = element_type
    project.mesh.thickness = thickness

    errors = validate_mesh(project)
    if errors:
        error_banner("Mesh configuration invalid", errors)
        return

    st.divider()
    st.subheader("Mesh Summary")
    try:
        mesh = _model_service.build_mesh(project)
        summary = _model_service.mesh_summary(mesh)
    except FiniteElementToolkitError as exc:
        st.error(describe_error(exc))
        return

    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Nodes", summary.num_nodes)
    col_b.metric("Elements", summary.num_elements)
    col_c.metric("Dimension", summary.dimension)
    col_d.metric("Element Type", summary.element_type)

    st.caption(
        f"Quality: min={summary.quality.min_quality:.3f}, "
        f"max={summary.quality.max_quality:.3f}, "
        f"average={summary.quality.average_quality:.3f} "
        f"(invalid elements: {summary.quality.num_invalid_elements})"
    )

    st.subheader("Mesh Preview")
    figure, axis = plt.subplots(figsize=(6, 6 * height / width if width else 6))
    for node in mesh.nodes:
        axis.plot(node.x, node.y, "o", color="tab:blue", markersize=2)
    for element in mesh.elements:
        xs = [node.x for node in element.nodes] + [element.nodes[0].x]
        ys = [node.y for node in element.nodes] + [element.nodes[0].y]
        axis.plot(xs, ys, "-", color="tab:gray", linewidth=0.5)
    axis.set_aspect("equal")
    axis.set_xlabel("x (m)")
    axis.set_ylabel("y (m)")
    st.pyplot(figure)
    plt.close(figure)
