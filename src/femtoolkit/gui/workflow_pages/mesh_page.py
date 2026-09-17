"""Mesh page: Model Preparation workflow (Version 24 spec section 8, extended
Version 25 spec sections 15-17: sizing, validation, quality, refinement, statistics).
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import streamlit as st

from femtoolkit.application.exceptions_display import describe_error, describe_visualization_error
from femtoolkit.application.mesh_preparation_service import MeshPreparationService
from femtoolkit.application.validation import validate_mesh
from femtoolkit.exceptions import FiniteElementToolkitError
from femtoolkit.gui.components import error_banner, require_project
from femtoolkit.gui.state import AppState
from femtoolkit.gui.visualization import is_pyvista_available, render_mesh_quality_screenshot
from femtoolkit.mesh.sizing import MeshSizingParameters

if TYPE_CHECKING:
    from femtoolkit.application.project import Project
    from femtoolkit.mesh.mesh import Mesh
    from femtoolkit.mesh.quality.quality_report import MeshQualityReport

_prep_service = MeshPreparationService()


def render(state: AppState) -> None:
    """Render the Mesh (Model Preparation) page."""
    st.header("Mesh")
    st.caption(
        "Model Preparation: Configure -> Generate -> Validate -> Evaluate Quality -> "
        "Refine -> Inspect -> Accept."
    )

    if not require_project(state):
        return

    project = state.project
    is_mechanical = project.analysis_type in (
        "linear_static",
        "nonlinear_static",
        "thermomechanical",
    )

    _render_geometry_and_sizing(project)
    if not _render_configuration(project, is_mechanical):
        return

    st.divider()
    st.subheader("2. Generate Mesh")
    try:
        mesh = _prep_service.preview(project)
    except FiniteElementToolkitError as exc:
        st.error(describe_error(exc))
        return
    st.success(f"Generated {len(mesh.elements)} elements, {len(mesh.nodes)} nodes.")

    st.divider()
    _render_validation_section(mesh)

    st.divider()
    quality_report = _render_quality_section(mesh)

    st.divider()
    _render_statistics_section(mesh)

    st.divider()
    _render_refinement_section(project)

    st.divider()
    st.subheader("7. Inspect Mesh")
    _render_2d_preview(mesh, project.mesh.width, project.mesh.height)
    _render_3d_quality_preview(mesh, quality_report)

    st.divider()
    st.subheader("8. Accept Mesh")
    st.info(
        "This configuration is already the mesh used when the simulation runs -- "
        "no separate save step is needed. Continue to Boundary Conditions when ready."
    )


def _render_geometry_and_sizing(project: Project) -> None:
    st.subheader("1. Configure Mesh")
    st.write("**Geometry**")
    col_w, col_h = st.columns(2)
    with col_w:
        project.mesh.width = st.number_input("Width (m)", value=project.mesh.width, min_value=0.01)
    with col_h:
        project.mesh.height = st.number_input(
            "Height (m)", value=project.mesh.height, min_value=0.01
        )

    st.write("**Mesh Sizing**")
    col_target, col_min, col_max = st.columns(3)
    with col_target:
        target_size = st.number_input("Global target size (m)", value=0.2, min_value=1e-6)
    with col_min:
        minimum_size = st.number_input("Minimum size (m)", value=0.01, min_value=1e-6)
    with col_max:
        maximum_size = st.number_input("Maximum size (m)", value=1.0, min_value=1e-6)

    if st.button("Apply Sizing"):
        try:
            sizing = MeshSizingParameters(
                target_size=target_size, minimum_size=minimum_size, maximum_size=maximum_size
            )
            _prep_service.apply_sizing(project, sizing)
            st.success(f"Applied sizing: nx={project.mesh.nx}, ny={project.mesh.ny}.")
        except FiniteElementToolkitError as exc:
            st.error(describe_error(exc))


def _render_configuration(project: Project, is_mechanical: bool) -> bool:
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

    project.mesh.nx = int(nx)
    project.mesh.ny = int(ny)
    project.mesh.element_type = element_type
    project.mesh.thickness = thickness

    errors = validate_mesh(project)
    if errors:
        error_banner("Mesh configuration invalid", errors)
        return False
    return True


def _render_validation_section(mesh: Mesh) -> None:
    st.subheader("3. Validate Mesh")
    report = _prep_service.validation_report(mesh)
    if report.status == "OK":
        st.success("Status: OK -- no problems found.")
    elif report.status == "WARNING":
        st.warning(f"Status: WARNING -- {len(report.warnings)} warning(s).")
    else:
        st.error(f"Status: ERROR -- {len(report.errors)} error(s).")

    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Duplicate Nodes", report.num_duplicate_nodes)
    col_b.metric("Isolated Nodes", report.num_isolated_nodes)
    col_c.metric("Degenerate Elements", report.num_degenerate_elements)
    col_d.metric("Duplicate Elements", report.num_duplicate_elements)

    for warning in report.warnings:
        st.caption(f"Warning: {warning}")
    for error in report.errors:
        st.caption(f"Error: {error}")


def _render_quality_section(mesh: Mesh) -> MeshQualityReport | None:
    st.subheader("4. Evaluate Quality")
    try:
        report = _prep_service.quality_report(mesh)
    except FiniteElementToolkitError as exc:
        st.error(describe_error(exc))
        return None

    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Elements Evaluated", report.num_elements_evaluated)
    col_b.metric("Minimum Quality", f"{report.minimum_quality:.2f}")
    col_c.metric("Mean Quality", f"{report.mean_quality:.2f}")
    col_d.metric("Poor Elements", len(report.poor_quality_element_ids))

    for warning in report.warnings:
        st.caption(f"Warning: {warning}")
    return report


def _render_statistics_section(mesh: Mesh) -> None:
    st.subheader("5. Mesh Statistics")
    from femtoolkit.mesh.statistics import compute_mesh_statistics

    stats = compute_mesh_statistics(mesh)
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Nodes", stats.num_nodes)
    col_b.metric("Elements", stats.num_elements)
    col_c.metric("Dimension", stats.dimension)

    st.caption(f"Element types: {stats.element_type_counts}")
    st.caption(
        f"Bounding box: {stats.bounding_box_min} to {stats.bounding_box_max} m"
    )
    if stats.characteristic_size is not None:
        st.caption(f"Characteristic element size: {stats.characteristic_size:.4f} m")
    if stats.num_boundary_edges is not None:
        st.caption(f"Boundary edges: {stats.num_boundary_edges}")


def _render_refinement_section(project: Project) -> None:
    st.subheader("6. Refine Mesh")
    st.caption(
        "Uniform refinement (edge-midpoint quadrisection); applies automatically "
        "the next time the mesh is generated, including at simulation run time."
    )
    col_refine, col_reset = st.columns(2)
    with col_refine:
        if st.button("Refine (uniform, 1 pass)"):
            _prep_service.refine(project)
            st.rerun()
    with col_reset:
        if st.button("Reset Refinement"):
            _prep_service.reset_refinement(project)
            st.rerun()
    st.caption(f"Refinement passes applied: {project.mesh.refinement_passes}")


def _render_2d_preview(mesh: Mesh, width: float, height: float) -> None:
    st.write("**2D Preview**")
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


def _render_3d_quality_preview(mesh: Mesh, quality_report: MeshQualityReport | None) -> None:
    st.write("**3D Quality Preview**")
    if quality_report is None:
        st.caption("No quality report available.")
        return
    if not is_pyvista_available():
        st.caption(
            '3D quality visualization requires PyVista: pip install "femtoolkit[viz3d]"'
        )
        return

    metric = st.selectbox("Color by", options=["quality", "aspect_ratio", "jacobian_determinant"])
    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            screenshot_path = Path(tmp_dir) / "quality.png"
            render_mesh_quality_screenshot(mesh, quality_report, screenshot_path, metric=metric)
            st.image(str(screenshot_path), caption=f"Mesh colored by {metric}")
    except Exception as exc:  # noqa: BLE001 - visualization failures must never crash the page
        st.error(describe_visualization_error(exc))
