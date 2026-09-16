"""Results page: the engineering results dashboard (Version 24, spec sections 13, 15)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import streamlit as st

from femtoolkit.application.analysis_types import SUPPORTED_ANALYSIS_TYPES
from femtoolkit.application.results_service import ResultsService
from femtoolkit.gui.components import metric_row, run_status_banner
from femtoolkit.gui.state import AppState
from femtoolkit.postprocessing.visualization import (
    plot_deformed_shape_2d,
    plot_element_scatter_2d,
    plot_heat_flux_vectors_2d,
    plot_nodal_contour_2d,
)

_results_service = ResultsService()


def render(state: AppState) -> None:
    """Render the Results dashboard page."""
    st.header("Results Dashboard")

    if not state.has_project():
        st.warning("Create or load a project first, on the Project page.")
        return

    project = state.project
    info = SUPPORTED_ANALYSIS_TYPES.get(project.analysis_type)

    if not state.has_results():
        st.subheader(project.name)
        st.write(f"Analysis: {info.label if info else project.analysis_type}")
        run_status_banner(state.last_run)
        return

    simulation = state.last_run.simulation
    summary = state.last_run.summary

    st.subheader(project.name)
    st.write(f"Analysis: {info.label if info else project.analysis_type}")
    st.write(f"Material: {project.material.name}")
    cell_count = project.mesh.nx * project.mesh.ny
    st.write(f"Mesh: {project.mesh.element_type.upper()}, {cell_count} cells")

    metrics = []
    if summary.maximum_displacement is not None:
        metrics.append(("Max Displacement (m)", f"{summary.maximum_displacement:.4e}"))
    if summary.maximum_von_mises_stress is not None:
        metrics.append(("Max von Mises Stress (Pa)", f"{summary.maximum_von_mises_stress:.4e}"))
    if summary.maximum_temperature is not None:
        metrics.append(("Max Temperature (K)", f"{summary.maximum_temperature:.2f}"))
    if summary.maximum_heat_flux is not None:
        metrics.append(("Max Heat Flux (W/m^2)", f"{summary.maximum_heat_flux:.4e}"))
    metric_row(metrics)
    st.success("Status: Completed")

    st.divider()
    _render_mechanical_plots(simulation)
    _render_thermal_plots(simulation)


def _render_mechanical_plots(simulation) -> None:
    nodal_fields = simulation.final_step.nodal_fields
    element_fields = simulation.final_step.element_fields
    if "displacement" not in nodal_fields:
        return

    st.subheader("Mechanical Results")
    col_stress, col_deformed = st.columns(2)
    with col_stress:
        if "von_mises_stress" in element_fields:
            figure = plot_element_scatter_2d(simulation, "von_mises_stress")
            st.pyplot(figure)
            plt.close(figure)
    with col_deformed:
        scale = st.slider("Deformation display scale", 1.0, 2000.0, 200.0, key="deformed_scale")
        figure = plot_deformed_shape_2d(simulation, scale=scale, color_field="von_mises_stress")
        st.pyplot(figure)
        plt.close(figure)


def _render_thermal_plots(simulation) -> None:
    nodal_fields = simulation.final_step.nodal_fields
    element_fields = simulation.final_step.element_fields
    if "temperature" not in nodal_fields:
        return

    st.subheader("Thermal Results")
    col_temp, col_flux = st.columns(2)
    with col_temp:
        figure = plot_nodal_contour_2d(simulation, "temperature", cmap="inferno")
        st.pyplot(figure)
        plt.close(figure)
    with col_flux:
        if "heat_flux" in element_fields:
            figure = plot_heat_flux_vectors_2d(simulation)
            st.pyplot(figure)
            plt.close(figure)
