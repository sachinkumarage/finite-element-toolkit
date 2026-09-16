"""Project page: create, name, save, load, reset (Version 24, spec section 5)."""

from __future__ import annotations

import streamlit as st

from femtoolkit.application.analysis_types import SUPPORTED_ANALYSIS_TYPES
from femtoolkit.application.exceptions_display import describe_error
from femtoolkit.application.project_service import ProjectService
from femtoolkit.exceptions import FiniteElementToolkitError
from femtoolkit.gui.state import AppState

_project_service = ProjectService()


def render(state: AppState) -> None:
    """Render the Project workspace page."""
    st.header("Project")
    st.caption("Create, save, load, or reset the engineering project workspace.")

    _render_create_section(state)
    st.divider()
    _render_save_load_section(state)
    st.divider()
    _render_current_project(state)


def _render_create_section(state: AppState) -> None:
    st.subheader("New Project")
    with st.form("create_project_form"):
        name = st.text_input("Project name", value="Untitled Project")
        analysis_labels = {key: info.label for key, info in SUPPORTED_ANALYSIS_TYPES.items()}
        analysis_type = st.selectbox(
            "Analysis type",
            options=list(analysis_labels),
            format_func=lambda key: analysis_labels[key],
        )
        info = SUPPORTED_ANALYSIS_TYPES[analysis_type]
        if not info.available:
            st.warning(f"Not yet available in this workspace: {info.reason}")
        submitted = st.form_submit_button("Create Project")
    if submitted:
        try:
            state.project = _project_service.create_project(name, analysis_type)
            st.success(f"Created project '{name}'.")
        except FiniteElementToolkitError as exc:
            st.error(describe_error(exc))


def _render_save_load_section(state: AppState) -> None:
    st.subheader("Save / Load")
    col_save, col_load = st.columns(2)

    with col_save:
        if state.has_project():
            project_json = _project_service.to_json(state.project)
            st.download_button(
                "Download project JSON",
                data=project_json,
                file_name=f"{state.project.name.replace(' ', '_').lower()}.json",
                mime="application/json",
            )
        else:
            st.caption("No project to save yet.")

    with col_load:
        uploaded = st.file_uploader("Load project JSON", type="json")
        if uploaded is not None:
            try:
                text = uploaded.getvalue().decode("utf-8")
                state.project = _project_service.from_json(text)
                st.success(f"Loaded project '{state.project.name}'.")
            except FiniteElementToolkitError as exc:
                st.error(describe_error(exc))

    if st.button("Reset Project", type="secondary"):
        state.reset()
        st.success("Project state reset.")


def _render_current_project(state: AppState) -> None:
    st.subheader("Current Project")
    if not state.has_project():
        st.info("No project loaded. Create one above.")
        return

    project = state.project
    info = SUPPORTED_ANALYSIS_TYPES.get(project.analysis_type)
    st.json(
        {
            "name": project.name,
            "analysis_type": info.label if info else project.analysis_type,
            "created_at": project.created_at,
            "boundary_conditions": len(project.boundary_conditions),
            "loads": len(project.loads),
        }
    )
