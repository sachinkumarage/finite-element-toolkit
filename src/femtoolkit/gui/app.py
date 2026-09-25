"""Streamlit entry point: Engineering GUI & Interactive Simulation Workspace (Version 24).

Run with::

    streamlit run src/femtoolkit/gui/app.py

This module is deliberately thin: it sets up the page, builds the
:class:`~femtoolkit.gui.state.AppState` wrapper around
``st.session_state``, and dispatches to the selected page module under
:mod:`femtoolkit.gui.workflow_pages`. It contains no solver logic itself -- see
the "Important Architecture Rule" in this version's design docs
(``docs/gui.md``): the GUI must never become the FEA engine.
"""

from __future__ import annotations

import streamlit as st

from femtoolkit.config import __version__
from femtoolkit.gui.state import AppState
from femtoolkit.gui.workflow_pages import (
    boundary_conditions_page,
    loads_page,
    material_page,
    mesh_page,
    project_page,
    results_page,
    run_page,
    solver_page,
    studies_page,
    uncertainty_page,
    verification_page,
    visualization_page,
)

_PAGES = {
    "Project": project_page,
    "Material": material_page,
    "Mesh": mesh_page,
    "Boundary Conditions": boundary_conditions_page,
    "Loads": loads_page,
    "Solver": solver_page,
    "Run": run_page,
    "Results": results_page,
    "3D Visualization": visualization_page,
    "Verification & Validation": verification_page,
    "Simulation Studies": studies_page,
    "Uncertainty Analysis": uncertainty_page,
}


def main() -> None:
    """Configure the page and render the currently selected workflow step."""
    st.set_page_config(page_title="Finite Element Toolkit", page_icon="\U0001f4d0", layout="wide")

    state = AppState(st.session_state)

    with st.sidebar:
        st.title("Finite Element Toolkit")
        st.caption(f"Engineering Simulation Workspace -- v{__version__}")
        if state.has_project():
            st.write(f"**Project:** {state.project.name}")
        page_name = st.radio("Workflow", options=list(_PAGES), label_visibility="collapsed")

    _PAGES[page_name].render(state)


if __name__ == "__main__":
    main()
