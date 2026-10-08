"""Quickstart Streamlit application entry point (Version 36).

Run with::

    streamlit run streamlit_app.py

Deliberately thin: it sets up the page, builds the sidebar navigation, and
dispatches to the selected page module under :mod:`app.streamlit.pages`. It
contains no FEA, optimization, surrogate, or adaptive-sampling logic itself --
see those pages, and the :mod:`femtoolkit` library they call, for that.

This is a separate, smaller companion to the existing full engineering GUI
(``streamlit run src/femtoolkit/gui/app.py``) -- five pages for a quick local
demo and future Streamlit Community Cloud deployment, not a replacement for the
full project-building workflow.
"""

from __future__ import annotations

import streamlit as st

from app.streamlit.pages import fea, home, optimization, results, surrogate_optimization
from femtoolkit.config import __version__

_PAGES = {
    "Home": home,
    "FEA Analysis": fea,
    "Optimization": optimization,
    "Surrogate-Assisted Optimization": surrogate_optimization,
    "Results": results,
}


def main() -> None:
    """Configure the page and render the currently selected page."""
    st.set_page_config(page_title="Finite Element Toolkit", page_icon="\U0001f4d0", layout="wide")

    with st.sidebar:
        st.title("Finite Element Toolkit")
        st.caption(f"Quickstart -- v{__version__}")
        page_name = st.radio("Navigation", options=list(_PAGES), label_visibility="collapsed")

    _PAGES[page_name].render()


if __name__ == "__main__":
    main()
