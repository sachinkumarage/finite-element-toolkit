"""Streamlit Community Cloud entry point: launches the full engineering GUI.

Streamlit Community Cloud's default deployment flow expects a main file at the
repository root; this module exists only to satisfy that convention. It
contains no pages, state, or logic of its own -- it simply calls
:func:`femtoolkit.gui.app.main`, the same entry point ``streamlit run
src/femtoolkit/gui/app.py`` already uses, so both commands launch the
identical application::

    streamlit run streamlit_app.py
    streamlit run src/femtoolkit/gui/app.py
"""

from __future__ import annotations

from femtoolkit.gui.app import main

if __name__ == "__main__":
    main()
