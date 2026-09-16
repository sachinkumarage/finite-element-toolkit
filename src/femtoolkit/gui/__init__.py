"""The Streamlit engineering GUI (Version 24).

This package is the top layer of the three-layer architecture this
version establishes:

.. code-block:: text

    femtoolkit.gui              (this package -- Streamlit widgets only)
         |
    femtoolkit.application      (service layer -- no Streamlit import)
         |
    femtoolkit core             (materials, mesh, analysis, thermal, postprocessing)

Only :mod:`femtoolkit.gui.app` and the individual page modules under
:mod:`femtoolkit.gui.workflow_pages` import :mod:`streamlit`; this package's
``__init__`` deliberately does not, so importing ``femtoolkit.gui``
itself never requires Streamlit to be installed (see
:mod:`femtoolkit.gui.state`, which is plain, Streamlit-free application
state logic). Run the application with::

    streamlit run src/femtoolkit/gui/app.py
"""
