"""The lightweight Streamlit presentation layer for the Finite Element Toolkit.

Nothing under :mod:`app` is imported by :mod:`femtoolkit` itself -- the core
library remains completely independent of Streamlit. Every page here only ever
calls existing library APIs; no FEA, optimization, surrogate, or adaptive logic is
implemented in this package.
"""
