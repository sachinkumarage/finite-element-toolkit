"""Finite Element Toolkit.

An open-source Python toolkit for developing finite element analysis (FEA)
capabilities. This release (Version 1) provides only the project
foundation: domain data models for materials, nodes, elements, and meshes.
It does not contain any numerical FEA solver.
"""

from femtoolkit import logging_config  # noqa: F401  (attaches NullHandler on import)
from femtoolkit.config import __version__

__all__ = ["__version__"]
