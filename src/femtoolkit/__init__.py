"""Finite Element Toolkit.

An open-source Python toolkit for developing finite element analysis (FEA)
capabilities. Version 1 established the project foundation: domain data
models for materials, nodes, elements, and meshes. Version 2 adds the
basic mathematical foundation for FEA: degrees of freedom, boundary
conditions, nodal loads, the 1D bar element stiffness matrix, global
matrix assembly, and a basic linear solver for ``[K]{u} = {F}``.

The toolkit does not yet implement beam or truss element abstractions,
2D/3D elements, stress or strain recovery, or visualization.
"""

from femtoolkit import logging_config  # noqa: F401  (attaches NullHandler on import)
from femtoolkit.config import __version__

__all__ = ["__version__"]
