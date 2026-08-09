"""Finite Element Toolkit.

An open-source Python toolkit for developing finite element analysis (FEA)
capabilities. Version 1 established the project foundation: domain data
models for materials, nodes, elements, and meshes. Version 2 added the
basic mathematical foundation for FEA: degrees of freedom, boundary
conditions, nodal loads, the 1D bar element stiffness matrix, global
matrix assembly, and a basic linear solver for ``[K]{u} = {F}``. Version 3
turns that foundation into a validated 1D structural analysis capability:
a dedicated bar element with cross-sectional area, a
:class:`~femtoolkit.analysis.static_linear.StaticLinearAnalysis` workflow,
and results giving displacement, reaction, strain, stress, and axial
force.

The toolkit does not yet implement beam, truss, 2D, or 3D elements,
nonlinear or dynamic analysis, or visualization.
"""

from femtoolkit import logging_config  # noqa: F401  (attaches NullHandler on import)
from femtoolkit.config import __version__

__all__ = ["__version__"]
