"""Finite Element Toolkit.

An open-source Python toolkit for developing finite element analysis (FEA)
capabilities. Version 1 established the project foundation. Version 2
added the basic mathematical foundation for FEA. Version 3 turned that
into a validated 1D structural analysis capability (a bar element, a
:class:`~femtoolkit.analysis.static_linear.StaticLinearAnalysis` workflow,
and results). Version 4 extended the same workflow to 2D pin-jointed truss
structures: two translational DOFs per node, a
:class:`~femtoolkit.mesh.truss_element.TrussElement2D` transformed from
local to global coordinates via its direction cosines, and X/Y loads,
constraints, displacements, reactions, and member forces. Version 5 added
2D Euler-Bernoulli beam and frame analysis: a rotational DOF per node
(:class:`~femtoolkit.analysis.dof.RotationDOF`), a
:class:`~femtoolkit.mesh.frame_element.FrameElement2D` that resists axial
force, shear force, and bending moment, and per-element shear force,
bending moment, and bending stress results. Version 6 introduces the
toolkit's first 2D continuum element: a
:class:`~femtoolkit.mesh.cst_element.CSTElement2D` (3-node constant
strain triangle) representing a finite area of material, with plane
stress/plane strain constitutive models
(:class:`~femtoolkit.materials.linear_elastic_2d.LinearElastic2D`), a
strain-displacement matrix, and von Mises/principal stress recovery, all
built from the independently testable math in :mod:`femtoolkit.continuum`.

The toolkit does not yet implement quadrilateral or higher-order
continuum elements, 3D beams, Timoshenko beams, plate, shell, or 3D solid
elements, nonlinear or dynamic analysis, or visualization.
"""

from femtoolkit import logging_config  # noqa: F401  (attaches NullHandler on import)
from femtoolkit.config import __version__

__all__ = ["__version__"]
