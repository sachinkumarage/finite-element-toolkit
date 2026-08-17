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
bending moment, and bending stress results. Version 6 introduced the
toolkit's first 2D continuum element: a
:class:`~femtoolkit.mesh.cst_element.CSTElement2D` (3-node constant
strain triangle) representing a finite area of material, with plane
stress/plane strain constitutive models
(:class:`~femtoolkit.materials.linear_elastic_2d.LinearElastic2D`), a
strain-displacement matrix, and von Mises/principal stress recovery, all
built from the independently testable math in :mod:`femtoolkit.continuum`.
Version 7 added a second continuum element,
:class:`~femtoolkit.mesh.quad_element.QuadElement2D` (4-node bilinear
quadrilateral, "Q4"): natural coordinates, isoparametric mapping, the
Jacobian, and 2x2 Gauss quadrature -- needed because, unlike the CST
element, a Q4 element's strain-displacement matrix has no closed form and
varies within the element. It satisfies the same element protocols
introduced in Version 6, so the solver required no Q4-specific code.
Version 8 adds automatic structured 2D mesh generation
(:mod:`femtoolkit.mesh.generator`): :func:`~femtoolkit.mesh.generator.create_quad_mesh`
and :func:`~femtoolkit.mesh.generator.create_triangular_mesh` turn a
rectangular domain into a fully connected, deterministically numbered
mesh, alongside whole-mesh validation
(:mod:`femtoolkit.mesh.validation`), shape-quality metrics
(:mod:`femtoolkit.mesh.quality`), and a JSON export/import foundation
(:mod:`femtoolkit.mesh.serialization`). Generated meshes work with the
existing solver unchanged. Version 9 adds a lightweight 2D geometry
foundation (:mod:`femtoolkit.geometry`): :class:`~femtoolkit.geometry.point.Point2D`,
:class:`~femtoolkit.geometry.line.LineSegment2D`,
:class:`~femtoolkit.geometry.rectangle.Rectangle`, and named
:class:`~femtoolkit.geometry.boundary.BoundaryRegion` objects, plus
:meth:`~femtoolkit.mesh.mesh.Mesh.nodes_on_boundary` for tolerance-based,
coordinate-driven node selection and generic topological boundary-edge
detection (:mod:`femtoolkit.mesh.edges`). Distributed surface tractions
(:class:`~femtoolkit.analysis.distributed_load.DistributedLoad`) are
converted to equivalent nodal forces via edge integration
(:mod:`femtoolkit.continuum.edge`, 2-point Gauss quadrature for Q4, exact
linear integration for CST) and fed to the unmodified Version 3 solver,
alongside boundary-region boundary conditions
(:func:`~femtoolkit.analysis.boundary_conditions.boundary_conditions_for_region`)
and a :class:`~femtoolkit.analysis.load_case.LoadCase` workflow
abstraction.

The toolkit does not yet implement CAD or NURBS geometry, curved or 3D
geometry, unstructured or CAD-driven meshing, adaptive mesh refinement,
higher-order continuum elements, 3D beams, Timoshenko beams, plate,
shell, or 3D solid elements, load combinations, body forces, nonlinear
or dynamic analysis, or visualization.
"""

from femtoolkit import logging_config  # noqa: F401  (attaches NullHandler on import)
from femtoolkit.config import __version__

__all__ = ["__version__"]
