"""Validation: nonlinear solver + elastic material must reproduce the linear solver (Section 23).

This is the critical validation the Version 13 spec calls out by name:
solving a purely linear problem through the Newton-Raphson machinery
(:class:`~femtoolkit.analysis.nonlinear_analysis.NonlinearAnalysis`,
with :class:`~femtoolkit.materials.nonlinear.ElasticMaterialAdapter`)
must reproduce :class:`~femtoolkit.analysis.static_linear.StaticLinearAnalysis`'s
displacement field to numerical precision -- proof the internal-force,
tangent-stiffness, residual, assembly, and boundary-condition machinery
built for Version 13 is correct before any genuine material
nonlinearity is layered on top.
"""

import numpy as np
from numpy.testing import assert_allclose

from femtoolkit.analysis import BoundaryCondition, NodalLoad, StaticLinearAnalysis, TranslationDOF
from femtoolkit.analysis.nonlinear_analysis import NonlinearAnalysis, NonlinearSolverSettings
from femtoolkit.materials import ElasticMaterialAdapter, LinearElastic2D
from femtoolkit.mesh.cst_element import CSTElement2D
from femtoolkit.mesh.mesh import Mesh
from femtoolkit.mesh.node import Node
from femtoolkit.mesh.quad_element import QuadElement2D

X = TranslationDOF.X
Y = TranslationDOF.Y


def test_cst_patch_nonlinear_matches_linear() -> None:
    """A two-triangle patch (square domain) under uniaxial tension."""
    material = LinearElastic2D(youngs_modulus=210e9, poisson_ratio=0.3, formulation="plane_stress")
    node_1 = Node(id=1, x=0.0, y=0.0, z=0.0)
    node_2 = Node(id=2, x=1.0, y=0.0, z=0.0)
    node_3 = Node(id=3, x=1.0, y=1.0, z=0.0)
    node_4 = Node(id=4, x=0.0, y=1.0, z=0.0)
    lower = CSTElement2D(id=1, nodes=(node_1, node_2, node_3), material=material, thickness=0.01)
    upper = CSTElement2D(id=2, nodes=(node_1, node_3, node_4), material=material, thickness=0.01)

    mesh = Mesh()
    for node in (node_1, node_2, node_3, node_4):
        mesh.add_node(node)
    mesh.add_element(lower)
    mesh.add_element(upper)

    linear_analysis = StaticLinearAnalysis(mesh)
    linear_analysis.add_boundary_condition(BoundaryCondition(1, X, 0.0))
    linear_analysis.add_boundary_condition(BoundaryCondition(1, Y, 0.0))
    linear_analysis.add_boundary_condition(BoundaryCondition(4, X, 0.0))
    linear_analysis.add_boundary_condition(BoundaryCondition(4, Y, 0.0))
    linear_analysis.add_load(NodalLoad(2, X, 1.0e6))
    linear_analysis.add_load(NodalLoad(3, X, 1.0e6))
    linear_result = linear_analysis.solve()

    materials = {
        e.id: ElasticMaterialAdapter.from_linear_elastic_2d(material) for e in (lower, upper)
    }
    nonlinear_analysis = NonlinearAnalysis(
        mesh, materials, NonlinearSolverSettings(load_steps=8, tolerance=1e-10)
    )
    nonlinear_analysis.add_boundary_condition(BoundaryCondition(1, X, 0.0))
    nonlinear_analysis.add_boundary_condition(BoundaryCondition(1, Y, 0.0))
    nonlinear_analysis.add_boundary_condition(BoundaryCondition(4, X, 0.0))
    nonlinear_analysis.add_boundary_condition(BoundaryCondition(4, Y, 0.0))
    nonlinear_analysis.add_load(NodalLoad(2, X, 1.0e6))
    nonlinear_analysis.add_load(NodalLoad(3, X, 1.0e6))
    nonlinear_result = nonlinear_analysis.solve()

    assert nonlinear_result.converged
    for node_id in (2, 3):
        for dof in (X, Y):
            assert_allclose(
                nonlinear_result.displacement(node_id, dof),
                linear_result.displacement(node_id, dof),
                rtol=1e-8,
                atol=1e-15,
            )

    # A purely linear problem needs at most a predictor + one verification
    # iteration per load step -- never the full iteration budget.
    assert np.all(nonlinear_result.iteration_counts() <= 2)


def test_quad_plate_nonlinear_matches_linear() -> None:
    """A single Q4 element under combined tension and shear-inducing loads."""
    material = LinearElastic2D(youngs_modulus=70e9, poisson_ratio=0.33, formulation="plane_stress")
    node_1 = Node(id=1, x=0.0, y=0.0, z=0.0)
    node_2 = Node(id=2, x=2.0, y=0.0, z=0.0)
    node_3 = Node(id=3, x=2.0, y=1.0, z=0.0)
    node_4 = Node(id=4, x=0.0, y=1.0, z=0.0)
    quad = QuadElement2D(
        id=1, nodes=(node_1, node_2, node_3, node_4), material=material, thickness=0.02
    )

    mesh = Mesh()
    for node in (node_1, node_2, node_3, node_4):
        mesh.add_node(node)
    mesh.add_element(quad)

    linear_analysis = StaticLinearAnalysis(mesh)
    linear_analysis.add_boundary_condition(BoundaryCondition(1, X, 0.0))
    linear_analysis.add_boundary_condition(BoundaryCondition(1, Y, 0.0))
    linear_analysis.add_boundary_condition(BoundaryCondition(4, X, 0.0))
    linear_analysis.add_boundary_condition(BoundaryCondition(4, Y, 0.0))
    linear_analysis.add_load(NodalLoad(2, X, 4.0e5))
    linear_analysis.add_load(NodalLoad(3, X, 4.0e5))
    linear_analysis.add_load(NodalLoad(3, Y, 1.0e5))
    linear_result = linear_analysis.solve()

    materials = {quad.id: ElasticMaterialAdapter.from_linear_elastic_2d(material)}
    nonlinear_analysis = NonlinearAnalysis(
        mesh, materials, NonlinearSolverSettings(load_steps=6, tolerance=1e-10)
    )
    nonlinear_analysis.add_boundary_condition(BoundaryCondition(1, X, 0.0))
    nonlinear_analysis.add_boundary_condition(BoundaryCondition(1, Y, 0.0))
    nonlinear_analysis.add_boundary_condition(BoundaryCondition(4, X, 0.0))
    nonlinear_analysis.add_boundary_condition(BoundaryCondition(4, Y, 0.0))
    nonlinear_analysis.add_load(NodalLoad(2, X, 4.0e5))
    nonlinear_analysis.add_load(NodalLoad(3, X, 4.0e5))
    nonlinear_analysis.add_load(NodalLoad(3, Y, 1.0e5))
    nonlinear_result = nonlinear_analysis.solve()

    assert nonlinear_result.converged
    for node_id in (2, 3):
        for dof in (X, Y):
            assert_allclose(
                nonlinear_result.displacement(node_id, dof),
                linear_result.displacement(node_id, dof),
                rtol=1e-8,
                atol=1e-15,
            )
