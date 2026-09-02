"""Validation: 3D nonlinear solver + elastic material must reproduce the linear solver.

The 3D analogue of tests/validation/test_nonlinear_linear_regression.py
(Version 13's critical validation): solving a purely linear TET4/HEX8
problem through the Newton-Raphson machinery
(:class:`~femtoolkit.analysis.nonlinear_analysis.NonlinearAnalysis`, with
:class:`~femtoolkit.materials.nonlinear.ElasticMaterialAdapter.from_linear_elastic_3d`)
must reproduce :class:`~femtoolkit.analysis.static_linear.StaticLinearAnalysis`'s
displacement field to numerical precision -- proof the 3D internal-force,
tangent-stiffness, residual, assembly, and boundary-condition machinery is
correct *before* J2 plasticity is layered on top (spec section 34).
"""

import numpy as np
from numpy.testing import assert_allclose

from femtoolkit.analysis import BoundaryCondition, NodalLoad, StaticLinearAnalysis, TranslationDOF
from femtoolkit.analysis.nonlinear_analysis import NonlinearAnalysis, NonlinearSolverSettings
from femtoolkit.materials import LinearElastic3D
from femtoolkit.materials.nonlinear import ElasticMaterialAdapter
from femtoolkit.mesh import Hex8Element3D, Mesh, Node, Tet4Element3D

X = TranslationDOF.X
Y = TranslationDOF.Y
Z = TranslationDOF.Z


def test_tet4_nonlinear_matches_linear() -> None:
    material = LinearElastic3D(youngs_modulus=70e9, poisson_ratio=0.33, density=2700.0)
    n1 = Node(id=1, x=0.0, y=0.0, z=0.0)
    n2 = Node(id=2, x=1.0, y=0.0, z=0.0)
    n3 = Node(id=3, x=0.0, y=1.0, z=0.0)
    n4 = Node(id=4, x=0.0, y=0.0, z=1.0)
    tet = Tet4Element3D(id=1, nodes=(n1, n2, n3, n4), material=material)

    mesh = Mesh()
    for node in (n1, n2, n3, n4):
        mesh.add_node(node)
    mesh.add_element(tet)

    def _apply_bcs(analysis) -> None:
        for dof in (X, Y, Z):
            analysis.add_boundary_condition(BoundaryCondition(1, dof, 0.0))
        analysis.add_boundary_condition(BoundaryCondition(3, X, 0.0))
        analysis.add_boundary_condition(BoundaryCondition(3, Y, 0.0))
        analysis.add_boundary_condition(BoundaryCondition(3, Z, 0.0))
        analysis.add_boundary_condition(BoundaryCondition(4, X, 0.0))
        analysis.add_boundary_condition(BoundaryCondition(4, Y, 0.0))
        analysis.add_boundary_condition(BoundaryCondition(4, Z, 0.0))

    linear_analysis = StaticLinearAnalysis(mesh)
    _apply_bcs(linear_analysis)
    linear_analysis.add_load(NodalLoad(2, X, 5.0e6))
    linear_result = linear_analysis.solve()

    materials = {tet.id: ElasticMaterialAdapter.from_linear_elastic_3d(material)}
    nonlinear_analysis = NonlinearAnalysis(
        mesh, materials, NonlinearSolverSettings(load_steps=6, tolerance=1e-10)
    )
    _apply_bcs(nonlinear_analysis)
    nonlinear_analysis.add_load(NodalLoad(2, X, 5.0e6))
    nonlinear_result = nonlinear_analysis.solve()

    assert nonlinear_result.converged
    for dof in (X, Y, Z):
        assert_allclose(
            nonlinear_result.displacement(2, dof),
            linear_result.displacement(2, dof),
            rtol=1e-8,
            atol=1e-15,
        )
    assert np.all(nonlinear_result.iteration_counts() <= 2)


def test_hex8_nonlinear_matches_linear() -> None:
    material = LinearElastic3D(youngs_modulus=70e9, poisson_ratio=0.33, density=2700.0)
    coords = [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (1.0, 1.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (1.0, 0.0, 1.0),
        (1.0, 1.0, 1.0),
        (0.0, 1.0, 1.0),
    ]
    nodes = tuple(Node(id=i + 1, x=x, y=y, z=z) for i, (x, y, z) in enumerate(coords))
    hexa = Hex8Element3D(id=1, nodes=nodes, material=material)

    mesh = Mesh()
    for node in nodes:
        mesh.add_node(node)
    mesh.add_element(hexa)

    fixed_face = (1, 4, 5, 8)  # x = 0 face
    loaded_face = (2, 3, 6, 7)  # x = 1 face

    def _apply_bcs(analysis) -> None:
        for node_id in fixed_face:
            for dof in (X, Y, Z):
                analysis.add_boundary_condition(BoundaryCondition(node_id, dof, 0.0))

    linear_analysis = StaticLinearAnalysis(mesh)
    _apply_bcs(linear_analysis)
    for node_id in loaded_face:
        linear_analysis.add_load(NodalLoad(node_id, X, 2.0e6))
    linear_result = linear_analysis.solve()

    materials = {hexa.id: ElasticMaterialAdapter.from_linear_elastic_3d(material)}
    nonlinear_analysis = NonlinearAnalysis(
        mesh, materials, NonlinearSolverSettings(load_steps=6, tolerance=1e-10)
    )
    _apply_bcs(nonlinear_analysis)
    for node_id in loaded_face:
        nonlinear_analysis.add_load(NodalLoad(node_id, X, 2.0e6))
    nonlinear_result = nonlinear_analysis.solve()

    assert nonlinear_result.converged
    for node_id in loaded_face:
        for dof in (X, Y, Z):
            assert_allclose(
                nonlinear_result.displacement(node_id, dof),
                linear_result.displacement(node_id, dof),
                rtol=1e-8,
                atol=1e-15,
            )
