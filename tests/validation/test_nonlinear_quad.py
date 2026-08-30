"""Validation: a small nonlinear Q4 model (Section 27).

Verifies Gauss-point state, internal force, tangent stiffness, Newton
iterations, and convergence for a single Q4 element. No complex
engineering benchmark is required by the spec; this keeps the case small
and deterministic while still exercising every required piece.
"""

import numpy as np
from numpy.testing import assert_allclose

from femtoolkit.analysis import BoundaryCondition, NodalLoad, TranslationDOF
from femtoolkit.analysis.nonlinear_analysis import NonlinearAnalysis, NonlinearSolverSettings
from femtoolkit.materials import ElasticMaterialAdapter, LinearElastic2D
from femtoolkit.mesh.mesh import Mesh
from femtoolkit.mesh.node import Node
from femtoolkit.mesh.quad_element import QuadElement2D

X = TranslationDOF.X
Y = TranslationDOF.Y


def test_single_quad_nonlinear_uniaxial_tension() -> None:
    material_2d = LinearElastic2D(
        youngs_modulus=200e9, poisson_ratio=0.25, formulation="plane_stress"
    )
    node_1 = Node(id=1, x=0.0, y=0.0, z=0.0)
    node_2 = Node(id=2, x=1.0, y=0.0, z=0.0)
    node_3 = Node(id=3, x=1.0, y=1.0, z=0.0)
    node_4 = Node(id=4, x=0.0, y=1.0, z=0.0)
    quad = QuadElement2D(
        id=1, nodes=(node_1, node_2, node_3, node_4), material=material_2d, thickness=0.01
    )

    mesh = Mesh()
    for node in (node_1, node_2, node_3, node_4):
        mesh.add_node(node)
    mesh.add_element(quad)

    materials = {quad.id: ElasticMaterialAdapter.from_linear_elastic_2d(material_2d)}
    settings = NonlinearSolverSettings(load_steps=5, tolerance=1e-10)
    analysis = NonlinearAnalysis(mesh, materials, settings)
    # A true uniaxial-tension patch test: ux=0 along the whole left edge
    # (prevents rotation) but uy=0 at only one node (removes the rigid-body
    # y-translation without also over-constraining Poisson contraction,
    # which would otherwise make the strain field genuinely non-uniform).
    analysis.add_boundary_condition(BoundaryCondition(1, X, 0.0))
    analysis.add_boundary_condition(BoundaryCondition(1, Y, 0.0))
    analysis.add_boundary_condition(BoundaryCondition(4, X, 0.0))
    analysis.add_load(NodalLoad(2, X, 2.0e5))
    analysis.add_load(NodalLoad(3, X, 2.0e5))
    result = analysis.solve()

    assert result.converged
    assert np.all(result.iteration_counts() <= 2)

    # Uniform uniaxial tension: every Gauss point should report (nearly)
    # the same uniform strain/stress state.
    states = result.step_results[-1].element_states[quad.id].states
    assert len(states) == 4
    strains = np.array([state.strain for state in states])
    assert_allclose(strains, np.broadcast_to(strains[0], strains.shape), atol=1e-9)


def test_single_quad_nonlinear_bending_produces_gauss_point_variation() -> None:
    """A non-uniform (bending-like) load must produce different Gauss-point
    states -- confirming genuinely independent per-Gauss-point evaluation
    rather than a single averaged stress.
    """
    material_2d = LinearElastic2D(
        youngs_modulus=200e9, poisson_ratio=0.3, formulation="plane_stress"
    )
    node_1 = Node(id=1, x=0.0, y=0.0, z=0.0)
    node_2 = Node(id=2, x=2.0, y=0.0, z=0.0)
    node_3 = Node(id=3, x=2.0, y=0.5, z=0.0)
    node_4 = Node(id=4, x=0.0, y=0.5, z=0.0)
    quad = QuadElement2D(
        id=1, nodes=(node_1, node_2, node_3, node_4), material=material_2d, thickness=0.01
    )

    mesh = Mesh()
    for node in (node_1, node_2, node_3, node_4):
        mesh.add_node(node)
    mesh.add_element(quad)

    materials = {quad.id: ElasticMaterialAdapter.from_linear_elastic_2d(material_2d)}
    analysis = NonlinearAnalysis(mesh, materials, NonlinearSolverSettings(load_steps=4))
    analysis.add_boundary_condition(BoundaryCondition(1, X, 0.0))
    analysis.add_boundary_condition(BoundaryCondition(1, Y, 0.0))
    analysis.add_boundary_condition(BoundaryCondition(4, X, 0.0))
    analysis.add_boundary_condition(BoundaryCondition(4, Y, 0.0))
    # Opposing tip loads at the free edge: bending-like deformation.
    analysis.add_load(NodalLoad(2, Y, 5.0e4))
    analysis.add_load(NodalLoad(3, Y, -5.0e4))
    result = analysis.solve()

    assert result.converged
    states = result.step_results[-1].element_states[quad.id].states
    stresses = [state.stress for state in states]
    assert not all(np.allclose(stresses[0], stress) for stress in stresses[1:])
    assert len({id(state) for state in states}) == 4
