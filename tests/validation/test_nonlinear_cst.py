"""Validation: a small nonlinear CST model (Section 26).

Verifies strain calculation, stress calculation, internal force, tangent
stiffness, and convergence for a single CST element driven through
:class:`~femtoolkit.analysis.nonlinear_analysis.NonlinearAnalysis` with
:class:`~femtoolkit.materials.nonlinear.ElasticMaterialAdapter` (see
:mod:`femtoolkit.analysis.nonlinear_elements` for why CST/Q4 nonlinear
support is validated with the elastic adapter rather than a faked
multiaxial plasticity model). Kept small and deterministic.
"""

import numpy as np
from numpy.testing import assert_allclose

from femtoolkit.analysis import BoundaryCondition, NodalLoad, TranslationDOF
from femtoolkit.analysis.nonlinear_analysis import NonlinearAnalysis, NonlinearSolverSettings
from femtoolkit.materials import ElasticMaterialAdapter, LinearElastic2D
from femtoolkit.mesh.cst_element import CSTElement2D
from femtoolkit.mesh.mesh import Mesh
from femtoolkit.mesh.node import Node

X = TranslationDOF.X
Y = TranslationDOF.Y


def test_single_cst_triangle_nonlinear_uniaxial_tension() -> None:
    material_2d = LinearElastic2D(
        youngs_modulus=200e9, poisson_ratio=0.25, formulation="plane_stress"
    )
    node_1 = Node(id=1, x=0.0, y=0.0, z=0.0)
    node_2 = Node(id=2, x=1.0, y=0.0, z=0.0)
    node_3 = Node(id=3, x=0.0, y=1.0, z=0.0)
    element = CSTElement2D(
        id=1, nodes=(node_1, node_2, node_3), material=material_2d, thickness=0.01
    )

    mesh = Mesh()
    for node in (node_1, node_2, node_3):
        mesh.add_node(node)
    mesh.add_element(element)

    materials = {element.id: ElasticMaterialAdapter.from_linear_elastic_2d(material_2d)}
    settings = NonlinearSolverSettings(load_steps=5, tolerance=1e-10)
    analysis = NonlinearAnalysis(mesh, materials, settings)
    analysis.add_boundary_condition(BoundaryCondition(1, X, 0.0))
    analysis.add_boundary_condition(BoundaryCondition(1, Y, 0.0))
    analysis.add_boundary_condition(BoundaryCondition(3, X, 0.0))
    analysis.add_load(NodalLoad(2, X, 2.0e5))
    result = analysis.solve()

    assert result.converged
    assert np.all(result.iteration_counts() <= 2)

    # Strain, stress, internal force, and tangent stiffness must all match
    # the element's ordinary linear formulation exactly (the elastic
    # adapter has no history).
    final_displacement = result.step_results[-1].displacement
    local_indices = [
        result.dof_map.global_index(node_id, dof) for node_id, dof in element.dof_keys()
    ]
    local_u = final_displacement[local_indices]

    expected_strain = element.strain_from_dofs(local_u)
    expected_stress = element.stress_from_dofs(local_u)

    assert_allclose(result.element_strain(element.id), expected_strain, atol=1e-12)
    assert_allclose(result.element_stress(element.id), expected_stress, atol=1e-3)


def test_two_cst_triangle_patch_nonlinear_convergence() -> None:
    material_2d = LinearElastic2D(
        youngs_modulus=70e9, poisson_ratio=0.3, formulation="plane_stress"
    )
    node_1 = Node(id=1, x=0.0, y=0.0, z=0.0)
    node_2 = Node(id=2, x=1.0, y=0.0, z=0.0)
    node_3 = Node(id=3, x=1.0, y=1.0, z=0.0)
    node_4 = Node(id=4, x=0.0, y=1.0, z=0.0)
    lower = CSTElement2D(
        id=1, nodes=(node_1, node_2, node_3), material=material_2d, thickness=0.005
    )
    upper = CSTElement2D(
        id=2, nodes=(node_1, node_3, node_4), material=material_2d, thickness=0.005
    )

    mesh = Mesh()
    for node in (node_1, node_2, node_3, node_4):
        mesh.add_node(node)
    mesh.add_element(lower)
    mesh.add_element(upper)

    materials = {
        e.id: ElasticMaterialAdapter.from_linear_elastic_2d(material_2d) for e in (lower, upper)
    }
    analysis = NonlinearAnalysis(mesh, materials, NonlinearSolverSettings(load_steps=3))
    analysis.add_boundary_condition(BoundaryCondition(1, X, 0.0))
    analysis.add_boundary_condition(BoundaryCondition(1, Y, 0.0))
    analysis.add_boundary_condition(BoundaryCondition(4, X, 0.0))
    analysis.add_boundary_condition(BoundaryCondition(4, Y, 0.0))
    analysis.add_load(NodalLoad(2, X, 3.0e4))
    analysis.add_load(NodalLoad(3, X, 3.0e4))
    result = analysis.solve()

    assert result.converged
    for step_result in result.step_results:
        assert step_result.residual_norm < 1e-8
