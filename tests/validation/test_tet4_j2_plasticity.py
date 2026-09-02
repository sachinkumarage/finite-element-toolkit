"""Validation: a TET4 element driven into J2 plasticity through the full nonlinear solver.

Verifies (spec section 43): load stepping, Newton convergence, plastic
yielding, stress, equivalent plastic strain, and tangent stiffness -- the
full 3D nonlinear pipeline (Mesh -> NonlinearAnalysis -> Newton-Raphson ->
J2Plasticity3D return mapping) working together, not just the material in
isolation (see tests/test_j2_plasticity.py for that).
"""

import pytest

from femtoolkit.analysis import BoundaryCondition, NodalLoad, TranslationDOF
from femtoolkit.analysis.nonlinear_analysis import NonlinearAnalysis, NonlinearSolverSettings
from femtoolkit.continuum.stress import von_mises_3d
from femtoolkit.materials import J2Plasticity3D, LinearElastic3D
from femtoolkit.mesh import Mesh, Node, Tet4Element3D

X = TranslationDOF.X
Y = TranslationDOF.Y
Z = TranslationDOF.Z

YOUNGS_MODULUS = 200e9
POISSON_RATIO = 0.3
YIELD_STRESS = 250e6
HARDENING_MODULUS = 10e9


def _build_mesh_and_element() -> tuple[Mesh, Tet4Element3D, J2Plasticity3D]:
    # The element's own `material` is a placeholder LinearElastic3D (elements
    # always carry a linear material -- see mesh/tet4_element.py); the
    # nonlinear material is supplied separately to NonlinearAnalysis, exactly
    # like examples/plastic_cst.py and examples/plastic_q4.py do for CST/Q4.
    placeholder = LinearElastic3D(
        youngs_modulus=YOUNGS_MODULUS, poisson_ratio=POISSON_RATIO, density=7850.0
    )
    material = J2Plasticity3D(
        youngs_modulus=YOUNGS_MODULUS,
        poisson_ratio=POISSON_RATIO,
        yield_stress=YIELD_STRESS,
        hardening_modulus=HARDENING_MODULUS,
    )
    n1 = Node(id=1, x=0.0, y=0.0, z=0.0)
    n2 = Node(id=2, x=1.0, y=0.0, z=0.0)
    n3 = Node(id=3, x=0.0, y=1.0, z=0.0)
    n4 = Node(id=4, x=0.0, y=0.0, z=1.0)
    tet = Tet4Element3D(id=1, nodes=(n1, n2, n3, n4), material=placeholder)

    mesh = Mesh()
    for node in (n1, n2, n3, n4):
        mesh.add_node(node)
    mesh.add_element(tet)
    return mesh, tet, material


def test_tet4_load_stepping_yields_and_converges() -> None:
    mesh, tet, material = _build_mesh_and_element()

    settings = NonlinearSolverSettings(load_steps=15, tolerance=1e-9, max_iterations=50)
    analysis = NonlinearAnalysis(mesh, {tet.id: material}, settings)
    analysis.add_boundary_condition(BoundaryCondition(1, X, 0.0))
    analysis.add_boundary_condition(BoundaryCondition(1, Y, 0.0))
    analysis.add_boundary_condition(BoundaryCondition(1, Z, 0.0))
    analysis.add_boundary_condition(BoundaryCondition(3, X, 0.0))
    analysis.add_boundary_condition(BoundaryCondition(3, Y, 0.0))
    analysis.add_boundary_condition(BoundaryCondition(3, Z, 0.0))
    analysis.add_boundary_condition(BoundaryCondition(4, X, 0.0))
    analysis.add_boundary_condition(BoundaryCondition(4, Y, 0.0))
    analysis.add_boundary_condition(BoundaryCondition(4, Z, 0.0))
    analysis.add_load(NodalLoad(2, X, 3.0e8))

    result = analysis.solve()
    assert result.converged

    # Every load step must have taken at least one Newton iteration.
    assert all(step.iterations >= 1 for step in result.step_results)
    # Newton must not have needed the full iteration budget at any step.
    assert all(step.iterations < settings.max_iterations for step in result.step_results)

    final_state = result.step_results[-1].element_states[1].states[0]
    assert final_state.yielded
    assert final_state.hardening_variable > 0.0

    von_mises = von_mises_3d(*final_state.stress)
    current_yield = YIELD_STRESS + HARDENING_MODULUS * final_state.hardening_variable
    assert von_mises == pytest.approx(current_yield, rel=1e-4)


def test_tet4_equivalent_plastic_strain_grows_monotonically_with_load() -> None:
    mesh, tet, material = _build_mesh_and_element()

    settings = NonlinearSolverSettings(load_steps=20, tolerance=1e-9, max_iterations=50)
    analysis = NonlinearAnalysis(mesh, {tet.id: material}, settings)
    analysis.add_boundary_condition(BoundaryCondition(1, X, 0.0))
    analysis.add_boundary_condition(BoundaryCondition(1, Y, 0.0))
    analysis.add_boundary_condition(BoundaryCondition(1, Z, 0.0))
    analysis.add_boundary_condition(BoundaryCondition(3, X, 0.0))
    analysis.add_boundary_condition(BoundaryCondition(3, Y, 0.0))
    analysis.add_boundary_condition(BoundaryCondition(3, Z, 0.0))
    analysis.add_boundary_condition(BoundaryCondition(4, X, 0.0))
    analysis.add_boundary_condition(BoundaryCondition(4, Y, 0.0))
    analysis.add_boundary_condition(BoundaryCondition(4, Z, 0.0))
    analysis.add_load(NodalLoad(2, X, 3.0e8))

    result = analysis.solve()
    assert result.converged

    alphas = [
        step.element_states[1].states[0].hardening_variable for step in result.step_results
    ]
    # Once yielding begins, alpha must never decrease.
    non_zero_alphas = [a for a in alphas if a > 0.0]
    assert non_zero_alphas == sorted(non_zero_alphas)
