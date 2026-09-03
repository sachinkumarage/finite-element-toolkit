"""Validation: TET4 driven through Newton-Raphson under geometric nonlinearity (spec section 27)."""

import pytest
from numpy.testing import assert_allclose

from femtoolkit.analysis import BoundaryCondition, NodalLoad, TranslationDOF
from femtoolkit.analysis.nonlinear_analysis import NonlinearAnalysis, NonlinearSolverSettings
from femtoolkit.materials import LinearElastic3D, SaintVenantKirchhoff3D
from femtoolkit.mesh import Mesh, Node, Tet4Element3D

X = TranslationDOF.X
Y = TranslationDOF.Y
Z = TranslationDOF.Z


def _build_analysis(load: float) -> NonlinearAnalysis:
    placeholder = LinearElastic3D(youngs_modulus=200e9, poisson_ratio=0.3, density=7850.0)
    node_1 = Node(id=1, x=0.0, y=0.0, z=0.0)
    node_2 = Node(id=2, x=1.0, y=0.0, z=0.0)
    node_3 = Node(id=3, x=0.0, y=1.0, z=0.0)
    node_4 = Node(id=4, x=0.0, y=0.0, z=1.0)
    tet = Tet4Element3D(id=1, nodes=(node_1, node_2, node_3, node_4), material=placeholder)

    mesh = Mesh()
    for node in (node_1, node_2, node_3, node_4):
        mesh.add_node(node)
    mesh.add_element(tet)

    material = SaintVenantKirchhoff3D(youngs_modulus=200e9, poisson_ratio=0.3)
    settings = NonlinearSolverSettings(load_steps=10, tolerance=1e-9, max_iterations=40)
    analysis = NonlinearAnalysis(mesh, {tet.id: material}, settings, geometric_nonlinearity=True)
    for node_id in (1, 3, 4):
        for dof in (X, Y, Z):
            analysis.add_boundary_condition(BoundaryCondition(node_id, dof, 0.0))
    analysis.add_load(NodalLoad(2, X, load))
    return analysis


def test_converges_under_large_displacement() -> None:
    analysis = _build_analysis(load=3.0e7)
    result = analysis.solve()
    assert result.converged
    assert all(step.iterations >= 1 for step in result.step_results)
    assert all(step.iterations < 40 for step in result.step_results)


def test_load_stepping_reaches_full_load_factor() -> None:
    analysis = _build_analysis(load=3.0e7)
    result = analysis.solve()
    load_factors = result.load_factors()
    assert load_factors[-1] == pytest.approx(1.0)
    assert_allclose(load_factors, sorted(load_factors))  # strictly increasing


def test_final_strain_is_consistent_with_reported_displacement() -> None:
    """Strain recovered from the converged state must match a fresh recomputation
    from the same converged displacement -- an end-to-end consistency check."""
    from femtoolkit.analysis.geometric_nonlinear import (
        initial_element_state,
        tet4_geometric_internal_force_and_tangent,
    )

    analysis = _build_analysis(load=1.0e7)
    result = analysis.solve()
    assert result.converged

    placeholder = LinearElastic3D(youngs_modulus=200e9, poisson_ratio=0.3, density=7850.0)
    node_1 = Node(id=1, x=0.0, y=0.0, z=0.0)
    node_2 = Node(id=2, x=1.0, y=0.0, z=0.0)
    node_3 = Node(id=3, x=0.0, y=1.0, z=0.0)
    node_4 = Node(id=4, x=0.0, y=0.0, z=1.0)
    tet = Tet4Element3D(id=1, nodes=(node_1, node_2, node_3, node_4), material=placeholder)
    material = SaintVenantKirchhoff3D(youngs_modulus=200e9, poisson_ratio=0.3)
    committed = initial_element_state(tet, material)

    final_displacements = [
        result.displacement(node.id, dof)
        for node in (node_1, node_2, node_3, node_4)
        for dof in (X, Y, Z)
    ]
    _, _, trial_state = tet4_geometric_internal_force_and_tangent(
        tet, material, final_displacements, committed
    )
    assert_allclose(trial_state.states[0].strain, result.element_strain(1), atol=1e-10)


def test_reaction_forces_balance_applied_load() -> None:
    """Global equilibrium: sum of reactions at fixed nodes must balance the applied load."""
    analysis = _build_analysis(load=2.0e7)
    result = analysis.solve()
    assert result.converged

    reaction_x_total = sum(result.reaction(node_id, X) for node_id in (1, 3, 4))
    # Reactions oppose the applied load; equilibrium: sum(reactions) + applied = 0
    # (reactions are defined as F_int - F_ext, so at the free/loaded DOF the
    # residual is ~0 and reactions concentrate at the fixed DOFs).
    assert reaction_x_total == pytest.approx(-2.0e7, rel=1e-6)
