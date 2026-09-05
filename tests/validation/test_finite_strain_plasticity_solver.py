"""Validation: J2FiniteStrainPlasticity3D driven through Newton-Raphson (spec sections 14, 15).

Mirrors tests/validation/test_nonlinear_hex8_geometric.py (Version 16) and
tests/validation/test_hyperelastic_block_test.py (Version 17)'s Newton-
Raphson patch tests, for the Version 18 finite-strain plastic material:
elastic loading, the elastic-to-plastic transition, multiple load
increments, convergence, and permanent (plastic) deformation after the
load is fully applied.
"""

import numpy as np
import pytest

from femtoolkit.analysis import BoundaryCondition, NodalLoad, TranslationDOF
from femtoolkit.analysis.nonlinear_analysis import NonlinearAnalysis, NonlinearSolverSettings
from femtoolkit.materials import J2FiniteStrainPlasticity3D, LinearElastic3D
from femtoolkit.mesh import Hex8Element3D, Mesh, Node

X = TranslationDOF.X
Y = TranslationDOF.Y
Z = TranslationDOF.Z

_COORDS = [
    (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0), (0.0, 1.0, 1.0),
]
_FIXED_FACE = (1, 4, 5, 8)
_LOADED_FACE = (2, 3, 6, 7)


def _build_analysis(
    load_per_node: float, load_steps: int = 20
) -> tuple[NonlinearAnalysis, Hex8Element3D]:
    placeholder = LinearElastic3D(youngs_modulus=200e9, poisson_ratio=0.3, density=7850.0)
    nodes = tuple(Node(id=i + 1, x=x, y=y, z=z) for i, (x, y, z) in enumerate(_COORDS))
    hexa = Hex8Element3D(id=1, nodes=nodes, material=placeholder)

    mesh = Mesh()
    for node in nodes:
        mesh.add_node(node)
    mesh.add_element(hexa)

    material = J2FiniteStrainPlasticity3D(
        youngs_modulus=200e9, poisson_ratio=0.3, yield_stress=250e6, hardening_modulus=10e9
    )
    settings = NonlinearSolverSettings(load_steps=load_steps, tolerance=1e-8, max_iterations=40)
    analysis = NonlinearAnalysis(mesh, {hexa.id: material}, settings, geometric_nonlinearity=True)
    for node_id in _FIXED_FACE:
        for dof in (X, Y, Z):
            analysis.add_boundary_condition(BoundaryCondition(node_id, dof, 0.0))
    for node_id in _LOADED_FACE:
        analysis.add_load(NodalLoad(node_id, X, load_per_node))
    return analysis, hexa


def test_small_load_stays_fully_elastic() -> None:
    analysis, hexa = _build_analysis(load_per_node=2.0e6)
    result = analysis.solve()
    assert result.converged
    for gauss_point in range(8):
        assert not result.element_state(1, gauss_point=gauss_point).yielded


def test_elastic_to_plastic_transition_and_convergence() -> None:
    analysis, hexa = _build_analysis(load_per_node=8.0e7)
    result = analysis.solve()
    assert result.converged
    assert all(step.iterations >= 1 for step in result.step_results)
    assert all(step.iterations < 40 for step in result.step_results)

    yielded_flags = [
        result.element_state(1, gauss_point=gp).yielded for gp in range(8)
    ]
    assert any(yielded_flags)

    # Early (small-load-factor) steps must stay elastic; later ones yield --
    # confirms the elastic-to-plastic transition genuinely happens partway
    # through the load history, not immediately.
    early_step = result.step_results[1]
    early_state = early_step.element_states[1].states[0]
    assert not early_state.yielded


def test_reaction_forces_balance_applied_load_through_plastic_range() -> None:
    load_per_node = 9.0e7
    analysis, hexa = _build_analysis(load_per_node=load_per_node)
    result = analysis.solve()
    assert result.converged

    reaction_x_total = sum(result.reaction(node_id, X) for node_id in _FIXED_FACE)
    total_applied = load_per_node * len(_LOADED_FACE)
    assert reaction_x_total == pytest.approx(-total_applied, rel=1e-6)


def test_final_state_shows_permanent_plastic_deformation() -> None:
    analysis, hexa = _build_analysis(load_per_node=9.0e7)
    result = analysis.solve()
    assert result.converged

    for gauss_point in range(8):
        state = result.element_state(1, gauss_point=gauss_point)
        assert state.yielded
        assert state.hardening_variable > 0.0
        # plastic_deformation_gradient must have measurably departed from identity.
        assert not np.allclose(state.plastic_deformation_gradient, np.eye(3), atol=1e-6)


def test_gauss_points_all_report_valid_states() -> None:
    analysis, hexa = _build_analysis(load_per_node=6.0e7)
    result = analysis.solve()
    assert result.converged
    for gauss_point in range(8):
        state = result.element_state(1, gauss_point=gauss_point)
        assert state.strain.shape == (6,)
        assert state.stress.shape == (6,)
        assert state.plastic_deformation_gradient.shape == (3, 3)


def test_more_load_steps_gives_consistent_converged_result() -> None:
    """Solving with a finer load-step discretization should give a closely
    matching final converged state (numerical stability across step counts)."""
    coarse_analysis, _ = _build_analysis(load_per_node=7.0e7, load_steps=10)
    fine_analysis, _ = _build_analysis(load_per_node=7.0e7, load_steps=40)

    coarse_result = coarse_analysis.solve()
    fine_result = fine_analysis.solve()
    assert coarse_result.converged
    assert fine_result.converged

    coarse_state = coarse_result.element_state(1, gauss_point=0)
    fine_state = fine_result.element_state(1, gauss_point=0)
    assert coarse_state.stress[0] == pytest.approx(fine_state.stress[0], rel=0.03)
    assert coarse_state.hardening_variable == pytest.approx(
        fine_state.hardening_variable, rel=0.05
    )
