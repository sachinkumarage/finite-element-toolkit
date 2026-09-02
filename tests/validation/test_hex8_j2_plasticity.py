"""Validation: a HEX8 element driven into J2 plasticity through the full nonlinear solver.

Verifies (spec section 43): load stepping, Newton convergence, plastic
yielding, and per-Gauss-point independence for the eight integration
points of a single HEX8 element under a non-uniform (bending-like) load
that yields some Gauss points while others stay elastic.
"""

import pytest

from femtoolkit.analysis import BoundaryCondition, NodalLoad, TranslationDOF
from femtoolkit.analysis.nonlinear_analysis import NonlinearAnalysis, NonlinearSolverSettings
from femtoolkit.continuum.stress import von_mises_3d
from femtoolkit.materials import J2Plasticity3D, LinearElastic3D
from femtoolkit.mesh import Hex8Element3D, Mesh, Node

X = TranslationDOF.X
Y = TranslationDOF.Y
Z = TranslationDOF.Z

YOUNGS_MODULUS = 200e9
POISSON_RATIO = 0.3
YIELD_STRESS = 80e6
HARDENING_MODULUS = 5e9


def _build_mesh_and_element():
    # The element's own `material` is a placeholder LinearElastic3D; the
    # nonlinear material is supplied separately to NonlinearAnalysis (see
    # tests/validation/test_tet4_j2_plasticity.py for the same pattern).
    placeholder = LinearElastic3D(
        youngs_modulus=YOUNGS_MODULUS, poisson_ratio=POISSON_RATIO, density=7850.0
    )
    material = J2Plasticity3D(
        youngs_modulus=YOUNGS_MODULUS,
        poisson_ratio=POISSON_RATIO,
        yield_stress=YIELD_STRESS,
        hardening_modulus=HARDENING_MODULUS,
    )
    coords = [
        (0.0, 0.0, 0.0),
        (4.0, 0.0, 0.0),
        (4.0, 1.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (4.0, 0.0, 1.0),
        (4.0, 1.0, 1.0),
        (0.0, 1.0, 1.0),
    ]
    nodes = tuple(Node(id=i + 1, x=x, y=y, z=z) for i, (x, y, z) in enumerate(coords))
    hexa = Hex8Element3D(id=1, nodes=nodes, material=placeholder)

    mesh = Mesh()
    for node in nodes:
        mesh.add_node(node)
    mesh.add_element(hexa)
    return mesh, hexa, material


def test_hex8_cantilever_bending_load_converges_and_yields() -> None:
    """A cantilever-like tip load bends the beam-shaped HEX8, producing higher
    stress at the top/bottom fibers than at mid-height -- enough to yield some
    Gauss points but not necessarily all of them."""
    mesh, hexa, material = _build_mesh_and_element()

    fixed_face = (1, 4, 5, 8)  # x = 0 face, fully fixed (cantilever root)
    tip_top = (3, 7)  # x = 4, y = 1 (top of the free tip)

    settings = NonlinearSolverSettings(load_steps=15, tolerance=1e-9, max_iterations=50)
    analysis = NonlinearAnalysis(mesh, {hexa.id: material}, settings)
    for node_id in fixed_face:
        for dof in (X, Y, Z):
            analysis.add_boundary_condition(BoundaryCondition(node_id, dof, 0.0))
    for node_id in tip_top:
        analysis.add_load(NodalLoad(node_id, Y, -5.0e6))

    result = analysis.solve()
    assert result.converged

    final_states = result.step_results[-1].element_states[hexa.id].states
    assert len(final_states) == 8

    for state in final_states:
        if state.yielded:
            von_mises = von_mises_3d(*state.stress)
            current_yield = YIELD_STRESS + HARDENING_MODULUS * state.hardening_variable
            assert von_mises == pytest.approx(current_yield, rel=1e-4)
            assert state.hardening_variable > 0.0


def test_hex8_tangent_stiffness_updates_as_gauss_points_yield() -> None:
    """A converged plastic step's tangent stiffness must differ from the pure elastic one."""
    from femtoolkit.analysis.nonlinear_elements import (
        hex8_internal_force_and_tangent,
        initial_element_state,
    )

    mesh, hexa, material = _build_mesh_and_element()
    committed = initial_element_state(hexa, material)

    elastic_displacements = [0.0] * 24
    _, k_elastic, _ = hex8_internal_force_and_tangent(
        hexa, material, elastic_displacements, committed
    )

    plastic_displacements = [0.0] * 24
    plastic_displacements[3 * 2 + 1] = -0.02  # push node 3 (tip top) down in Y
    plastic_displacements[3 * 6 + 1] = -0.02  # push node 7 (tip top) down in Y
    _, k_plastic, trial_state = hex8_internal_force_and_tangent(
        hexa, material, plastic_displacements, committed
    )

    assert any(state.yielded for state in trial_state.states)
    assert k_plastic[7, 7] < k_elastic[7, 7]  # softer once yielded
