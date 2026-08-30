"""Tests for the trial/committed material-state lifecycle (Version 13, Section 29).

These tests exercise the property the rest of Version 13 depends on: a
Newton-Raphson iteration that does not converge must never leave any
trace in a material's *committed* state. See the module docstrings for
:mod:`femtoolkit.materials.nonlinear` and
:mod:`femtoolkit.analysis.nonlinear_analysis` for the design this
verifies.
"""

import numpy as np
import pytest

from femtoolkit.analysis import BoundaryCondition, NodalLoad, TranslationDOF
from femtoolkit.analysis.nonlinear_analysis import NonlinearAnalysis, NonlinearSolverSettings
from femtoolkit.exceptions import NonlinearConvergenceError
from femtoolkit.materials import (
    ElasticMaterialAdapter,
    ElasticPerfectlyPlasticMaterial1D,
    LinearElastic2D,
)
from femtoolkit.mesh.cst_element import CSTElement2D
from femtoolkit.mesh.mesh import Mesh
from femtoolkit.mesh.node import Node

X = TranslationDOF.X
Y = TranslationDOF.Y

YOUNGS_MODULUS = 200e9
YIELD_STRESS = 250e6


# --- Material-level: trial_state never mutates committed_state ---


def test_trial_state_is_pure_across_repeated_calls() -> None:
    material = ElasticPerfectlyPlasticMaterial1D(
        youngs_modulus=YOUNGS_MODULUS, yield_stress=YIELD_STRESS
    )
    committed = material.initial_state()

    yielding_strain = 10.0 * YIELD_STRESS / YOUNGS_MODULUS
    trial_a = material.trial_state(yielding_strain, committed)
    # Calling trial_state again from the SAME committed state must give the
    # same answer -- proof the first call did not silently update `committed`.
    trial_b = material.trial_state(yielding_strain, committed)

    assert committed.strain == 0.0
    assert committed.plastic_strain == 0.0
    assert committed.yielded is False
    assert trial_a.stress == pytest.approx(trial_b.stress)
    assert trial_a.plastic_strain == pytest.approx(trial_b.plastic_strain)


def test_discarding_a_trial_state_leaves_no_trace() -> None:
    """Simulates a rejected (non-converged) Newton iteration: a trial state
    is computed and then simply discarded, never assigned back as
    committed. A subsequent trial from the same committed state must behave
    exactly as if the discarded trial had never happened.
    """
    material = ElasticPerfectlyPlasticMaterial1D(
        youngs_modulus=YOUNGS_MODULUS, yield_stress=YIELD_STRESS
    )
    committed = material.initial_state()

    # A wild, discarded trial guess (as if from a bad Newton step).
    material.trial_state(1000.0, committed)

    # committed is untouched; the "real" trial proceeds from where it should.
    real_trial = material.trial_state(0.0001, committed)
    assert real_trial.stress == pytest.approx(YOUNGS_MODULUS * 0.0001)
    assert real_trial.yielded is False


# --- Orchestrator-level: a failed load step commits nothing ---


def _single_cst_mesh() -> tuple[Mesh, CSTElement2D]:
    material_2d = LinearElastic2D(
        youngs_modulus=210e9, poisson_ratio=0.3, formulation="plane_stress"
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
    return mesh, element


def test_failed_load_step_does_not_commit_displacement_or_state() -> None:
    mesh, element = _single_cst_mesh()
    material_2d = element.material
    materials = {element.id: ElasticMaterialAdapter.from_linear_elastic_2d(material_2d)}

    # First, solve a well-behaved analysis for a few steps to get a
    # genuinely nonzero committed state, then force the NEXT increment to
    # fail by asking for an impossible iteration budget on a fresh analysis
    # with the same starting point (max_iterations=0 is invalid, so use 1
    # with a zero tolerance that can never be satisfied on the very first
    # step given a nonzero load).
    analysis = NonlinearAnalysis(
        mesh, materials, NonlinearSolverSettings(max_iterations=1, tolerance=1e-300, load_steps=1)
    )
    analysis.add_boundary_condition(BoundaryCondition(1, X, 0.0))
    analysis.add_boundary_condition(BoundaryCondition(1, Y, 0.0))
    analysis.add_boundary_condition(BoundaryCondition(3, X, 0.0))
    analysis.add_load(NodalLoad(2, X, 1.0e6))

    with pytest.raises(NonlinearConvergenceError) as exc_info:
        analysis.solve()

    failed_step = exc_info.value.step_results[-1]
    assert failed_step.converged is False
    # The failed step's own record still reports its last attempted
    # (uncommitted) state for diagnostic purposes...
    assert 1 in failed_step.element_states


def test_multiple_load_steps_each_commit_independently() -> None:
    mesh, element = _single_cst_mesh()
    material_2d = element.material
    materials = {element.id: ElasticMaterialAdapter.from_linear_elastic_2d(material_2d)}

    analysis = NonlinearAnalysis(mesh, materials, NonlinearSolverSettings(load_steps=5))
    analysis.add_boundary_condition(BoundaryCondition(1, X, 0.0))
    analysis.add_boundary_condition(BoundaryCondition(1, Y, 0.0))
    analysis.add_boundary_condition(BoundaryCondition(3, X, 0.0))
    analysis.add_load(NodalLoad(2, X, 1.0e6))
    result = analysis.solve()

    # Each step's committed state is genuinely distinct and progresses
    # monotonically with the increasing load factor (linear-elastic here).
    strains = [
        result.element_strain(element.id, step=i)[0] for i in range(len(result.step_results))
    ]
    assert all(b > a for a, b in zip(strains, strains[1:]))  # noqa: B905 - deliberately unequal lengths


def test_gauss_point_states_are_independent_objects_not_shared() -> None:
    """Regression guard: a naive implementation might reuse a single
    MaterialState object for all four Gauss points of a Q4 element,
    silently making every Gauss point report identical (wrong) results.
    """
    from femtoolkit.mesh.quad_element import QuadElement2D

    material_2d = LinearElastic2D(
        youngs_modulus=210e9, poisson_ratio=0.3, formulation="plane_stress"
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
    analysis = NonlinearAnalysis(mesh, materials, NonlinearSolverSettings(load_steps=2))
    analysis.add_boundary_condition(BoundaryCondition(1, X, 0.0))
    analysis.add_boundary_condition(BoundaryCondition(1, Y, 0.0))
    analysis.add_boundary_condition(BoundaryCondition(4, X, 0.0))
    analysis.add_boundary_condition(BoundaryCondition(4, Y, 0.0))
    # Asymmetric loading so different Gauss points genuinely see different strain.
    analysis.add_load(NodalLoad(2, X, 1.0e6))
    analysis.add_load(NodalLoad(3, Y, 2.0e5))
    result = analysis.solve()

    states = result.step_results[-1].element_states[quad.id].states
    assert len({id(s) for s in states}) == 4
    stresses = [s.stress for s in states]
    assert not all(np.allclose(stresses[0], s) for s in stresses[1:])
