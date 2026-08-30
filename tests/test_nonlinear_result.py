"""Tests for LoadStepResult and NonlinearAnalysisResult."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from femtoolkit.analysis import BoundaryCondition, NodalLoad, TranslationDOF
from femtoolkit.analysis.dof import DOFMap
from femtoolkit.analysis.nonlinear_analysis import NonlinearAnalysis, NonlinearSolverSettings
from femtoolkit.exceptions import ValidationError
from femtoolkit.materials import ElasticMaterialAdapter, LinearElastic2D
from femtoolkit.mesh.cst_element import CSTElement2D
from femtoolkit.mesh.mesh import Mesh
from femtoolkit.mesh.node import Node
from femtoolkit.results.nonlinear_result import LoadStepResult, NonlinearAnalysisResult

X = TranslationDOF.X
Y = TranslationDOF.Y

LINEAR_MATERIAL = LinearElastic2D(
    youngs_modulus=210e9, poisson_ratio=0.3, formulation="plane_stress"
)


def _two_triangle_mesh() -> tuple[Mesh, CSTElement2D, CSTElement2D]:
    node_1 = Node(id=1, x=0.0, y=0.0, z=0.0)
    node_2 = Node(id=2, x=1.0, y=0.0, z=0.0)
    node_3 = Node(id=3, x=1.0, y=1.0, z=0.0)
    node_4 = Node(id=4, x=0.0, y=1.0, z=0.0)
    lower = CSTElement2D(
        id=1, nodes=(node_1, node_2, node_3), material=LINEAR_MATERIAL, thickness=0.01
    )
    upper = CSTElement2D(
        id=2, nodes=(node_1, node_3, node_4), material=LINEAR_MATERIAL, thickness=0.01
    )

    mesh = Mesh()
    for node in (node_1, node_2, node_3, node_4):
        mesh.add_node(node)
    mesh.add_element(lower)
    mesh.add_element(upper)
    return mesh, lower, upper


def _solved_result() -> NonlinearAnalysisResult:
    mesh, lower, upper = _two_triangle_mesh()
    materials = {
        e.id: ElasticMaterialAdapter.from_linear_elastic_2d(LINEAR_MATERIAL) for e in (lower, upper)
    }
    analysis = NonlinearAnalysis(mesh, materials, NonlinearSolverSettings(load_steps=4))
    analysis.add_boundary_condition(BoundaryCondition(1, X, 0.0))
    analysis.add_boundary_condition(BoundaryCondition(1, Y, 0.0))
    analysis.add_boundary_condition(BoundaryCondition(4, X, 0.0))
    analysis.add_boundary_condition(BoundaryCondition(4, Y, 0.0))
    analysis.add_load(NodalLoad(2, X, 1.0e6))
    analysis.add_load(NodalLoad(3, X, 1.0e6))
    return analysis.solve()


def test_converged_property_true_when_every_step_converges() -> None:
    result = _solved_result()

    assert result.converged is True
    assert all(step.converged for step in result.step_results)


def test_converged_property_false_if_any_step_failed() -> None:
    dof_map = DOFMap(node_ids=[1, 2], dofs_per_node=1)
    good = LoadStepResult(
        load_factor=0.5,
        displacement=np.zeros(2),
        reactions=np.zeros(2),
        residual_norm=1e-10,
        iterations=2,
        converged=True,
        element_states={},
    )
    bad = LoadStepResult(
        load_factor=1.0,
        displacement=np.zeros(2),
        reactions=np.zeros(2),
        residual_norm=0.5,
        iterations=25,
        converged=False,
        element_states={},
    )
    result = NonlinearAnalysisResult(dof_map=dof_map, step_results=(good, bad))

    assert result.converged is False


def test_displacement_and_history_accessors() -> None:
    result = _solved_result()

    final = result.displacement(2, X)
    history = result.displacement_history(2, X)

    assert history.shape == (4,)
    assert history[-1] == pytest.approx(final)
    # Monotonically increasing under monotonically increasing load.
    assert np.all(np.diff(history) > 0)


def test_node_displacement_returns_all_active_dofs() -> None:
    result = _solved_result()

    ux, uy = result.node_displacement(3)

    assert ux == pytest.approx(result.displacement(3, X))
    assert uy == pytest.approx(result.displacement(3, Y))


def test_reaction_and_node_reaction_accessors() -> None:
    result = _solved_result()

    reaction_x = result.reaction(1, X)
    rx, ry = result.node_reaction(1)

    assert rx == pytest.approx(reaction_x)
    assert isinstance(ry, float)


def test_element_state_accessors_match_underlying_material_state() -> None:
    result = _solved_result()

    stress = result.element_stress(1, gauss_point=0)
    strain = result.element_strain(1, gauss_point=0)
    plastic_strain = result.element_plastic_strain(1, gauss_point=0)
    yielded = result.element_yielded(1, gauss_point=0)

    raw_state = result.step_results[-1].element_states[1].states[0]
    assert_allclose(stress, raw_state.stress)
    assert_allclose(strain, raw_state.strain)
    assert_allclose(plastic_strain, raw_state.plastic_strain)
    assert yielded == raw_state.yielded


def test_element_state_rejects_unknown_element_id() -> None:
    result = _solved_result()

    with pytest.raises(ValidationError):
        result.element_state(999)


def test_load_factors_iterations_and_residual_norms_arrays() -> None:
    result = _solved_result()

    assert result.load_factors().shape == (4,)
    assert result.iteration_counts().shape == (4,)
    assert result.residual_norms().shape == (4,)
    assert np.all(result.iteration_counts() >= 1)


def test_step_index_selects_a_specific_load_step() -> None:
    result = _solved_result()

    first_step_displacement = result.displacement(2, X, step=0)
    last_step_displacement = result.displacement(2, X, step=-1)

    assert first_step_displacement < last_step_displacement
