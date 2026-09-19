"""Tests for StaticLinearAnalysis's Version 26 solver-strategy integration."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from femtoolkit.analysis import BoundaryCondition, NodalLoad, StaticLinearAnalysis, TranslationDOF
from femtoolkit.analysis.multi_point_constraint import MultiPointConstraint
from femtoolkit.materials import LinearElastic2D
from femtoolkit.mesh.generator import create_quad_mesh
from femtoolkit.solvers import ConjugateGradientSolver, DenseDirectSolver, SparseDirectSolver

X = TranslationDOF.X
Y = TranslationDOF.Y


@pytest.fixture
def material() -> LinearElastic2D:
    return LinearElastic2D(youngs_modulus=200e9, poisson_ratio=0.3, formulation="plane_stress")


def _cantilever_analysis(material: LinearElastic2D, solver=None) -> StaticLinearAnalysis:
    mesh = create_quad_mesh(width=2.0, height=0.4, nx=8, ny=2, material=material, thickness=0.02)
    analysis = StaticLinearAnalysis(mesh, solver=solver)
    for node in mesh.nodes:
        if node.x == 0.0:
            analysis.add_boundary_condition(BoundaryCondition(node.id, X, 0.0))
            analysis.add_boundary_condition(BoundaryCondition(node.id, Y, 0.0))
    for node in mesh.nodes:
        if node.x == 2.0 and node.y == 0.0:
            analysis.add_load(NodalLoad(node.id, Y, -5000.0))
    return analysis


def test_default_solver_is_none_and_preserves_legacy_result(material: LinearElastic2D) -> None:
    analysis = _cantilever_analysis(material)
    result = analysis.solve()

    assert analysis.last_solver_result is None
    assert isinstance(result.displacements, np.ndarray)
    assert result.displacements.ndim == 1


def test_sparse_direct_solver_matches_default_dense_path(material: LinearElastic2D) -> None:
    default_result = _cantilever_analysis(material).solve()
    sparse_analysis = _cantilever_analysis(material, solver=SparseDirectSolver())
    sparse_result = sparse_analysis.solve()

    assert_allclose(default_result.displacements, sparse_result.displacements, atol=1e-8)
    assert_allclose(default_result.reactions, sparse_result.reactions, atol=1e-3)
    assert sparse_analysis.last_solver_result is not None
    assert sparse_analysis.last_solver_result.solver_name == "Sparse Direct"


def test_conjugate_gradient_matches_default_dense_path(material: LinearElastic2D) -> None:
    default_result = _cantilever_analysis(material).solve()
    cg_analysis = _cantilever_analysis(
        material, solver=ConjugateGradientSolver(tolerance=1e-12, max_iterations=2000)
    )
    cg_result = cg_analysis.solve()

    assert_allclose(default_result.displacements, cg_result.displacements, atol=1e-6)
    assert cg_analysis.last_solver_result.iterations is not None
    assert cg_analysis.last_solver_result.iterations > 0


def test_explicit_dense_direct_solver_matches_default(material: LinearElastic2D) -> None:
    default_result = _cantilever_analysis(material).solve()
    explicit_analysis = _cantilever_analysis(material, solver=DenseDirectSolver())
    explicit_result = explicit_analysis.solve()

    assert_allclose(default_result.displacements, explicit_result.displacements)
    assert explicit_analysis.last_solver_result is not None
    assert explicit_analysis.last_solver_result.solver_name == "Dense Direct"


def test_multi_point_constraint_consistent_between_dense_and_sparse_solvers(
    material: LinearElastic2D,
) -> None:
    def build(solver):
        mesh = create_quad_mesh(
            width=2.0, height=0.4, nx=4, ny=2, material=material, thickness=0.02
        )
        analysis = StaticLinearAnalysis(mesh, solver=solver)
        left_nodes = [n for n in mesh.nodes if n.x == 0.0]
        for node in left_nodes:
            analysis.add_boundary_condition(BoundaryCondition(node.id, X, 0.0))
            analysis.add_boundary_condition(BoundaryCondition(node.id, Y, 0.0))
        top_right = max(mesh.nodes, key=lambda n: (n.x, n.y))
        other_node = next(n for n in mesh.nodes if n.x == 2.0 and n.y != top_right.y)
        analysis.add_multi_point_constraint(
            MultiPointConstraint(node_id_a=top_right.id, node_id_b=other_node.id, dof=Y)
        )
        analysis.add_load(NodalLoad(top_right.id, Y, -1000.0))
        return analysis.solve()

    dense_result = build(None)
    sparse_result = build(SparseDirectSolver())

    assert_allclose(dense_result.displacements, sparse_result.displacements, atol=1e-6)


def test_reactions_are_plain_ndarray_regardless_of_solver(material: LinearElastic2D) -> None:
    sparse_analysis = _cantilever_analysis(material, solver=SparseDirectSolver())
    result = sparse_analysis.solve()

    assert isinstance(result.reactions, np.ndarray)
    assert isinstance(result.displacements, np.ndarray)
