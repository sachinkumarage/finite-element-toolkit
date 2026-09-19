"""Tests for femtoolkit.solvers (Version 26): dense, sparse-direct, and
Conjugate Gradient solvers, plus the solver-selection mechanism.
"""

import numpy as np
import pytest
import scipy.sparse as sp
from numpy.testing import assert_allclose

from femtoolkit.analysis.assembly import ElementStiffnessContribution, assemble_global_stiffness
from femtoolkit.analysis.boundary_conditions import BoundaryCondition
from femtoolkit.analysis.dof import DOFMap, TranslationDOF
from femtoolkit.analysis.loads import NodalLoad
from femtoolkit.analysis.sparse_assembly import assemble_global_stiffness_sparse
from femtoolkit.analysis.system import LinearSystem, build_force_vector
from femtoolkit.analysis.system import solve as legacy_solve
from femtoolkit.exceptions import (
    InvalidSolverConfigurationError,
    SingularSystemError,
    SolverConvergenceError,
    SolverError,
    UnsupportedSolverError,
)
from femtoolkit.materials import LinearElastic2D
from femtoolkit.mesh.generator import create_quad_mesh
from femtoolkit.solvers import (
    ConjugateGradientSolver,
    DenseDirectSolver,
    SparseDirectSolver,
    create_solver,
)

X = TranslationDOF.X
Y = TranslationDOF.Y


@pytest.fixture
def material() -> LinearElastic2D:
    return LinearElastic2D(youngs_modulus=200e9, poisson_ratio=0.3, formulation="plane_stress")


def _cantilever_system(material: LinearElastic2D, sparse: bool = False):
    """A 6x3 Q4 cantilever mesh, fixed at x=0, loaded at x=2, y=0."""
    mesh = create_quad_mesh(width=2.0, height=1.0, nx=6, ny=3, material=material, thickness=0.01)
    dof_map = DOFMap(node_ids=[n.id for n in mesh.nodes], dofs_per_node=2)
    contributions = [
        ElementStiffnessContribution(e.dof_keys(), e.stiffness_matrix) for e in mesh.elements
    ]
    stiffness = (
        assemble_global_stiffness_sparse(dof_map, contributions)
        if sparse
        else assemble_global_stiffness(dof_map, contributions)
    )

    left_nodes = [n for n in mesh.nodes if n.x == 0.0]
    right_nodes = [n for n in mesh.nodes if n.x == 2.0]
    bcs = [BoundaryCondition(n.id, X, 0.0) for n in left_nodes] + [
        BoundaryCondition(n.id, Y, 0.0) for n in left_nodes
    ]
    loads = [NodalLoad(n.id, Y, -1000.0) for n in right_nodes]
    forces = build_force_vector(dof_map, loads)

    return LinearSystem(
        dof_map=dof_map, stiffness=stiffness, forces=forces, boundary_conditions=bcs
    )


# -- Basic solve correctness against the existing dense solve() ------------


def test_dense_direct_solver_matches_legacy_solve(material: LinearElastic2D) -> None:
    system = _cantilever_system(material, sparse=False)
    legacy = legacy_solve(system)
    result = DenseDirectSolver().solve(system)

    assert_allclose(legacy, result.solution, atol=1e-9)
    assert result.converged
    assert result.iterations is None
    assert result.solver_name == "Dense Direct"
    assert result.solve_time >= 0.0


def test_sparse_direct_solver_matches_legacy_solve(material: LinearElastic2D) -> None:
    dense_system = _cantilever_system(material, sparse=False)
    sparse_system = _cantilever_system(material, sparse=True)
    legacy = legacy_solve(dense_system)
    result = SparseDirectSolver().solve(sparse_system)

    assert_allclose(legacy, result.solution, atol=1e-9)
    assert result.converged
    assert result.iterations is None
    assert result.solver_name == "Sparse Direct"
    assert result.diagnostics["nnz"] > 0
    assert 0 < result.diagnostics["density"] <= 1.0


def test_conjugate_gradient_matches_legacy_solve(material: LinearElastic2D) -> None:
    dense_system = _cantilever_system(material, sparse=False)
    sparse_system = _cantilever_system(material, sparse=True)
    legacy = legacy_solve(dense_system)
    result = ConjugateGradientSolver(tolerance=1e-12, max_iterations=2000).solve(sparse_system)

    assert_allclose(legacy, result.solution, atol=1e-6)
    assert result.converged
    assert result.iterations is not None
    assert result.iterations > 0
    assert result.solver_name == "Conjugate Gradient"


def test_all_three_solvers_agree_with_each_other(material: LinearElastic2D) -> None:
    dense_system = _cantilever_system(material, sparse=False)
    sparse_system = _cantilever_system(material, sparse=True)

    u_dense = DenseDirectSolver().solve(dense_system).solution
    u_sparse = SparseDirectSolver().solve(sparse_system).solution
    u_cg = ConjugateGradientSolver(tolerance=1e-12, max_iterations=2000).solve(
        sparse_system
    ).solution

    assert_allclose(u_dense, u_sparse, atol=1e-8)
    assert_allclose(u_dense, u_cg, atol=1e-6)


# -- Residual computation ---------------------------------------------------


def test_residual_is_small_for_a_well_posed_solve(material: LinearElastic2D) -> None:
    sparse_system = _cantilever_system(material, sparse=True)
    result = SparseDirectSolver().solve(sparse_system)

    assert result.residual_norm < 1e-6
    assert result.relative_residual < 1e-9


# -- Singular system detection ----------------------------------------------


def _mechanism_system(material: LinearElastic2D, sparse: bool):
    """A structure fixed at only one DOF -- a genuine rigid-body mechanism."""
    mesh = create_quad_mesh(width=2.0, height=1.0, nx=6, ny=3, material=material, thickness=0.01)
    dof_map = DOFMap(node_ids=[n.id for n in mesh.nodes], dofs_per_node=2)
    contributions = [
        ElementStiffnessContribution(e.dof_keys(), e.stiffness_matrix) for e in mesh.elements
    ]
    stiffness = (
        assemble_global_stiffness_sparse(dof_map, contributions)
        if sparse
        else assemble_global_stiffness(dof_map, contributions)
    )
    left_nodes = [n for n in mesh.nodes if n.x == 0.0]
    right_nodes = [n for n in mesh.nodes if n.x == 2.0]
    loads = [NodalLoad(n.id, Y, -1000.0) for n in right_nodes]
    forces = build_force_vector(dof_map, loads)
    bcs = [BoundaryCondition(left_nodes[0].id, X, 0.0)]
    return LinearSystem(
        dof_map=dof_map, stiffness=stiffness, forces=forces, boundary_conditions=bcs
    )


def test_dense_direct_solver_detects_singular_system(material: LinearElastic2D) -> None:
    system = _mechanism_system(material, sparse=False)
    with pytest.raises(SingularSystemError):
        DenseDirectSolver().solve(system)


def test_sparse_direct_solver_detects_singular_system(material: LinearElastic2D) -> None:
    system = _mechanism_system(material, sparse=True)
    with pytest.raises(SingularSystemError):
        SparseDirectSolver().solve(system)


# -- Iterative solver: convergence, tolerance, iteration limits ------------


def test_cg_converges_within_reasonable_iterations(material: LinearElastic2D) -> None:
    system = _cantilever_system(material, sparse=True)
    result = ConjugateGradientSolver(tolerance=1e-8, max_iterations=500).solve(system)

    assert result.converged
    assert result.iterations <= 500
    assert result.relative_residual <= 1e-8 * 10  # small margin for the final iteration


def test_cg_raises_on_non_convergence_by_default(material: LinearElastic2D) -> None:
    system = _cantilever_system(material, sparse=True)
    with pytest.raises(SolverConvergenceError):
        ConjugateGradientSolver(tolerance=1e-14, max_iterations=1).solve(system)


def test_cg_returns_non_converged_result_without_raising(material: LinearElastic2D) -> None:
    system = _cantilever_system(material, sparse=True)
    result = ConjugateGradientSolver(
        tolerance=1e-14, max_iterations=1, raise_on_non_convergence=False
    ).solve(system)

    assert not result.converged
    assert result.iterations == 1


def test_cg_rejects_asymmetric_matrix(material: LinearElastic2D) -> None:
    system = _cantilever_system(material, sparse=True)
    asymmetric = system.stiffness.tolil()
    # Pick a pair of genuinely free interior DOFs to break symmetry on.
    free_row, free_col = 10, 11
    asymmetric[free_row, free_col] += 1e10
    system.stiffness = asymmetric.tocsr()

    with pytest.raises(InvalidSolverConfigurationError):
        ConjugateGradientSolver().solve(system)


def test_cg_symmetry_check_can_be_disabled(material: LinearElastic2D) -> None:
    system = _cantilever_system(material, sparse=True)
    result = ConjugateGradientSolver(
        check_symmetry=False, tolerance=1e-10, max_iterations=2000
    ).solve(system)
    assert result.solver_name == "Conjugate Gradient"


def test_cg_rejects_invalid_tolerance() -> None:
    with pytest.raises(InvalidSolverConfigurationError):
        ConjugateGradientSolver(tolerance=0.0)
    with pytest.raises(InvalidSolverConfigurationError):
        ConjugateGradientSolver(tolerance=-1.0)
    with pytest.raises(InvalidSolverConfigurationError):
        ConjugateGradientSolver(tolerance=float("nan"))


def test_cg_rejects_invalid_max_iterations() -> None:
    with pytest.raises(InvalidSolverConfigurationError):
        ConjugateGradientSolver(max_iterations=0)
    with pytest.raises(InvalidSolverConfigurationError):
        ConjugateGradientSolver(max_iterations=-5)


# -- Zero-free-DOF edge case --------------------------------------------------


def test_solvers_handle_fully_prescribed_system(material: LinearElastic2D) -> None:
    mesh = create_quad_mesh(width=1.0, height=1.0, nx=1, ny=1, material=material, thickness=0.01)
    dof_map = DOFMap(node_ids=[n.id for n in mesh.nodes], dofs_per_node=2)
    contributions = [
        ElementStiffnessContribution(e.dof_keys(), e.stiffness_matrix) for e in mesh.elements
    ]
    sparse_stiffness = assemble_global_stiffness_sparse(dof_map, contributions)
    bcs = [BoundaryCondition(n.id, dof, 0.0) for n in mesh.nodes for dof in (X, Y)]
    forces = np.zeros(dof_map.total_dofs)
    system = LinearSystem(
        dof_map=dof_map, stiffness=sparse_stiffness, forces=forces, boundary_conditions=bcs
    )

    for solver in (SparseDirectSolver(), ConjugateGradientSolver()):
        result = solver.solve(system)
        assert result.converged
        assert_allclose(result.solution, np.zeros(dof_map.total_dofs))


# -- Solver-selection mechanism (create_solver) ------------------------------


def test_create_solver_default_is_dense_direct() -> None:
    solver = create_solver()
    assert isinstance(solver, DenseDirectSolver)


def test_create_solver_sparse_direct() -> None:
    solver = create_solver(matrix_type="sparse", solver_type="direct")
    assert isinstance(solver, SparseDirectSolver)


def test_create_solver_conjugate_gradient() -> None:
    solver = create_solver(
        matrix_type="sparse", solver_type="conjugate_gradient", tolerance=1e-9, max_iterations=500
    )
    assert isinstance(solver, ConjugateGradientSolver)
    assert solver.tolerance == pytest.approx(1e-9)
    assert solver.max_iterations == 500


def test_create_solver_rejects_dense_conjugate_gradient() -> None:
    with pytest.raises(UnsupportedSolverError):
        create_solver(matrix_type="dense", solver_type="conjugate_gradient")


def test_create_solver_rejects_unknown_matrix_type() -> None:
    with pytest.raises(UnsupportedSolverError):
        create_solver(matrix_type="bogus", solver_type="direct")


def test_create_solver_rejects_unknown_solver_type() -> None:
    with pytest.raises(UnsupportedSolverError):
        create_solver(matrix_type="dense", solver_type="bogus")


# -- MATRIX_TYPE attributes ---------------------------------------------------


def test_matrix_type_attributes() -> None:
    assert DenseDirectSolver.MATRIX_TYPE == "dense"
    assert SparseDirectSolver.MATRIX_TYPE == "sparse"
    assert ConjugateGradientSolver.MATRIX_TYPE == "sparse"


# -- Pre-solve validation -----------------------------------------------------


def test_solve_rejects_nan_in_stiffness(material: LinearElastic2D) -> None:
    system = _cantilever_system(material, sparse=True)
    corrupted = system.stiffness.tolil()
    corrupted[0, 0] = np.nan
    system.stiffness = corrupted.tocsr()

    with pytest.raises(SolverError):
        SparseDirectSolver().solve(system)


def test_solve_rejects_nan_in_forces(material: LinearElastic2D) -> None:
    system = _cantilever_system(material, sparse=True)
    system.forces[0] = np.nan

    with pytest.raises(SolverError):
        SparseDirectSolver().solve(system)


def test_sparse_matrix_type_check() -> None:
    dof_map = DOFMap(node_ids=[1, 2], dofs_per_node=1)
    stiffness = sp.csr_matrix(np.eye(2))
    forces = np.array([1.0, 0.0])
    bcs = [BoundaryCondition(1, X, 0.0)]
    system = LinearSystem(
        dof_map=dof_map, stiffness=stiffness, forces=forces, boundary_conditions=bcs
    )
    result = SparseDirectSolver().solve(system)
    assert result.converged
