"""Tests for StaticLinearAnalysis's Version 27 execution-strategy integration."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from femtoolkit.analysis import BoundaryCondition, NodalLoad, StaticLinearAnalysis, TranslationDOF
from femtoolkit.execution import ExecutionConfig
from femtoolkit.materials import LinearElastic2D
from femtoolkit.mesh.generator import create_quad_mesh
from femtoolkit.solvers import SparseDirectSolver

X = TranslationDOF.X
Y = TranslationDOF.Y


@pytest.fixture
def material() -> LinearElastic2D:
    return LinearElastic2D(youngs_modulus=200e9, poisson_ratio=0.3, formulation="plane_stress")


def _cantilever_analysis(
    material: LinearElastic2D, execution=None, solver=None
) -> StaticLinearAnalysis:
    mesh = create_quad_mesh(width=2.0, height=0.4, nx=6, ny=2, material=material, thickness=0.02)
    analysis = StaticLinearAnalysis(mesh, solver=solver, execution=execution)
    for node in mesh.nodes:
        if node.x == 0.0:
            analysis.add_boundary_condition(BoundaryCondition(node.id, X, 0.0))
            analysis.add_boundary_condition(BoundaryCondition(node.id, Y, 0.0))
    for node in mesh.nodes:
        if node.x == 2.0 and node.y == 0.0:
            analysis.add_load(NodalLoad(node.id, Y, -5000.0))
    return analysis


def test_default_execution_is_none_and_reports_serial(material: LinearElastic2D) -> None:
    analysis = _cantilever_analysis(material)
    analysis.solve()

    report = analysis.last_performance_report
    assert report is not None
    assert report.execution_mode == "serial"
    assert report.workers is None
    assert report.element_time is not None
    assert report.assembly_time is not None
    assert report.solve_time is not None
    assert report.total_time > 0.0


def test_parallel_process_execution_matches_serial_displacements(
    material: LinearElastic2D,
) -> None:
    serial_result = _cantilever_analysis(material).solve()
    parallel_analysis = _cantilever_analysis(
        material, execution=ExecutionConfig(mode="parallel", workers=2, backend="process")
    )
    parallel_result = parallel_analysis.solve()

    assert_allclose(serial_result.displacements, parallel_result.displacements)
    assert parallel_analysis.last_performance_report.execution_mode == "parallel"
    assert parallel_analysis.last_performance_report.workers == 2


def test_parallel_thread_execution_matches_serial_displacements(material: LinearElastic2D) -> None:
    serial_result = _cantilever_analysis(material).solve()
    parallel_analysis = _cantilever_analysis(
        material, execution=ExecutionConfig(mode="parallel", workers=2, backend="thread")
    )
    parallel_result = parallel_analysis.solve()

    assert_allclose(serial_result.displacements, parallel_result.displacements)


def test_parallel_execution_combined_with_sparse_solver(material: LinearElastic2D) -> None:
    dense_result = _cantilever_analysis(material).solve()
    combined_analysis = _cantilever_analysis(
        material,
        execution=ExecutionConfig(mode="parallel", workers=2, backend="process"),
        solver=SparseDirectSolver(),
    )
    combined_result = combined_analysis.solve()

    assert_allclose(dense_result.displacements, combined_result.displacements, atol=1e-8)
    assert combined_analysis.last_solver_result is not None
    assert combined_analysis.last_solver_result.solver_name == "Sparse Direct"
    assert combined_analysis.last_performance_report.execution_mode == "parallel"


def test_reactions_unaffected_by_execution_strategy(material: LinearElastic2D) -> None:
    serial_result = _cantilever_analysis(material).solve()
    parallel_analysis = _cantilever_analysis(
        material, execution=ExecutionConfig(mode="parallel", workers=2, backend="process")
    )
    parallel_result = parallel_analysis.solve()

    assert isinstance(parallel_result.reactions, np.ndarray)
    assert_allclose(serial_result.reactions, parallel_result.reactions, atol=1e-6)
