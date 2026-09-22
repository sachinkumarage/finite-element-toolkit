"""Tests for femtoolkit.application.simulation_service."""

from unittest.mock import patch

import pytest

from femtoolkit.application.model_service import ModelService
from femtoolkit.application.project import BoundaryConditionConfig, LoadConfig, Project
from femtoolkit.application.simulation_service import (
    RUN_STATUS_COMPLETED,
    RUN_STATUS_FAILED,
    RUN_STATUS_INVALID,
    SimulationService,
)
from femtoolkit.exceptions import ValidationError


def _mechanical_project() -> Project:
    project = Project(name="Beam", analysis_type="linear_static")
    project.material.youngs_modulus = 200e9
    project.material.poisson_ratio = 0.3
    project.mesh.width = 2.0
    project.mesh.height = 0.4
    project.mesh.nx = 4
    project.mesh.ny = 2
    project.mesh.thickness = 0.02
    project.boundary_conditions = [
        BoundaryConditionConfig(region="left", dof="X", value=0.0),
        BoundaryConditionConfig(region="left", dof="Y", value=0.0),
    ]
    project.loads = [LoadConfig(region="right", dof="Y", magnitude=-500.0)]
    return project


def _thermal_project() -> Project:
    project = Project(name="Plate", analysis_type="thermal_steady_state")
    project.material.thermal_conductivity = 45.0
    project.material.specific_heat = 460.0
    project.material.density = 7850.0
    project.mesh.width = 1.0
    project.mesh.height = 1.0
    project.mesh.nx = 4
    project.mesh.ny = 4
    project.boundary_conditions = [
        BoundaryConditionConfig(region="left", dof="TEMPERATURE", value=373.15),
        BoundaryConditionConfig(region="right", dof="TEMPERATURE", value=293.15),
    ]
    return project


def test_run_linear_static_completes() -> None:
    result = SimulationService().run(_mechanical_project())

    assert result.status == RUN_STATUS_COMPLETED
    assert result.succeeded
    assert result.errors == []
    assert result.simulation is not None
    assert result.summary.maximum_von_mises_stress is not None
    assert result.summary.maximum_displacement is not None


def test_run_thermal_steady_state_completes() -> None:
    result = SimulationService().run(_thermal_project())

    assert result.status == RUN_STATUS_COMPLETED
    assert result.summary.maximum_temperature == pytest.approx(373.15)
    assert result.summary.minimum_temperature == pytest.approx(293.15)


def test_run_invalid_project_never_solves() -> None:
    project = _mechanical_project()
    project.boundary_conditions = []

    result = SimulationService().run(project)

    assert result.status == RUN_STATUS_INVALID
    assert not result.succeeded
    assert result.simulation is None
    assert any("boundary conditions" in error for error in result.errors)


def test_run_reports_toolkit_error_as_failed() -> None:
    project = _mechanical_project()
    service = SimulationService()

    with patch.object(
        ModelService, "build_mesh", side_effect=ValidationError("synthetic solver failure")
    ):
        result = service.run(project)

    assert result.status == RUN_STATUS_FAILED
    assert not result.succeeded
    assert result.simulation is None
    assert any("Solver error" in error for error in result.errors)


def test_run_reports_unexpected_error_as_failed() -> None:
    project = _mechanical_project()
    service = SimulationService()

    with patch.object(ModelService, "build_mesh", side_effect=RuntimeError("boom")):
        result = service.run(project)

    assert result.status == RUN_STATUS_FAILED
    assert any("Unexpected error" in error for error in result.errors)


def test_run_unavailable_analysis_type_is_invalid() -> None:
    project = _mechanical_project()
    project.analysis_type = "nonlinear_static"

    result = SimulationService().run(project)

    assert result.status == RUN_STATUS_INVALID
    assert any("not yet available" in error for error in result.errors)


def test_run_default_solver_has_no_diagnostics() -> None:
    result = SimulationService().run(_mechanical_project())
    assert result.solver_diagnostics is None


def test_run_sparse_direct_populates_diagnostics() -> None:
    project = _mechanical_project()
    project.solver.matrix_type = "sparse"
    project.solver.solver_type = "direct"

    result = SimulationService().run(project)

    assert result.succeeded
    assert result.solver_diagnostics is not None
    assert result.solver_diagnostics.solver_name == "Sparse Direct"
    assert result.solver_diagnostics.converged


def test_run_conjugate_gradient_populates_diagnostics() -> None:
    project = _mechanical_project()
    project.solver.matrix_type = "sparse"
    project.solver.solver_type = "conjugate_gradient"
    project.solver.tolerance = 1e-10
    project.solver.max_iterations = 1000

    result = SimulationService().run(project)

    assert result.succeeded
    assert result.solver_diagnostics is not None
    assert result.solver_diagnostics.solver_name == "Conjugate Gradient"
    assert result.solver_diagnostics.iterations is not None


def test_run_sparse_matches_dense_summary() -> None:
    dense_result = SimulationService().run(_mechanical_project())

    sparse_project = _mechanical_project()
    sparse_project.solver.matrix_type = "sparse"
    sparse_project.solver.solver_type = "direct"
    sparse_result = SimulationService().run(sparse_project)

    assert dense_result.summary.maximum_displacement == pytest.approx(
        sparse_result.summary.maximum_displacement, rel=1e-6
    )
    assert dense_result.summary.maximum_von_mises_stress == pytest.approx(
        sparse_result.summary.maximum_von_mises_stress, rel=1e-6
    )


def test_run_invalid_solver_combo_is_invalid() -> None:
    project = _mechanical_project()
    project.solver.matrix_type = "dense"
    project.solver.solver_type = "conjugate_gradient"

    result = SimulationService().run(project)

    assert result.status == RUN_STATUS_INVALID
    assert any("Conjugate Gradient" in error for error in result.errors)


def test_run_thermal_sparse_matches_dense_summary() -> None:
    dense_result = SimulationService().run(_thermal_project())

    sparse_project = _thermal_project()
    sparse_project.solver.matrix_type = "sparse"
    sparse_project.solver.solver_type = "direct"
    sparse_result = SimulationService().run(sparse_project)

    assert dense_result.summary.maximum_temperature == pytest.approx(
        sparse_result.summary.maximum_temperature, rel=1e-6
    )


def test_run_default_execution_populates_serial_performance_report() -> None:
    result = SimulationService().run(_mechanical_project())

    assert result.succeeded
    assert result.performance_report is not None
    assert result.performance_report.execution_mode == "serial"
    assert result.performance_report.total_time > 0.0


def test_run_parallel_execution_populates_performance_report() -> None:
    project = _mechanical_project()
    project.execution.mode = "parallel"
    project.execution.workers = 2

    result = SimulationService().run(project)

    assert result.succeeded
    assert result.performance_report is not None
    assert result.performance_report.execution_mode == "parallel"
    assert result.performance_report.workers == 2


def test_run_parallel_execution_matches_serial_summary() -> None:
    serial_result = SimulationService().run(_mechanical_project())

    parallel_project = _mechanical_project()
    parallel_project.execution.mode = "parallel"
    parallel_project.execution.workers = 2
    parallel_result = SimulationService().run(parallel_project)

    assert serial_result.summary.maximum_displacement == pytest.approx(
        parallel_result.summary.maximum_displacement, rel=1e-9
    )


def test_run_thermal_parallel_execution_matches_serial_summary() -> None:
    serial_result = SimulationService().run(_thermal_project())

    parallel_project = _thermal_project()
    parallel_project.execution.mode = "parallel"
    parallel_project.execution.workers = 2
    parallel_result = SimulationService().run(parallel_project)

    assert serial_result.summary.maximum_temperature == pytest.approx(
        parallel_result.summary.maximum_temperature, rel=1e-9
    )


def test_run_invalid_execution_mode_is_invalid() -> None:
    project = _mechanical_project()
    project.execution.mode = "bogus"

    result = SimulationService().run(project)

    assert result.status == RUN_STATUS_INVALID
    assert any("mode" in error for error in result.errors)
