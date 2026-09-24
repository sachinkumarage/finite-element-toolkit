"""Tests for femtoolkit.runs.manager."""

from femtoolkit.application.project import BoundaryConditionConfig, LoadConfig, Project
from femtoolkit.runs.manager import SimulationRunManager
from femtoolkit.runs.models import RunStatus
from femtoolkit.verification.status import VerificationStatus


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


def test_execute_completed_run_populates_everything() -> None:
    project = _mechanical_project()
    run = SimulationRunManager().execute(project)

    assert run.status == RunStatus.COMPLETED
    assert run.succeeded
    assert run.project_id == project.project_id
    assert run.scenario_id is None
    assert run.started_at is not None
    assert run.completed_at is not None
    assert run.execution_time_seconds is not None
    assert run.execution_time_seconds >= 0.0
    assert run.result is not None
    assert run.result.summary.maximum_displacement is not None
    assert run.verification_status == VerificationStatus.PASS
    assert run.reproducibility_metadata is not None
    assert run.reproducibility_metadata.model_name == project.name
    assert run.error_stage is None
    assert run.error_message is None


def test_execute_snapshot_is_immune_to_later_project_mutation() -> None:
    project = _mechanical_project()
    run = SimulationRunManager().execute(project)

    original_nx = run.configuration_snapshot.mesh.nx
    project.mesh.nx = 999
    project.name = "Mutated"

    assert run.configuration_snapshot.mesh.nx == original_nx
    assert run.configuration_snapshot.name == "Beam"


def test_execute_scenario_id_is_carried_through() -> None:
    project = _mechanical_project()
    run = SimulationRunManager().execute(project, scenario_id="scenario-1")
    assert run.scenario_id == "scenario-1"


def test_execute_invalid_project_fails_at_validation_stage() -> None:
    project = Project(name="Incomplete", analysis_type="linear_static")
    run = SimulationRunManager().execute(project)

    assert run.status == RunStatus.FAILED
    assert run.failed
    assert run.error_stage == "validation"
    assert run.error_message
    assert run.verification_status == VerificationStatus.NOT_RUN
    assert run.reproducibility_metadata is None


def test_execute_invalid_project_still_records_result_errors() -> None:
    project = Project(name="Incomplete", analysis_type="linear_static")
    run = SimulationRunManager().execute(project)

    assert run.result is not None
    assert run.result.status == "invalid"
    assert run.result.errors
