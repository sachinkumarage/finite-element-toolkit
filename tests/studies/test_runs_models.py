"""Tests for femtoolkit.runs.models."""

from femtoolkit.application.project import MeshConfig, Project
from femtoolkit.runs.models import RunStatus, SimulationRun, new_run_id, snapshot_configuration
from femtoolkit.verification.status import VerificationStatus


def test_new_run_id_is_unique() -> None:
    assert new_run_id() != new_run_id()


def test_snapshot_configuration_is_independent_copy() -> None:
    project = Project(name="Original", mesh=MeshConfig(nx=10))
    snapshot = snapshot_configuration(project)

    assert snapshot is not project
    assert snapshot.mesh is not project.mesh
    assert snapshot.mesh.nx == 10

    project.mesh.nx = 999
    assert snapshot.mesh.nx == 10


def test_simulation_run_defaults() -> None:
    project = Project(name="Defaults")
    run = SimulationRun(
        run_id="run-1", project_id=project.project_id, configuration_snapshot=project
    )

    assert run.status == RunStatus.PENDING
    assert run.scenario_id is None
    assert run.verification_status == VerificationStatus.NOT_RUN
    assert run.validation_result is None
    assert run.result is None
    assert run.error_stage is None
    assert not run.succeeded
    assert not run.failed


def test_simulation_run_succeeded_and_failed_properties() -> None:
    project = Project(name="Status Check")
    completed = SimulationRun(
        run_id="run-completed",
        project_id=project.project_id,
        configuration_snapshot=project,
        status=RunStatus.COMPLETED,
    )
    failed = SimulationRun(
        run_id="run-failed",
        project_id=project.project_id,
        configuration_snapshot=project,
        status=RunStatus.FAILED,
    )

    assert completed.succeeded
    assert not completed.failed
    assert failed.failed
    assert not failed.succeeded
