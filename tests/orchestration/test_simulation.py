"""Tests for femtoolkit.orchestration.simulation (Version 34)."""

from __future__ import annotations

from femtoolkit.application.project import BoundaryConditionConfig, LoadConfig, Project
from femtoolkit.orchestration.config import OrchestrationConfig
from femtoolkit.orchestration.simulation import (
    SimulationTask,
    evaluate_simulation_batch,
    execute_simulation_task,
)
from femtoolkit.runs.models import RunStatus
from femtoolkit.studies.extractors import get_extractor


def _base_project() -> Project:
    project = Project(name="Cantilever", analysis_type="linear_static")
    project.material.youngs_modulus = 200e9
    project.material.poisson_ratio = 0.3
    project.mesh.width, project.mesh.height = 2.0, 0.4
    project.mesh.nx, project.mesh.ny, project.mesh.thickness = 6, 2, 0.02
    project.boundary_conditions = [
        BoundaryConditionConfig(region="left", dof="X", value=0.0),
        BoundaryConditionConfig(region="left", dof="Y", value=0.0),
    ]
    project.loads = [LoadConfig(region="right", dof="Y", magnitude=-1000.0)]
    return project


def _tasks(loads: list[float]) -> list[SimulationTask]:
    tasks = []
    for i, load in enumerate(loads):
        project = _base_project()
        project.loads = [LoadConfig(region="right", dof="Y", magnitude=load)]
        scenario_id = f"scenario-{i}"
        tasks.append(SimulationTask(task_id=scenario_id, project=project, scenario_id=scenario_id))
    return tasks


def test_execute_simulation_task_returns_completed_run() -> None:
    task = SimulationTask(task_id="t", project=_base_project(), scenario_id="s")
    run = execute_simulation_task(task)
    assert run.status is RunStatus.COMPLETED
    assert run.scenario_id == "s"


def test_evaluate_simulation_batch_serial_default() -> None:
    runs, summary = evaluate_simulation_batch(_tasks([-1000.0, -2000.0, -3000.0]))
    assert [r.scenario_id for r in runs] == ["scenario-0", "scenario-1", "scenario-2"]
    assert all(r.status is RunStatus.COMPLETED for r in runs)
    assert summary.execution_mode == "serial"


def test_evaluate_simulation_batch_parallel_matches_serial() -> None:
    loads = [-1000.0, -2000.0, -3000.0, -4000.0]
    serial_runs, _ = evaluate_simulation_batch(_tasks(loads))
    parallel_runs, summary = evaluate_simulation_batch(
        _tasks(loads), config=OrchestrationConfig(execution_mode="parallel", max_workers=2)
    )
    assert summary.execution_mode == "parallel"
    assert [r.scenario_id for r in parallel_runs] == [r.scenario_id for r in serial_runs]

    extractor = get_extractor("maximum_displacement")
    serial_values = [extractor(r) for r in serial_runs]
    parallel_values = [extractor(r) for r in parallel_runs]
    assert serial_values == parallel_values


def test_evaluate_simulation_batch_identity_preserved_with_completion_order() -> None:
    # preserve_order=False returns completion order, not submission order --
    # identity must still be recoverable via scenario_id regardless.
    loads = [-1000.0, -2000.0, -3000.0, -4000.0]
    runs, _summary = evaluate_simulation_batch(
        _tasks(loads),
        config=OrchestrationConfig(execution_mode="parallel", max_workers=4, preserve_order=False),
    )
    assert {r.scenario_id for r in runs} == {f"scenario-{i}" for i in range(4)}
    assert all(r.status is RunStatus.COMPLETED for r in runs)
