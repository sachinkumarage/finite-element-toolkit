"""Tests for femtoolkit.runs.history."""

import pytest

from femtoolkit.application.project import BoundaryConditionConfig, LoadConfig, Project
from femtoolkit.exceptions import ValidationError
from femtoolkit.runs.history import RunHistory, RunRecord, record_from_run
from femtoolkit.runs.manager import SimulationRunManager
from femtoolkit.runs.models import RunStatus


def _mechanical_project(name: str = "Beam") -> Project:
    project = Project(name=name, analysis_type="linear_static")
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


def test_record_from_run_captures_key_results_and_no_full_result() -> None:
    project = _mechanical_project()
    run = SimulationRunManager().execute(project)
    record = record_from_run(run)

    assert record.run_id == run.run_id
    assert record.project_id == project.project_id
    assert record.project_name == "Beam"
    assert record.status == "completed"
    assert "maximum_displacement" in record.key_results
    assert "maximum_von_mises_stress" in record.key_results
    assert record.configuration_snapshot["name"] == "Beam"
    assert not hasattr(record, "result")


def test_record_from_run_failed_run_has_no_key_results() -> None:
    run = SimulationRunManager().execute(Project(name="Incomplete"))
    record = record_from_run(run)

    assert record.status == "failed"
    assert record.key_results == {}
    assert record.error_stage == "validation"
    assert record.error_message


def test_run_record_to_dict_from_dict_round_trip() -> None:
    project = _mechanical_project()
    run = SimulationRunManager().execute(project)
    record = record_from_run(run)

    restored = RunRecord.from_dict(record.to_dict())
    assert restored == record


def test_run_history_add_and_query() -> None:
    project_a = _mechanical_project("Project A")
    project_b = _mechanical_project("Project B")
    manager = SimulationRunManager()

    run_a1 = manager.execute(project_a, scenario_id="s1")
    run_a2 = manager.execute(project_a, scenario_id="s2")
    run_b1 = manager.execute(project_b)

    history = RunHistory()
    for run in (run_a1, run_a2, run_b1):
        history.add(record_from_run(run))

    assert len(history) == 3
    assert len(history.by_project_id(project_a.project_id)) == 2
    assert len(history.by_project_id(project_b.project_id)) == 1
    assert history.by_run_id(run_a1.run_id).scenario_id == "s1"
    assert history.by_run_id("does-not-exist") is None
    assert len(history.by_scenario_id("s1")) == 1
    assert len(history.by_status(RunStatus.COMPLETED)) == 3
    assert len(history.by_status("failed")) == 0


def test_run_history_json_round_trip() -> None:
    run = SimulationRunManager().execute(_mechanical_project())
    history = RunHistory()
    history.add(record_from_run(run))

    restored = RunHistory.from_json(history.to_json())
    assert len(restored) == 1
    assert restored.all()[0].run_id == run.run_id


def test_run_history_from_json_invalid_json_raises() -> None:
    with pytest.raises(ValidationError):
        RunHistory.from_json("{not valid json")


def test_run_history_save_and_load(tmp_path) -> None:
    run = SimulationRunManager().execute(_mechanical_project())
    history = RunHistory()
    history.add(record_from_run(run))

    path = history.save(tmp_path / "subdir" / "history.json")
    assert path.exists()

    loaded = RunHistory.load(path)
    assert len(loaded) == 1
    assert loaded.all()[0].run_id == run.run_id


def test_run_history_load_missing_file_raises(tmp_path) -> None:
    with pytest.raises(ValidationError):
        RunHistory.load(tmp_path / "does_not_exist.json")
