"""Tests for femtoolkit.application.project and .project_service."""

import pytest

from femtoolkit.application.project import (
    BoundaryConditionConfig,
    ExecutionSettingsConfig,
    LoadConfig,
    MaterialConfig,
    MeshConfig,
    Project,
    SolverConfig,
)
from femtoolkit.application.project_service import ProjectService
from femtoolkit.exceptions import ValidationError


def test_create_project_defaults() -> None:
    service = ProjectService()
    project = service.create_project("Cantilever Beam")

    assert project.name == "Cantilever Beam"
    assert project.analysis_type == "linear_static"
    assert isinstance(project.material, MaterialConfig)
    assert isinstance(project.mesh, MeshConfig)
    assert project.boundary_conditions == []
    assert project.loads == []
    assert isinstance(project.solver, SolverConfig)
    assert isinstance(project.execution, ExecutionSettingsConfig)
    assert project.execution.mode == "serial"


def test_create_project_blank_name_raises() -> None:
    service = ProjectService()
    with pytest.raises(ValidationError):
        service.create_project("   ")


def test_create_project_custom_analysis_type() -> None:
    service = ProjectService()
    project = service.create_project("Heated Plate", "thermal_steady_state")
    assert project.analysis_type == "thermal_steady_state"


def test_reset_returns_default_project() -> None:
    service = ProjectService()
    project = service.reset()
    assert project.name == "Untitled Project"
    assert project.analysis_type == "linear_static"


def test_to_dict_from_dict_round_trip() -> None:
    project = Project(
        name="Round Trip",
        analysis_type="linear_static",
        material=MaterialConfig(name="Steel", youngs_modulus=200e9, poisson_ratio=0.3),
        mesh=MeshConfig(width=1.5, height=0.3, nx=6, ny=3),
        boundary_conditions=[BoundaryConditionConfig(region="left", dof="X", value=0.0)],
        loads=[LoadConfig(region="right", dof="Y", magnitude=-100.0)],
        solver=SolverConfig(tolerance=1e-5, max_iterations=10),
        execution=ExecutionSettingsConfig(mode="parallel", workers=4, batch_size=25),
    )
    restored = Project.from_dict(project.to_dict())

    assert restored.name == project.name
    assert restored.analysis_type == project.analysis_type
    assert restored.material == project.material
    assert restored.mesh == project.mesh
    assert restored.boundary_conditions == project.boundary_conditions
    assert restored.loads == project.loads
    assert restored.solver == project.solver
    assert restored.execution == project.execution


def test_from_dict_without_execution_key_defaults_to_serial() -> None:
    """A project saved by Version 26 (before `execution` existed) must still load cleanly."""
    data = Project().to_dict()
    del data["execution"]
    restored = Project.from_dict(data)
    assert restored.execution == ExecutionSettingsConfig()


def test_from_dict_ignores_unknown_keys() -> None:
    data = Project().to_dict()
    data["future_field"] = "should be ignored"
    restored = Project.from_dict(data)
    assert restored.name == "Untitled Project"


def test_project_service_json_round_trip() -> None:
    service = ProjectService()
    project = service.create_project("JSON Round Trip", "thermal_steady_state")
    project.boundary_conditions.append(
        BoundaryConditionConfig(region="left", dof="TEMPERATURE", value=373.15)
    )

    text = service.to_json(project)
    restored = service.from_json(text)

    assert restored.name == project.name
    assert restored.boundary_conditions == project.boundary_conditions


def test_project_service_from_json_invalid_json_raises() -> None:
    service = ProjectService()
    with pytest.raises(ValidationError):
        service.from_json("{not valid json")


def test_project_service_save_and_load(tmp_path) -> None:
    service = ProjectService()
    project = service.create_project("Saved Project")
    path = service.save(project, tmp_path / "subdir" / "project.json")

    assert path.exists()
    loaded = service.load(path)
    assert loaded.name == project.name


def test_project_service_load_missing_file_raises(tmp_path) -> None:
    service = ProjectService()
    with pytest.raises(ValidationError):
        service.load(tmp_path / "does_not_exist.json")


def test_from_dict_rejects_future_format_version() -> None:
    data = Project().to_dict()
    data["format_version"] = 999
    with pytest.raises(ValidationError):
        Project.from_dict(data)


def test_from_dict_accepts_current_format_version() -> None:
    data = Project().to_dict()
    restored = Project.from_dict(data)
    assert restored.format_version == data["format_version"]


def test_project_id_is_unique_per_instance() -> None:
    assert Project().project_id != Project().project_id


def test_project_id_round_trips_through_to_dict_from_dict() -> None:
    project = Project(name="Has An Id")
    restored = Project.from_dict(project.to_dict())
    assert restored.project_id == project.project_id


def test_from_dict_without_project_id_key_generates_one() -> None:
    """A project saved before Version 30 (no `project_id` field) must still load cleanly."""
    data = Project().to_dict()
    del data["project_id"]
    restored = Project.from_dict(data)
    assert restored.project_id
