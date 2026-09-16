"""Tests for femtoolkit.application.project and .project_service."""

import pytest

from femtoolkit.application.project import (
    BoundaryConditionConfig,
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
    )
    restored = Project.from_dict(project.to_dict())

    assert restored.name == project.name
    assert restored.analysis_type == project.analysis_type
    assert restored.material == project.material
    assert restored.mesh == project.mesh
    assert restored.boundary_conditions == project.boundary_conditions
    assert restored.loads == project.loads
    assert restored.solver == project.solver


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
