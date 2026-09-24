"""Tests for femtoolkit.studies.extractors."""

import pytest

from femtoolkit.application.project import BoundaryConditionConfig, LoadConfig, Project
from femtoolkit.exceptions import ValidationError
from femtoolkit.runs.manager import SimulationRunManager
from femtoolkit.studies.extractors import (
    EXTRACTORS,
    execution_time,
    get_extractor,
    maximum_displacement,
    maximum_von_mises_stress,
)


def _mechanical_project() -> Project:
    project = Project(name="Beam", analysis_type="linear_static")
    project.material.youngs_modulus = 200e9
    project.material.poisson_ratio = 0.3
    project.mesh.width, project.mesh.height = 2.0, 0.4
    project.mesh.nx, project.mesh.ny, project.mesh.thickness = 4, 2, 0.02
    project.boundary_conditions = [
        BoundaryConditionConfig(region="left", dof="X", value=0.0),
        BoundaryConditionConfig(region="left", dof="Y", value=0.0),
    ]
    project.loads = [LoadConfig(region="right", dof="Y", magnitude=-500.0)]
    return project


def test_extractors_on_completed_run() -> None:
    run = SimulationRunManager().execute(_mechanical_project())

    assert maximum_displacement(run) == run.result.summary.maximum_displacement
    assert maximum_von_mises_stress(run) == run.result.summary.maximum_von_mises_stress
    assert execution_time(run) == run.execution_time_seconds
    assert maximum_displacement(run) is not None


def test_extractors_on_failed_run_return_none() -> None:
    run = SimulationRunManager().execute(Project(name="Incomplete"))

    assert maximum_displacement(run) is None
    assert maximum_von_mises_stress(run) is None


def test_execution_time_is_available_even_for_failed_run() -> None:
    run = SimulationRunManager().execute(Project(name="Incomplete"))
    assert execution_time(run) is not None


def test_get_extractor_returns_registered_extractor() -> None:
    assert get_extractor("maximum_displacement") is maximum_displacement


def test_get_extractor_unknown_name_raises() -> None:
    with pytest.raises(ValidationError):
        get_extractor("not_a_real_quantity")


def test_every_extractor_name_is_callable_and_registered() -> None:
    for name, extractor in EXTRACTORS.items():
        assert get_extractor(name) is extractor
