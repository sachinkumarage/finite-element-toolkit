"""Tests for femtoolkit.studies.comparison."""

import pytest

from femtoolkit.application.project import BoundaryConditionConfig, LoadConfig, Project
from femtoolkit.exceptions import ValidationError
from femtoolkit.runs.manager import SimulationRunManager
from femtoolkit.studies.comparison import (
    absolute_difference,
    compare_runs,
    percentage_change,
    relative_difference,
)


def _project_with_load(magnitude: float) -> Project:
    project = Project(name=f"Beam-{magnitude}", analysis_type="linear_static")
    project.material.youngs_modulus = 200e9
    project.material.poisson_ratio = 0.3
    project.mesh.width, project.mesh.height = 2.0, 0.4
    project.mesh.nx, project.mesh.ny, project.mesh.thickness = 8, 2, 0.02
    project.boundary_conditions = [
        BoundaryConditionConfig(region="left", dof="X", value=0.0),
        BoundaryConditionConfig(region="left", dof="Y", value=0.0),
    ]
    project.loads = [LoadConfig(region="right", dof="Y", magnitude=magnitude)]
    return project


def test_absolute_relative_percentage_are_signed() -> None:
    assert absolute_difference(10.0, 15.0) == 5.0
    assert absolute_difference(10.0, 5.0) == -5.0
    assert relative_difference(10.0, 15.0) == pytest.approx(0.5)
    assert relative_difference(10.0, 5.0) == pytest.approx(-0.5)
    assert percentage_change(10.0, 15.0) == pytest.approx(50.0)
    assert percentage_change(10.0, 5.0) == pytest.approx(-50.0)


def test_relative_difference_near_zero_baseline_uses_epsilon_floor() -> None:
    value = relative_difference(0.0, 1.0, epsilon=1e-6)
    assert value == pytest.approx(1.0 / 1e-6)


def test_compare_runs_linear_elastic_proportionality() -> None:
    manager = SimulationRunManager()
    runs = [manager.execute(_project_with_load(-1000.0 * k)) for k in (1, 2, 3)]

    result = compare_runs(runs, lambda run: run.result.summary.maximum_displacement, "Displacement")

    assert result.baseline_run_id == runs[0].run_id
    assert len(result.entries) == 2
    assert result.entries[0].percentage_change == pytest.approx(100.0, rel=1e-6)
    assert result.entries[1].percentage_change == pytest.approx(200.0, rel=1e-6)
    assert result.entries[0].absolute_difference > 0


def test_compare_runs_requires_at_least_two_runs() -> None:
    manager = SimulationRunManager()
    run = manager.execute(_project_with_load(-1000.0))
    with pytest.raises(ValidationError):
        compare_runs([run], lambda r: r.result.summary.maximum_displacement, "Displacement")


def test_compare_runs_missing_quantity_raises() -> None:
    manager = SimulationRunManager()
    runs = [manager.execute(_project_with_load(-1000.0 * k)) for k in (1, 2)]
    with pytest.raises(ValidationError):
        compare_runs(runs, lambda r: r.result.summary.maximum_temperature, "Temperature")
