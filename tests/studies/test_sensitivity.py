"""Tests for femtoolkit.studies.sensitivity."""

import pytest

from femtoolkit.application.project import BoundaryConditionConfig, LoadConfig, Project
from femtoolkit.exceptions import ValidationError
from femtoolkit.runs.manager import SimulationRunManager
from femtoolkit.studies.sensitivity import compute_sensitivity, compute_sensitivity_series


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


def test_compute_sensitivity_basic_ratio() -> None:
    result = compute_sensitivity(10.0, 20.0, 100.0, 300.0)
    # dp/p = 1.0, dy/y = 2.0 -> S = 2.0
    assert result.sensitivity == pytest.approx(2.0)


def test_compute_sensitivity_raises_for_equal_parameter_values() -> None:
    with pytest.raises(ValidationError):
        compute_sensitivity(10.0, 10.0, 100.0, 200.0)


def test_compute_sensitivity_near_zero_reference_value_uses_epsilon() -> None:
    result = compute_sensitivity(0.0, 1.0, 0.0, 1.0, epsilon=1e-6)
    assert result.sensitivity == pytest.approx(1e6 / 1e6)


def test_compute_sensitivity_series_length_and_validation() -> None:
    with pytest.raises(ValidationError):
        compute_sensitivity_series([1.0], [1.0])
    with pytest.raises(ValidationError):
        compute_sensitivity_series([1.0, 2.0], [1.0])

    series = compute_sensitivity_series([1.0, 2.0, 4.0], [10.0, 20.0, 40.0])
    assert len(series) == 2
    assert all(entry.sensitivity == pytest.approx(1.0) for entry in series)


def test_sensitivity_of_linear_elastic_tip_displacement_to_load_is_unity() -> None:
    manager = SimulationRunManager()
    load_magnitudes = [1000.0, 2000.0, 3000.0, 4000.0]
    runs = [manager.execute(_project_with_load(-magnitude)) for magnitude in load_magnitudes]
    # EngineeringSummary.maximum_displacement is an unsigned vector magnitude, so it is
    # compared against the load's unsigned magnitude too, not its signed (downward) value.
    displacements = [run.result.summary.maximum_displacement for run in runs]

    series = compute_sensitivity_series(
        load_magnitudes,
        displacements,
        parameter_label="load magnitude",
        quantity_label="tip displacement",
    )

    assert len(series) == 3
    for entry in series:
        assert entry.sensitivity == pytest.approx(1.0, rel=1e-6)
