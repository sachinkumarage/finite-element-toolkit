"""Tests for femtoolkit.digital_twin.updating."""

from __future__ import annotations

import pytest

from femtoolkit.application.project import BoundaryConditionConfig, LoadConfig, Project
from femtoolkit.application.simulation_service import SimulationService
from femtoolkit.digital_twin.measurements import MeasurementData, MeasurementPoint
from femtoolkit.digital_twin.parameters import ModelParameter
from femtoolkit.digital_twin.problem import ModelUpdateProblem
from femtoolkit.digital_twin.results import ModelUpdateStatus
from femtoolkit.digital_twin.updating import ModelUpdateConfig, run_model_update


def _cantilever_project(youngs_modulus: float = 200e9) -> Project:
    project = Project(name="DT Updating Cantilever", analysis_type="linear_static")
    project.material.youngs_modulus = youngs_modulus
    project.material.poisson_ratio = 0.3
    project.material.density = 7850.0
    project.mesh.width = 2.0
    project.mesh.height = 0.4
    project.mesh.nx = 6
    project.mesh.ny = 2
    project.mesh.thickness = 0.01
    project.boundary_conditions = [
        BoundaryConditionConfig(region="left", dof="X", value=0.0),
        BoundaryConditionConfig(region="left", dof="Y", value=0.0),
    ]
    project.loads = [LoadConfig(region="right", dof="Y", magnitude=-4000.0)]
    return project


def _measured_displacement_at(youngs_modulus: float) -> float:
    result = SimulationService().run(_cantilever_project(youngs_modulus))
    assert result.succeeded
    return result.summary.maximum_displacement


def test_run_model_update_with_no_updatable_parameters_is_invalid() -> None:
    fixed_parameter = ModelParameter(
        name="youngs_modulus", current_value=200e9, lower_bound=150e9, upper_bound=250e9,
        path="material.youngs_modulus", updatable=False,
    )
    measurement_data = MeasurementData(quantity_name="maximum_displacement")
    measurement_data.add_point(MeasurementPoint(measurement_id="m1", measured_value=0.004))
    problem = ModelUpdateProblem(
        base_project=_cantilever_project(), parameters=[fixed_parameter],
        measurement_data=measurement_data,
    )
    result = run_model_update(problem)
    assert result.status is ModelUpdateStatus.INVALID
    assert result.updated_parameters == result.initial_parameters


def test_run_model_update_fails_with_a_singular_baseline_project() -> None:
    unconstrained_project = Project(name="Unconstrained", analysis_type="linear_static")
    unconstrained_project.material.youngs_modulus = 200e9
    unconstrained_project.material.poisson_ratio = 0.3
    unconstrained_project.material.density = 7850.0
    unconstrained_project.mesh.width = 2.0
    unconstrained_project.mesh.height = 0.4
    unconstrained_project.mesh.nx = 4
    unconstrained_project.mesh.ny = 2
    unconstrained_project.mesh.thickness = 0.01
    unconstrained_project.loads = [LoadConfig(region="right", dof="Y", magnitude=-4000.0)]
    # No boundary conditions at all -- a singular, unconstrained system.

    parameter = ModelParameter(
        name="youngs_modulus", current_value=200e9, lower_bound=150e9, upper_bound=250e9,
        path="material.youngs_modulus",
    )
    measurement_data = MeasurementData(quantity_name="maximum_displacement")
    measurement_data.add_point(MeasurementPoint(measurement_id="m1", measured_value=0.004))
    problem = ModelUpdateProblem(
        base_project=unconstrained_project, parameters=[parameter],
        measurement_data=measurement_data,
    )
    result = run_model_update(problem)
    assert result.status is ModelUpdateStatus.FAILED
    assert result.initial_objective is None


@pytest.mark.slow
def test_run_model_update_recovers_the_true_parameter_value() -> None:
    true_youngs_modulus = 180e9
    measured_value = _measured_displacement_at(true_youngs_modulus)

    measurement_data = MeasurementData(quantity_name="maximum_displacement", units="m")
    measurement_data.add_point(MeasurementPoint(measurement_id="m1", measured_value=measured_value))

    parameter = ModelParameter(
        name="youngs_modulus", current_value=200e9, lower_bound=150e9, upper_bound=250e9,
        path="material.youngs_modulus",
    )
    problem = ModelUpdateProblem(
        base_project=_cantilever_project(), parameters=[parameter],
        measurement_data=measurement_data,
    )
    config = ModelUpdateConfig(algorithm="coordinate_search", max_evaluations=40, seed=0)
    result = run_model_update(problem, config)

    assert result.status in (ModelUpdateStatus.CONVERGED, ModelUpdateStatus.MAX_EVALUATIONS)
    assert result.improved
    assert result.updated_parameters["youngs_modulus"] == pytest.approx(
        true_youngs_modulus, rel=0.02
    )
    assert result.final_objective < result.initial_objective
    assert result.n_evaluations > 0


@pytest.mark.slow
def test_run_model_update_respects_parameter_bounds() -> None:
    # A "true" value far outside the search bounds -- the update must still
    # respect theta_min <= theta <= theta_max rather than escaping them.
    measured_value = _measured_displacement_at(100e9)

    measurement_data = MeasurementData(quantity_name="maximum_displacement")
    measurement_data.add_point(MeasurementPoint(measurement_id="m1", measured_value=measured_value))
    parameter = ModelParameter(
        name="youngs_modulus", current_value=200e9, lower_bound=190e9, upper_bound=210e9,
        path="material.youngs_modulus",
    )
    problem = ModelUpdateProblem(
        base_project=_cantilever_project(), parameters=[parameter],
        measurement_data=measurement_data,
    )
    result = run_model_update(problem, ModelUpdateConfig(max_evaluations=20, seed=0))

    assert 190e9 <= result.updated_parameters["youngs_modulus"] <= 210e9


@pytest.mark.slow
def test_run_model_update_fixed_parameter_is_never_changed() -> None:
    measured_value = _measured_displacement_at(180e9)
    measurement_data = MeasurementData(quantity_name="maximum_displacement")
    measurement_data.add_point(MeasurementPoint(measurement_id="m1", measured_value=measured_value))

    updatable = ModelParameter(
        name="youngs_modulus", current_value=200e9, lower_bound=150e9, upper_bound=250e9,
        path="material.youngs_modulus",
    )
    fixed = ModelParameter(
        name="density", current_value=7850.0, lower_bound=7000.0, upper_bound=8500.0,
        path="material.density", updatable=False,
    )
    problem = ModelUpdateProblem(
        base_project=_cantilever_project(), parameters=[updatable, fixed],
        measurement_data=measurement_data,
    )
    result = run_model_update(problem, ModelUpdateConfig(max_evaluations=20, seed=0))

    assert result.updated_parameters["density"] == 7850.0
    assert result.updated_parameters["youngs_modulus"] != 200e9
