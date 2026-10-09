"""Tests for femtoolkit.digital_twin.problem."""

from __future__ import annotations

import pytest

from femtoolkit.application.project import BoundaryConditionConfig, LoadConfig, Project
from femtoolkit.digital_twin.measurements import MeasurementData, MeasurementPoint
from femtoolkit.digital_twin.parameters import ModelParameter
from femtoolkit.digital_twin.problem import ModelUpdateProblem
from femtoolkit.exceptions import ValidationError


def _cantilever_project() -> Project:
    project = Project(name="DT Problem Cantilever", analysis_type="linear_static")
    project.material.youngs_modulus = 200e9
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


def _parameter() -> ModelParameter:
    return ModelParameter(
        name="youngs_modulus", current_value=200e9, lower_bound=150e9, upper_bound=250e9,
        path="material.youngs_modulus",
    )


def _measurement_data() -> MeasurementData:
    data = MeasurementData(quantity_name="maximum_displacement", units="m")
    data.add_point(MeasurementPoint(measurement_id="m1", measured_value=0.004))
    return data


def test_valid_problem() -> None:
    problem = ModelUpdateProblem(
        base_project=_cantilever_project(), parameters=[_parameter()],
        measurement_data=_measurement_data(),
    )
    assert problem.updatable_parameters == [problem.parameters[0]]
    assert problem.current_parameter_values() == {"youngs_modulus": 200e9}


def test_problem_rejects_no_parameters() -> None:
    with pytest.raises(ValidationError):
        ModelUpdateProblem(
            base_project=_cantilever_project(), parameters=[], measurement_data=_measurement_data()
        )


def test_problem_rejects_duplicate_parameter_names() -> None:
    with pytest.raises(ValidationError):
        ModelUpdateProblem(
            base_project=_cantilever_project(), parameters=[_parameter(), _parameter()],
            measurement_data=_measurement_data(),
        )


def test_problem_rejects_empty_measurement_data() -> None:
    empty_data = MeasurementData(quantity_name="maximum_displacement")
    with pytest.raises(ValidationError):
        ModelUpdateProblem(
            base_project=_cantilever_project(), parameters=[_parameter()],
            measurement_data=empty_data,
        )


def test_problem_rejects_unknown_quantity_name() -> None:
    bad_data = MeasurementData(quantity_name="not_a_real_quantity")
    bad_data.add_point(MeasurementPoint(measurement_id="m1", measured_value=1.0))
    with pytest.raises(ValidationError):
        ModelUpdateProblem(
            base_project=_cantilever_project(), parameters=[_parameter()], measurement_data=bad_data
        )


def test_problem_rejects_non_positive_tolerance() -> None:
    with pytest.raises(ValidationError):
        ModelUpdateProblem(
            base_project=_cantilever_project(), parameters=[_parameter()],
            measurement_data=_measurement_data(), tolerance=0.0,
        )


@pytest.mark.slow
def test_problem_predict_runs_real_fea() -> None:
    problem = ModelUpdateProblem(
        base_project=_cantilever_project(), parameters=[_parameter()],
        measurement_data=_measurement_data(),
    )
    predictions = problem.predict({"youngs_modulus": 200e9})
    assert predictions.shape == (1,)
    assert predictions[0] > 0.0

    # A stiffer beam (higher E) deflects less.
    stiffer_predictions = problem.predict({"youngs_modulus": 250e9})
    assert stiffer_predictions[0] < predictions[0]
