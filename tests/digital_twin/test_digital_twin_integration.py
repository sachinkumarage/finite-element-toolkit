"""Integration test: the full digital twin workflow, end to end.

.. code-block:: text

    measurement data -> baseline simulation -> parameter update ->
    updated simulation -> validation

Mirrors the cantilever beam example.
"""

from __future__ import annotations

import pytest

from femtoolkit.application.project import BoundaryConditionConfig, LoadConfig, Project
from femtoolkit.application.simulation_service import SimulationService
from femtoolkit.digital_twin.measurements import MeasurementData, MeasurementPoint
from femtoolkit.digital_twin.parameters import ModelParameter
from femtoolkit.digital_twin.problem import ModelUpdateProblem
from femtoolkit.digital_twin.results import ModelUpdateStatus
from femtoolkit.digital_twin.updating import ModelUpdateConfig, run_model_update
from femtoolkit.digital_twin.validation import compare_model_accuracy


def _cantilever_project(youngs_modulus: float) -> Project:
    project = Project(name="DT Integration Cantilever", analysis_type="linear_static")
    project.material.youngs_modulus = youngs_modulus
    project.material.poisson_ratio = 0.3
    project.material.density = 7850.0
    project.mesh.width = 2.0
    project.mesh.height = 0.4
    project.mesh.nx = 8
    project.mesh.ny = 2
    project.mesh.thickness = 0.012
    project.boundary_conditions = [
        BoundaryConditionConfig(region="left", dof="X", value=0.0),
        BoundaryConditionConfig(region="left", dof="Y", value=0.0),
    ]
    project.loads = [LoadConfig(region="right", dof="Y", magnitude=-5000.0)]
    return project


@pytest.mark.slow
def test_full_digital_twin_workflow() -> None:
    # 1. Physical reference data -- here, synthetic: the "true" structure has
    #    a Young's modulus of 190 GPa, unknown to the baseline model below.
    true_youngs_modulus = 190e9
    reference_run = SimulationService().run(_cantilever_project(true_youngs_modulus))
    assert reference_run.succeeded
    measured_value = reference_run.summary.maximum_displacement

    # 2. MeasurementData.
    measurement_data = MeasurementData(
        quantity_name="maximum_displacement", units="m",
        description="Synthetic reference data for the integration test.",
    )
    measurement_data.add_point(MeasurementPoint(measurement_id="m1", measured_value=measured_value))

    # 3. Baseline simulation model -- deliberately starts from a wrong guess.
    baseline_project = _cantilever_project(youngs_modulus=210e9)
    parameter = ModelParameter(
        name="youngs_modulus", current_value=210e9, lower_bound=150e9, upper_bound=250e9,
        path="material.youngs_modulus",
    )
    problem = ModelUpdateProblem(
        base_project=baseline_project, parameters=[parameter], measurement_data=measurement_data
    )

    # 4/5/6. Baseline prediction, comparison, discrepancy -- all produced by run_model_update.
    # 7/8. Parameter updating and re-running the simulation.
    result = run_model_update(problem, ModelUpdateConfig(max_evaluations=40, seed=0))

    assert result.status in (ModelUpdateStatus.CONVERGED, ModelUpdateStatus.MAX_EVALUATIONS)
    assert result.n_evaluations > 0
    assert result.initial_predictions.shape == (1,)
    assert result.updated_predictions.shape == (1,)

    # 9. Compare before/after accuracy.
    comparison = compare_model_accuracy(result)
    assert comparison.improved
    assert result.improved

    # 10. The updated model: closer to the true parameter than the baseline guess.
    updated_value = result.updated_parameters["youngs_modulus"]
    assert abs(updated_value - true_youngs_modulus) < abs(210e9 - true_youngs_modulus)
