"""Example: digital twin model updating for a cantilever beam (Version 38).

**Procedure.** A baseline FEA model guesses Young's modulus at 200 GPa. A
*synthetic* "measured" tip displacement (clearly labeled as such -- this is
not real experimental data) is generated from a model whose true Young's
modulus is 180 GPa, representing a structure whose actual stiffness differs
from the baseline assumption. Model updating adjusts the baseline's Young's
modulus so the simulation agrees with the measured displacement, then
compares before/after accuracy.

.. code-block:: text

    Measured tip displacement
            |
            v
    Baseline FEA prediction
            |
            v
    Difference
            |
            v
    Update Young's modulus
            |
            v
    New FEA prediction
            |
            v
    Improved agreement
"""

from __future__ import annotations

from femtoolkit.application.project import BoundaryConditionConfig, LoadConfig, Project
from femtoolkit.application.simulation_service import SimulationService
from femtoolkit.digital_twin.measurements import MeasurementData, MeasurementPoint
from femtoolkit.digital_twin.parameters import ModelParameter
from femtoolkit.digital_twin.problem import ModelUpdateProblem
from femtoolkit.digital_twin.updating import ModelUpdateConfig, run_model_update
from femtoolkit.digital_twin.validation import compare_model_accuracy

_TRUE_YOUNGS_MODULUS = 180e9  # The structure's real (unknown to the baseline) stiffness.
_BASELINE_YOUNGS_MODULUS = 200e9  # The baseline model's initial guess.


def build_project(youngs_modulus: float) -> Project:
    """A simple cantilever beam, length 2 m, depth 0.4 m, thickness 0.01 m."""
    project = Project(name="Digital Twin Cantilever", analysis_type="linear_static")
    project.material.youngs_modulus = youngs_modulus
    project.material.poisson_ratio = 0.3
    project.material.density = 7850.0
    project.mesh.width = 2.0
    project.mesh.height = 0.4
    project.mesh.nx = 10
    project.mesh.ny = 3
    project.mesh.thickness = 0.01
    project.boundary_conditions = [
        BoundaryConditionConfig(region="left", dof="X", value=0.0),
        BoundaryConditionConfig(region="left", dof="Y", value=0.0),
    ]
    project.loads = [LoadConfig(region="right", dof="Y", magnitude=-4000.0)]
    return project


def main() -> None:
    print("Generating SYNTHETIC measured data (not real experimental data) from a")
    print(f"structure whose true Young's modulus is {_TRUE_YOUNGS_MODULUS / 1e9:.0f} GPa...")
    true_project = build_project(_TRUE_YOUNGS_MODULUS)
    true_result = SimulationService().run(true_project)
    measured_displacement = true_result.summary.maximum_displacement
    print(f"Synthetic measured tip displacement: {measured_displacement:.6e} m\n")

    measurement_data = MeasurementData(
        quantity_name="maximum_displacement", units="m",
        description="SYNTHETIC data generated for this example -- not real experimental data.",
    )
    measurement_data.add_point(
        MeasurementPoint(measurement_id="tip-displacement-1", measured_value=measured_displacement)
    )

    baseline_project = build_project(_BASELINE_YOUNGS_MODULUS)
    parameter = ModelParameter(
        name="youngs_modulus", current_value=_BASELINE_YOUNGS_MODULUS,
        lower_bound=150e9, upper_bound=250e9, units="Pa", path="material.youngs_modulus",
    )
    problem = ModelUpdateProblem(
        base_project=baseline_project, parameters=[parameter], measurement_data=measurement_data
    )

    print(f"Baseline Young's modulus guess: {_BASELINE_YOUNGS_MODULUS / 1e9:.1f} GPa")
    result = run_model_update(problem, ModelUpdateConfig(algorithm="coordinate_search", seed=0))

    print(f"\nStatus: {result.status.value}")
    print(f"Evaluations used: {result.n_evaluations}")
    print(f"Baseline prediction:  {result.initial_predictions[0]:.6e} m")
    print(f"Updated prediction:   {result.updated_predictions[0]:.6e} m")
    print(f"Measured value:       {result.measured_values[0]:.6e} m")
    print(f"Baseline residual:    {result.residuals_before[0]:.6e} m")
    print(f"Updated residual:     {result.residuals_after[0]:.6e} m")
    print(f"Updated Young's modulus: {result.updated_parameters['youngs_modulus'] / 1e9:.2f} GPa")

    comparison = compare_model_accuracy(result)
    print("\n                 Before Update      After Update")
    print(f"MAE              {comparison.before.mae:.4e}       {comparison.after.mae:.4e}")
    print(f"RMSE             {comparison.before.rmse:.4e}       {comparison.after.rmse:.4e}")
    print(
        f"Max Error        {comparison.before_max_absolute_error:.4e}       "
        f"{comparison.after_max_absolute_error:.4e}"
    )
    print(
        f"Relative Error   {comparison.before.mean_relative_error:.2%}             "
        f"{comparison.after.mean_relative_error:.2%}"
    )
    print(f"\nModel updating improved agreement with measurements: {comparison.improved}")


if __name__ == "__main__":
    main()
