"""Example: failure handling in a parallel batch (Version 34).

**Procedure.** Builds a parameter study mixing five valid thickness
scenarios with two deliberately invalid ones (a negative Young's
modulus -- physically meaningless, rejected during validation before
any solver runs), runs the whole batch in parallel, and demonstrates
that:

1. A single task's failure never aborts the batch -- every other task
   still runs and reports its own real result (spec section 15: "do
   not make individual failures abort the entire study unless
   fail_fast=True").
2. Each failure is reported with structured information (which
   scenario, which stage, what the error was) -- never silently
   swallowed.
3. The same batch with ``fail_fast=True`` behaves differently: it stops
   submitting new tasks after the first failure (though, as documented
   in :mod:`femtoolkit.orchestration.backends.local_process`, tasks
   already dispatched to a worker are allowed to finish rather than
   being forcibly killed).
"""

from __future__ import annotations

from femtoolkit.application.project import BoundaryConditionConfig, LoadConfig, Project
from femtoolkit.orchestration.config import OrchestrationConfig
from femtoolkit.runs.models import RunStatus
from femtoolkit.studies import ParameterDefinition, SimulationStudy, StudyRunner
from femtoolkit.studies.scenarios import Scenario


def build_base_project() -> Project:
    project = Project(name="Failure Handling Study", analysis_type="linear_static")
    project.material.youngs_modulus = 200e9
    project.material.poisson_ratio = 0.3
    project.material.density = 7850.0
    project.mesh.width = 2.0
    project.mesh.height = 0.4
    project.mesh.nx = 16
    project.mesh.ny = 4
    project.mesh.thickness = 0.02
    project.boundary_conditions = [
        BoundaryConditionConfig(region="left", dof="X", value=0.0),
        BoundaryConditionConfig(region="left", dof="Y", value=0.0),
    ]
    project.loads = [LoadConfig(region="right", dof="Y", magnitude=-2000.0)]
    return project


def build_study() -> SimulationStudy:
    valid_thicknesses = ParameterDefinition(
        path="mesh.thickness", label="Plate thickness (m)",
        values=[0.010, 0.014, 0.018, 0.022, 0.026],
    )
    # Deliberately invalid: a negative Young's modulus is physically meaningless
    # and is rejected during configuration validation, before any solver runs --
    # a controlled, reliable, always-reproducible way to trigger RunStatus.FAILED.
    invalid_scenarios = [
        Scenario(
            scenario_id="invalid-negative-modulus",
            name="Invalid: negative Young's modulus",
            parameter_overrides={"material.youngs_modulus": -1.0},
        ),
        Scenario(
            scenario_id="invalid-zero-modulus",
            name="Invalid: zero Young's modulus",
            parameter_overrides={"material.youngs_modulus": 0.0},
        ),
    ]
    return SimulationStudy(
        study_id="failure-handling-study", name="Failure Handling Study",
        base_project=build_base_project(), parameters=[valid_thicknesses],
        scenarios=invalid_scenarios,
    )


def main() -> None:
    """Run a mixed valid/invalid batch in parallel and report on its failures."""
    print("Finite Element Toolkit")
    print("Version 34 -- Failure Handling in a Parallel Batch")
    print("=" * 52)

    study = build_study()
    print(f"\nBatch: {len(study.scenarios) + 5} scenarios "
          f"(5 valid thickness values + 2 deliberately invalid).")

    print("\nRunning the full batch in parallel (fail_fast=False, the default)...")
    orchestration_config = OrchestrationConfig(execution_mode="parallel", max_workers=4)
    result, summary = StudyRunner().run_with_summary(
        study, orchestration_config=orchestration_config
    )

    print(f"\nExecution summary (task-level -- did each task's function run without "
          f"raising?): total={summary.total_tasks}, completed={summary.completed_tasks}, "
          f"failed={summary.failed_tasks}, cancelled={summary.cancelled_tasks}")
    print(
        "  Note: every task here shows COMPLETED, including the two invalid scenarios --"
        " SimulationRunManager.execute() never raises for an ordinary validation/solver"
        " failure, it returns a structured FAILED SimulationRun instead (see its own"
        " docstring). 'Task completed' (no exception) and 'simulation succeeded' are"
        " deliberately different questions; the per-scenario outcome below is where"
        " the real simulation-level result lives."
    )
    print(f"\nStudy completed despite failures: "
          f"{len(result.successful_runs)}/{len(result.runs)} runs succeeded "
          "(simulation-level, from each run's own RunStatus).")

    print("\nPer-scenario outcome:")
    for run in result.runs:
        if run.status is RunStatus.COMPLETED:
            print(f"  {run.scenario_id:>28}: COMPLETED")
        else:
            print(f"  {run.scenario_id:>28}: {run.status.value.upper()} "
                  f"(stage={run.error_stage}) -- {run.error_message}")

    assert len(result.successful_runs) == 5, "Expected exactly the 5 valid scenarios to succeed."
    assert len(result.failed_runs) == 2, "Expected exactly the 2 invalid scenarios to fail."
    print("\nConfirmed: both invalid scenarios failed with structured error information, "
          "and every valid scenario still completed successfully -- the batch was never "
          "aborted by the two failures.")

    print("\nNow comparing against fail_fast=True on the same batch...")
    fail_fast_config = OrchestrationConfig(
        execution_mode="parallel", max_workers=4, fail_fast=True
    )
    fail_fast_result, fail_fast_summary = StudyRunner().run_with_summary(
        study, orchestration_config=fail_fast_config
    )
    print(f"  fail_fast execution summary: total={fail_fast_summary.total_tasks}, "
          f"completed={fail_fast_summary.completed_tasks}, "
          f"failed={fail_fast_summary.failed_tasks}, "
          f"cancelled={fail_fast_summary.cancelled_tasks}")
    print(
        "  With fail_fast=True, no further tasks are submitted once a failure is "
        "observed -- but tasks already dispatched to a worker when that failure is "
        "detected are still allowed to finish (a deliberately softer stop than "
        "forcibly killing in-flight work; see OrchestrationConfig.fail_fast's "
        "docstring). The exact cancelled-vs-completed split above is therefore "
        "timing-dependent, not a fixed number this example can assert on."
    )


if __name__ == "__main__":
    main()
