"""Example: parallel parameter study -- serial vs. parallel execution (Version 34).

**Procedure.** Builds the same cantilever beam parameter study as
``examples/studies/cantilever_load_study.py`` (Version 30), but sweeps
over plate *thickness* (not load) over eleven values, and runs it twice
through :class:`~femtoolkit.studies.runner.StudyRunner`: once serially
(``orchestration_config=None``, every prior version's exact behavior),
once in parallel across local worker processes
(:class:`~femtoolkit.orchestration.config.OrchestrationConfig`). Both
runs' results are compared scenario-by-scenario to confirm they are
identical -- parallel execution changes *how* scenarios are dispatched,
never *what* they compute -- and the two runs' wall-clock time is
compared to report a real, measured speedup and efficiency, never a
value engineered to look good.

**An honest number, not a guaranteed one.** This toolkit's FEA models
are small and fast; whether parallel execution actually wins depends on
how many scenarios there are, how expensive each one is, and how many
CPU cores the machine running this example has. The thickness sweep
below uses a moderately fine mesh specifically so each scenario takes
long enough for process-pool overhead to be worth paying -- see
``docs/orchestration.md``'s "recommended worker counts" section for
when parallel execution is (and is not) worth it.
"""

from __future__ import annotations

import time

from femtoolkit.application.project import BoundaryConditionConfig, LoadConfig, Project
from femtoolkit.orchestration.config import OrchestrationConfig
from femtoolkit.studies import ParameterDefinition, SimulationStudy, StudyRunner, get_extractor


def build_base_project() -> Project:
    """The same cantilever beam every Version 30/31/34 study example uses."""
    project = Project(name="Parallel Thickness Study", analysis_type="linear_static")
    project.material.youngs_modulus = 200e9
    project.material.poisson_ratio = 0.3
    project.material.density = 7850.0
    project.mesh.width = 2.0
    project.mesh.height = 0.4
    project.mesh.nx = 40
    project.mesh.ny = 10
    project.mesh.thickness = 0.02
    project.boundary_conditions = [
        BoundaryConditionConfig(region="left", dof="X", value=0.0),
        BoundaryConditionConfig(region="left", dof="Y", value=0.0),
    ]
    project.loads = [LoadConfig(region="right", dof="Y", magnitude=-5000.0)]
    return project


def main() -> None:
    """Run the same thickness sweep serially and in parallel, and compare them."""
    print("Finite Element Toolkit")
    print("Version 34 -- Parallel Parameter Study")
    print("=" * 45)

    thickness = ParameterDefinition(
        path="mesh.thickness", label="Plate thickness (m)",
        values=[0.010 + 0.001 * i for i in range(11)],
    )
    study = SimulationStudy(
        study_id="parallel-thickness-study", name="Parallel Thickness Study",
        base_project=build_base_project(), parameters=[thickness],
    )

    print(f"\nRunning {len(thickness.values)} scenarios serially...")
    start = time.perf_counter()
    serial_result = StudyRunner().run(study)
    serial_elapsed = time.perf_counter() - start
    print(f"  {len(serial_result.successful_runs)}/{len(serial_result.runs)} succeeded "
          f"in {serial_elapsed:.2f}s.")

    orchestration_config = OrchestrationConfig(execution_mode="parallel", max_workers=4)
    print(f"\nRunning the same {len(thickness.values)} scenarios in parallel "
          f"(max_workers={orchestration_config.max_workers})...")
    start = time.perf_counter()
    parallel_result, summary = StudyRunner().run_with_summary(
        study, orchestration_config=orchestration_config
    )
    parallel_elapsed = time.perf_counter() - start
    print(f"  {len(parallel_result.successful_runs)}/{len(parallel_result.runs)} succeeded "
          f"in {parallel_elapsed:.2f}s.")

    displacement = get_extractor("maximum_displacement")
    serial_values = [displacement(run) for run in serial_result.runs]
    parallel_values = [displacement(run) for run in parallel_result.runs]
    print("\nVerifying every scenario produced an identical result, serial vs. parallel:")
    for scenario, serial_value, parallel_value in zip(
        serial_result.scenarios, serial_values, parallel_values, strict=True
    ):
        assert serial_value == parallel_value, (
            f"Scenario {scenario.scenario_id!r} differed between serial and parallel execution."
        )
        print(f"  {scenario.scenario_id}: displacement={serial_value:.6e} m (matches)")
    print("Confirmed: parallel execution produced bit-identical results to serial.")

    print("\nMeasured performance (this run, this machine -- not a guarantee):")
    print(f"  Serial time:    {serial_elapsed:.3f}s")
    print(f"  Parallel time:  {parallel_elapsed:.3f}s")
    print(f"  Execution summary: total={summary.total_tasks}, "
          f"completed={summary.completed_tasks}, worker_count={summary.worker_count}")
    if summary.parallel_speedup is not None:
        print(f"  Speedup (from ExecutionSummary): {summary.parallel_speedup:.2f}x")
        print(f"  Efficiency: {summary.parallel_efficiency:.1%}")
    measured_speedup = serial_elapsed / parallel_elapsed if parallel_elapsed > 0 else float("nan")
    print(f"  Measured wall-clock speedup (serial_elapsed / parallel_elapsed): "
          f"{measured_speedup:.2f}x")
    print(
        "  (These two speedup numbers usually differ: ExecutionSummary's compares "
        "against an *estimated* serial time from average per-task duration, ignoring "
        "this particular serial run's own fixed costs, while the wall-clock number "
        "above compares two whole, real runs -- including this parallel run's "
        "one-time worker-pool startup cost, which a longer-running batch would amortize "
        "better than this small example does.)"
    )


if __name__ == "__main__":
    main()
