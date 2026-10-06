"""Example: parallel optimization candidate evaluation (Version 34).

**Procedure.** Runs the same genetic-algorithm beam-thickness
optimization (minimize tip displacement, subject to a stress limit --
the same problem as ``examples/optimization/beam_thickness_optimization.py``,
Version 32) twice with an identical seed: once serially (every design
evaluated one at a time, in the calling process -- identical to every
prior version), once in parallel
(:class:`~femtoolkit.orchestration.config.OrchestrationConfig`, which
genetic algorithm uses to batch-evaluate generation 0's initial
population and every full generation's offspring across local worker
processes -- see
:mod:`femtoolkit.optimization.algorithms.genetic_algorithm`'s module
docstring for exactly which evaluations get batched and why). **The
algorithm itself is completely unchanged** -- this example configures a
budget (``tolerance``/``patience``) that forces both runs to stop at
``max_evaluations`` rather than an early convergence detection, which
is what makes an exact, evaluation-by-evaluation comparison meaningful
(see that same module docstring's note on why batched evaluation can
make a *convergence*-triggered stop land at a different evaluation
count than the un-batched loop would have -- a real, documented
trade-off this example deliberately avoids triggering so it can show a
clean, honest comparison).
"""

from __future__ import annotations

import time

from femtoolkit.application.project import BoundaryConditionConfig, LoadConfig, Project
from femtoolkit.optimization import (
    Constraint,
    ConstraintRelation,
    Objective,
    ObjectiveDirection,
    OptimizationConfig,
    OptimizationProblem,
    OptimizationRunner,
    from_result_extractor,
)
from femtoolkit.optimization.variables import DesignVariable, DesignVariableType
from femtoolkit.orchestration.config import OrchestrationConfig
from femtoolkit.studies.extractors import get_extractor


def build_problem() -> OptimizationProblem:
    base_project = Project(name="Parallel GA Beam Design", analysis_type="linear_static")
    base_project.material.youngs_modulus = 200e9
    base_project.material.poisson_ratio = 0.3
    base_project.material.density = 7850.0
    base_project.mesh.width = 2.0
    base_project.mesh.height = 0.4
    base_project.mesh.nx = 20
    base_project.mesh.ny = 5
    base_project.mesh.thickness = 0.010
    base_project.boundary_conditions = [
        BoundaryConditionConfig(region="left", dof="X", value=0.0),
        BoundaryConditionConfig(region="left", dof="Y", value=0.0),
    ]
    base_project.loads = [LoadConfig(region="right", dof="Y", magnitude=-4000.0)]

    thickness = DesignVariable(
        name="thickness", path="mesh.thickness", variable_type=DesignVariableType.CONTINUOUS,
        lower_bound=0.005, upper_bound=0.020, default_value=0.010, units="m",
    )
    displacement_objective = Objective(
        name="maximum_displacement", direction=ObjectiveDirection.MINIMIZE,
        evaluate=from_result_extractor(get_extractor("maximum_displacement")), units="m",
    )
    stress_constraint = Constraint(
        name="stress_limit",
        evaluate=from_result_extractor(get_extractor("maximum_von_mises_stress")),
        relation=ConstraintRelation.LESS_EQUAL, limit=250e6, units="Pa",
    )
    return OptimizationProblem(
        name="parallel-ga-beam", base_project=base_project, design_variables=[thickness],
        objectives=[displacement_objective], constraints=[stress_constraint],
    )


def run_genetic_algorithm(orchestration_config: OrchestrationConfig | None):
    config = OptimizationConfig(
        algorithm="genetic_algorithm", population_size=10, max_evaluations=60,
        max_generations=20, seed=11,
        # A wide-open convergence budget forces both runs to stop at
        # max_evaluations rather than an early CONVERGED detection -- see the
        # module docstring above for why that matters for this comparison.
        tolerance=1e-15, patience=100000,
    )
    return OptimizationRunner().run(
        build_problem(), config, orchestration_config=orchestration_config
    )


def main() -> None:
    """Run the same genetic-algorithm search serially and in parallel, and compare them."""
    print("Finite Element Toolkit")
    print("Version 34 -- Parallel Optimization (Genetic Algorithm)")
    print("=" * 55)

    print("\nRunning genetic algorithm serially...")
    start = time.perf_counter()
    serial_result = run_genetic_algorithm(None)
    serial_elapsed = time.perf_counter() - start
    print(f"  Stopped: {serial_result.stop_reason.value}. "
          f"{serial_result.history.n_evaluations} evaluations in {serial_elapsed:.2f}s.")

    orchestration_config = OrchestrationConfig(execution_mode="parallel", max_workers=4)
    print(
        f"\nRunning the same search in parallel "
        f"(max_workers={orchestration_config.max_workers})..."
    )
    start = time.perf_counter()
    parallel_result = run_genetic_algorithm(orchestration_config)
    parallel_elapsed = time.perf_counter() - start
    print(f"  Stopped: {parallel_result.stop_reason.value}. "
          f"{parallel_result.history.n_evaluations} evaluations in {parallel_elapsed:.2f}s.")

    print("\nVerifying the search produced identical candidates and results:")
    serial_evals = serial_result.history.evaluations
    parallel_evals = parallel_result.history.evaluations
    assert len(serial_evals) == len(parallel_evals), "Evaluation counts differed."
    for serial_eval, parallel_eval in zip(serial_evals, parallel_evals, strict=True):
        assert serial_eval.design_id == parallel_eval.design_id
        assert serial_eval.design_variables == parallel_eval.design_variables
        assert serial_eval.objective_values == parallel_eval.objective_values
    print(f"  All {len(serial_evals)} evaluations matched exactly "
          "(design IDs, variable values, and objective values).")

    best_serial = serial_result.history.best_feasible(serial_result.objectives[0])
    print(f"\nBest feasible design (serial run): {best_serial.design_id} -- "
          f"thickness={best_serial.design_variables['thickness']:.5f} m, "
          f"displacement={best_serial.objective_values['maximum_displacement']:.6e} m")

    print("\nMeasured wall-clock time (this run, this machine -- not a guarantee):")
    print(f"  Serial:    {serial_elapsed:.3f}s")
    print(f"  Parallel:  {parallel_elapsed:.3f}s")
    if parallel_elapsed > serial_elapsed:
        print(
            "\n  Parallel was slower here -- a genuine, expected result, not a bug. "
            "GeneticAlgorithm batch-evaluates generation 0, then each generation's "
            "offspring separately; LocalProcessBackend creates a fresh worker pool per "
            "batch (see its module docstring), so with a small population spread across "
            "several generations, the one-time pool-startup cost is paid repeatedly and "
            "can easily outweigh the time saved running a handful of small, fast FEA "
            "solves concurrently. A larger population evaluated over fewer generations "
            "(more work per pool) amortizes that cost far better -- see "
            "parallel_parameter_study.py for a batch shape (one pool, many scenarios) "
            "where parallel execution wins more reliably."
        )


if __name__ == "__main__":
    main()
