"""Example: comparing optimization algorithms on a mathematical benchmark (Version 33).

**Procedure.** Runs every algorithm in this toolkit (random search,
coordinate search, differential evolution, genetic algorithm, particle
swarm optimization) on the same mathematical benchmark function
(Rosenbrock, a curved, narrow valley with a known minimum at ``x = 1``
in every dimension) under the same evaluation budget and seed, then
reports each algorithm's final objective value, evaluation count, and
execution time side by side.

**No overall ranking is produced.** Which algorithm performs best is
problem-dependent -- this script reports what happened on *this one
benchmark*, not a general claim that any algorithm is universally
superior to another.
"""

from __future__ import annotations

import time

from femtoolkit.optimization.algorithms import OptimizationConfig
from femtoolkit.optimization.benchmarks import build_benchmark_problem, rosenbrock
from femtoolkit.optimization.runner import OptimizationRunner

_ALGORITHMS: list[tuple[str, dict]] = [
    ("random_search", {}),
    ("coordinate_search", {"step_size": 0.1}),
    (
        "differential_evolution",
        {"population_size": 15, "mutation_factor": 0.8, "crossover_probability": 0.9},
    ),
    (
        "genetic_algorithm",
        {
            "population_size": 15, "elite_count": 2, "tournament_size": 3,
            "mutation_probability": 0.1,
        },
    ),
    (
        "particle_swarm",
        {"population_size": 15, "inertia_weight": 0.7, "cognitive_coefficient": 1.5,
         "social_coefficient": 1.5},
    ),
]

_MAX_EVALUATIONS = 400
_SEED = 42
_N_DIMENSIONS = 3


def main() -> None:
    """Run every algorithm on the Rosenbrock benchmark and print a comparison table."""
    print("Finite Element Toolkit")
    print("Version 33 -- Algorithm Benchmark Comparison (Rosenbrock)")
    print("=" * 60)
    print(
        f"\n{_N_DIMENSIONS}-dimensional Rosenbrock function, known minimum f(1,...,1) = 0, "
        f"{_MAX_EVALUATIONS} evaluations, seed={_SEED} for every algorithm."
    )

    results = []
    for algorithm, extra_kwargs in _ALGORITHMS:
        problem = build_benchmark_problem(
            f"rosenbrock-{algorithm}", rosenbrock, n_dimensions=_N_DIMENSIONS,
            lower_bound=-2.0, upper_bound=2.0,
        )
        config = OptimizationConfig(
            algorithm=algorithm, max_evaluations=_MAX_EVALUATIONS, seed=_SEED,
            tolerance=1e-12, patience=_MAX_EVALUATIONS, **extra_kwargs,
        )
        start = time.perf_counter()
        result = OptimizationRunner().run(problem, config)
        elapsed = time.perf_counter() - start

        best = result.best_feasible()
        results.append(
            {
                "algorithm": algorithm,
                "best_f": best.objective_values["f"] if best else None,
                "n_evaluations": result.history.n_evaluations,
                "stop_reason": result.stop_reason.value,
                "elapsed_seconds": elapsed,
            }
        )

    header = (
        f"\n{'Algorithm':<22} {'Best f(x)':>14} {'Evaluations':>12} "
        f"{'Stop Reason':>16} {'Time (s)':>10}"
    )
    print(header)
    print("-" * 78)
    for row in results:
        print(
            f"{row['algorithm']:<22} {row['best_f']:>14.6e} {row['n_evaluations']:>12} "
            f"{row['stop_reason']:>16} {row['elapsed_seconds']:>10.3f}"
        )

    print(
        "\nNo overall ranking is produced -- these results describe this one benchmark "
        "and evaluation budget only. A different problem (e.g. a highly multimodal "
        "function, or a problem with many constraints) can favor a different algorithm."
    )


if __name__ == "__main__":
    main()
