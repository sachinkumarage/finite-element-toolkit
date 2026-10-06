"""Tests for femtoolkit.optimization.algorithms.genetic_algorithm.

Uses the sphere benchmark (femtoolkit.optimization.benchmarks) to
verify the algorithm's mechanics -- selection, crossover, mutation,
elitism, reproducibility, bounds -- independent of any particular FEA
physics.
"""

from __future__ import annotations

from femtoolkit.optimization.algorithms import OptimizationConfig, StopReason
from femtoolkit.optimization.algorithms.genetic_algorithm import GeneticAlgorithm
from femtoolkit.optimization.benchmarks import build_benchmark_problem, sphere
from femtoolkit.optimization.evaluation import DesignStatus
from femtoolkit.optimization.history import OptimizationHistory
from femtoolkit.runs.manager import SimulationRunManager


def _sphere_problem(n_dimensions: int = 2):
    return build_benchmark_problem(
        "ga-sphere", sphere, n_dimensions=n_dimensions, lower_bound=-5.0, upper_bound=5.0
    )


def test_genetic_algorithm_converges_toward_known_minimum() -> None:
    problem = _sphere_problem()
    config = OptimizationConfig(
        algorithm="genetic_algorithm",
        population_size=20,
        max_evaluations=800,
        max_generations=100,
        elite_count=2,
        tournament_size=3,
        crossover_probability=0.9,
        mutation_probability=0.1,
        seed=42,
        tolerance=1e-12,
        patience=800,
    )
    history = OptimizationHistory()
    GeneticAlgorithm().optimize(problem, config, SimulationRunManager(), history)
    best = history.best_feasible(problem.objectives[0])
    assert best is not None
    assert best.objective_values["f"] < 1.0


def test_genetic_algorithm_respects_bounds() -> None:
    problem = _sphere_problem()
    config = OptimizationConfig(
        algorithm="genetic_algorithm", population_size=10, max_evaluations=100, seed=1
    )
    history = OptimizationHistory()
    GeneticAlgorithm().optimize(problem, config, SimulationRunManager(), history)
    for evaluation in history.evaluations:
        assert evaluation.status != DesignStatus.INVALID
        for name, value in evaluation.design_variables.items():
            variable = problem.variable(name)
            assert variable.lower_bound <= value <= variable.upper_bound


def test_genetic_algorithm_is_reproducible_with_same_seed() -> None:
    problem = _sphere_problem()
    config = OptimizationConfig(
        algorithm="genetic_algorithm", population_size=8, max_evaluations=60, seed=9
    )

    history_a = OptimizationHistory()
    GeneticAlgorithm().optimize(problem, config, SimulationRunManager(), history_a)
    history_b = OptimizationHistory()
    GeneticAlgorithm().optimize(problem, config, SimulationRunManager(), history_b)

    values_a = [e.design_variables for e in history_a.evaluations]
    values_b = [e.design_variables for e in history_b.evaluations]
    assert values_a == values_b


def test_genetic_algorithm_elitism_never_loses_the_best_individual() -> None:
    """The best-so-far feasible objective value must never get worse from one
    generation's elite-preserved individuals to the next (a direct, mechanical
    consequence of elitism -- not a claim about the whole population's average)."""
    problem = _sphere_problem()
    config = OptimizationConfig(
        algorithm="genetic_algorithm",
        population_size=10,
        max_evaluations=200,
        elite_count=2,
        seed=3,
        tolerance=1e-300,
        patience=100000,
    )
    history = OptimizationHistory()
    GeneticAlgorithm().optimize(problem, config, SimulationRunManager(), history)

    objective = problem.objectives[0]
    best_so_far_series = history.best_so_far_series(objective)
    feasible_series = [value for value in best_so_far_series if value is not None]
    for earlier, later in zip(feasible_series, feasible_series[1:], strict=False):
        assert later <= earlier + 1e-9


def test_genetic_algorithm_never_exceeds_max_evaluations() -> None:
    problem = _sphere_problem()
    config = OptimizationConfig(
        algorithm="genetic_algorithm", population_size=6, max_evaluations=15, seed=2
    )
    history = OptimizationHistory()
    GeneticAlgorithm().optimize(problem, config, SimulationRunManager(), history)
    assert history.n_evaluations <= 15


def test_genetic_algorithm_records_generation_zero_for_initial_population() -> None:
    problem = _sphere_problem()
    config = OptimizationConfig(
        algorithm="genetic_algorithm", population_size=6, max_evaluations=6, seed=1
    )
    history = OptimizationHistory()
    GeneticAlgorithm().optimize(problem, config, SimulationRunManager(), history)
    assert all(e.generation == 0 for e in history.evaluations)


def test_genetic_algorithm_stops_at_max_generations() -> None:
    problem = _sphere_problem()
    config = OptimizationConfig(
        algorithm="genetic_algorithm",
        population_size=4,
        max_evaluations=1000,
        max_generations=2,
        elite_count=1,
        tournament_size=2,
        tolerance=1e-300,
        patience=100000,
        seed=5,
    )
    history = OptimizationHistory()
    stop_reason = GeneticAlgorithm().optimize(problem, config, SimulationRunManager(), history)
    assert stop_reason == StopReason.MAX_GENERATIONS


def test_genetic_algorithm_handles_constraint_failure_policy() -> None:
    """An impossible-to-satisfy constraint means every evaluation is infeasible;
    the genetic algorithm must still run to completion without crashing."""
    from femtoolkit.optimization.constraints import Constraint, ConstraintRelation

    problem = _sphere_problem()
    impossible_constraint = Constraint(
        name="impossible", evaluate=lambda ctx: 1.0,
        relation=ConstraintRelation.LESS_EQUAL, limit=-1.0,
    )
    problem.constraints.append(impossible_constraint)
    config = OptimizationConfig(
        algorithm="genetic_algorithm", population_size=6, max_evaluations=12, seed=1
    )
    history = OptimizationHistory()
    GeneticAlgorithm().optimize(problem, config, SimulationRunManager(), history)
    assert history.n_evaluations == 12
    assert all(e.status == DesignStatus.INFEASIBLE for e in history.evaluations)


# --- Version 34: parallel (batched) evaluation ---


def test_genetic_algorithm_parallel_matches_serial_at_max_evaluations() -> None:
    # Forcing a MAX_EVALUATIONS stop (rather than CONVERGED) avoids the
    # documented mid-generation-stopping-granularity difference between
    # serial and batched evaluation (see the algorithm's module
    # docstring) -- under this stop reason, serial and batched
    # evaluation must produce bit-identical results for the same seed.
    def _run(orchestration_config):
        problem = _sphere_problem()
        config = OptimizationConfig(
            algorithm="genetic_algorithm", population_size=6, max_evaluations=24,
            max_generations=10, seed=7, patience=1000, tolerance=1e-15,
        )
        history = OptimizationHistory()
        stop_reason = GeneticAlgorithm().optimize(
            problem, config, SimulationRunManager(), history,
            orchestration_config=orchestration_config,
        )
        return history, stop_reason

    from femtoolkit.orchestration.config import OrchestrationConfig

    serial_history, serial_stop = _run(None)
    parallel_history, parallel_stop = _run(
        OrchestrationConfig(execution_mode="parallel", max_workers=2)
    )

    assert serial_stop == StopReason.MAX_EVALUATIONS == parallel_stop
    assert serial_history.n_evaluations == parallel_history.n_evaluations == 24
    serial_values = [e.objective_values["f"] for e in serial_history.evaluations]
    parallel_values = [e.objective_values["f"] for e in parallel_history.evaluations]
    assert serial_values == parallel_values
    assert [e.design_id for e in serial_history.evaluations] == [
        e.design_id for e in parallel_history.evaluations
    ]


def test_genetic_algorithm_parallel_respects_max_evaluations_ceiling() -> None:
    from femtoolkit.orchestration.config import OrchestrationConfig

    problem = _sphere_problem()
    config = OptimizationConfig(
        algorithm="genetic_algorithm", population_size=8, max_evaluations=20,
        max_generations=20, seed=3,
    )
    history = OptimizationHistory()
    GeneticAlgorithm().optimize(
        problem, config, SimulationRunManager(), history,
        orchestration_config=OrchestrationConfig(execution_mode="parallel", max_workers=2),
    )
    assert history.n_evaluations <= 20
