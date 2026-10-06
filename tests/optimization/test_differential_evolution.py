"""Tests for femtoolkit.optimization.algorithms.differential_evolution.

Uses the sphere benchmark (femtoolkit.optimization.benchmarks), a known
analytic minimum at the origin, to verify the algorithm's mechanics
independent of any particular FEA physics -- a real (inexpensive,
always-solvable) simulation still runs for every evaluation, since
there is only one evaluation pipeline in this toolkit.
"""

from __future__ import annotations

import pytest

from femtoolkit.optimization.algorithms import OptimizationConfig, StopReason
from femtoolkit.optimization.algorithms.differential_evolution import DifferentialEvolution
from femtoolkit.optimization.benchmarks import build_benchmark_problem, sphere
from femtoolkit.optimization.evaluation import DesignStatus
from femtoolkit.optimization.history import OptimizationHistory
from femtoolkit.runs.manager import SimulationRunManager


def _sphere_problem(n_dimensions: int = 2):
    return build_benchmark_problem(
        "de-sphere", sphere, n_dimensions=n_dimensions, lower_bound=-5.0, upper_bound=5.0
    )


def test_differential_evolution_converges_near_known_minimum() -> None:
    problem = _sphere_problem()
    config = OptimizationConfig(
        algorithm="differential_evolution",
        population_size=15,
        max_evaluations=500,
        max_generations=80,
        mutation_factor=0.8,
        crossover_probability=0.9,
        seed=42,
        tolerance=1e-12,
        patience=500,
    )
    history = OptimizationHistory()
    DifferentialEvolution().optimize(problem, config, SimulationRunManager(), history)
    best = history.best_feasible(problem.objectives[0])
    assert best is not None
    assert best.objective_values["f"] < 0.1


def test_differential_evolution_respects_bounds() -> None:
    problem = _sphere_problem()
    config = OptimizationConfig(
        algorithm="differential_evolution", population_size=10, max_evaluations=100, seed=1
    )
    history = OptimizationHistory()
    DifferentialEvolution().optimize(problem, config, SimulationRunManager(), history)
    for evaluation in history.evaluations:
        assert evaluation.status != DesignStatus.INVALID
        for name, value in evaluation.design_variables.items():
            variable = problem.variable(name)
            assert variable.lower_bound <= value <= variable.upper_bound


def test_differential_evolution_is_reproducible_with_same_seed() -> None:
    problem = _sphere_problem()
    config = OptimizationConfig(
        algorithm="differential_evolution", population_size=8, max_evaluations=60, seed=7
    )

    history_a = OptimizationHistory()
    DifferentialEvolution().optimize(problem, config, SimulationRunManager(), history_a)
    history_b = OptimizationHistory()
    DifferentialEvolution().optimize(problem, config, SimulationRunManager(), history_b)

    values_a = [e.design_variables for e in history_a.evaluations]
    values_b = [e.design_variables for e in history_b.evaluations]
    assert values_a == values_b


def test_differential_evolution_differs_with_different_seed() -> None:
    problem = _sphere_problem()
    config_a = OptimizationConfig(
        algorithm="differential_evolution", population_size=8, max_evaluations=40, seed=1
    )
    config_b = OptimizationConfig(
        algorithm="differential_evolution", population_size=8, max_evaluations=40, seed=2
    )

    history_a = OptimizationHistory()
    DifferentialEvolution().optimize(problem, config_a, SimulationRunManager(), history_a)
    history_b = OptimizationHistory()
    DifferentialEvolution().optimize(problem, config_b, SimulationRunManager(), history_b)

    values_a = [e.design_variables for e in history_a.evaluations]
    values_b = [e.design_variables for e in history_b.evaluations]
    assert values_a != values_b


def test_differential_evolution_never_exceeds_max_evaluations() -> None:
    problem = _sphere_problem()
    config = OptimizationConfig(
        algorithm="differential_evolution", population_size=6, max_evaluations=10, seed=3
    )
    history = OptimizationHistory()
    stop_reason = DifferentialEvolution().optimize(
        problem, config, SimulationRunManager(), history
    )
    assert history.n_evaluations <= 10
    assert stop_reason in (StopReason.MAX_EVALUATIONS, StopReason.CONVERGED)


def test_differential_evolution_records_generation_on_every_evaluation() -> None:
    problem = _sphere_problem()
    config = OptimizationConfig(
        algorithm="differential_evolution", population_size=6, max_evaluations=20, seed=4
    )
    history = OptimizationHistory()
    DifferentialEvolution().optimize(problem, config, SimulationRunManager(), history)
    assert all(e.generation is not None for e in history.evaluations)
    # the initial population is generation 0; later trials are generation >= 1
    assert {e.generation for e in history.evaluations} >= {0}


def test_differential_evolution_requires_minimum_population() -> None:
    with pytest.raises(Exception, match="population_size"):
        OptimizationConfig(algorithm="differential_evolution", population_size=3)


def test_differential_evolution_stops_at_max_generations() -> None:
    problem = _sphere_problem()
    config = OptimizationConfig(
        algorithm="differential_evolution",
        population_size=4,
        max_evaluations=1000,
        max_generations=2,
        tolerance=1e-300,
        patience=100000,
        seed=5,
    )
    history = OptimizationHistory()
    stop_reason = DifferentialEvolution().optimize(
        problem, config, SimulationRunManager(), history
    )
    assert stop_reason == StopReason.MAX_GENERATIONS
    generations = {e.generation for e in history.evaluations}
    assert max(generations) <= 2


# --- Version 34: parallel (batched generation-0) evaluation ---


def test_differential_evolution_parallel_matches_serial() -> None:
    # DE only batches generation 0 (per-generation steady-state replacement
    # is left sequential -- see the algorithm's module docstring), which
    # was never an early-stop-checked region in serial mode either, so
    # serial and parallel must match exactly regardless of stop reason.
    def _run(orchestration_config):
        problem = _sphere_problem()
        config = OptimizationConfig(
            algorithm="differential_evolution", population_size=6, max_evaluations=30,
            max_generations=5, seed=11,
        )
        history = OptimizationHistory()
        DifferentialEvolution().optimize(
            problem, config, SimulationRunManager(), history,
            orchestration_config=orchestration_config,
        )
        return history

    from femtoolkit.orchestration.config import OrchestrationConfig

    serial_history = _run(None)
    parallel_history = _run(OrchestrationConfig(execution_mode="parallel", max_workers=2))

    serial_values = [e.objective_values["f"] for e in serial_history.evaluations]
    parallel_values = [e.objective_values["f"] for e in parallel_history.evaluations]
    assert serial_values == parallel_values
    assert [e.design_id for e in serial_history.evaluations] == [
        e.design_id for e in parallel_history.evaluations
    ]
