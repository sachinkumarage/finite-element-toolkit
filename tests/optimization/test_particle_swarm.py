"""Tests for femtoolkit.optimization.algorithms.particle_swarm.

Uses the sphere benchmark (femtoolkit.optimization.benchmarks) to
verify particle initialization, velocity/position updates, bounds, and
reproducibility independent of any particular FEA physics.
"""

from __future__ import annotations

from femtoolkit.optimization.algorithms import OptimizationConfig, StopReason
from femtoolkit.optimization.algorithms.particle_swarm import ParticleSwarmOptimization
from femtoolkit.optimization.benchmarks import build_benchmark_problem, sphere
from femtoolkit.optimization.evaluation import DesignStatus
from femtoolkit.optimization.history import OptimizationHistory
from femtoolkit.runs.manager import SimulationRunManager


def _sphere_problem(n_dimensions: int = 2):
    return build_benchmark_problem(
        "pso-sphere", sphere, n_dimensions=n_dimensions, lower_bound=-5.0, upper_bound=5.0
    )


def test_particle_swarm_converges_near_known_minimum() -> None:
    problem = _sphere_problem()
    config = OptimizationConfig(
        algorithm="particle_swarm",
        population_size=15,
        max_evaluations=500,
        max_generations=80,
        inertia_weight=0.7,
        cognitive_coefficient=1.5,
        social_coefficient=1.5,
        velocity_limit=0.2,
        seed=42,
        tolerance=1e-12,
        patience=500,
    )
    history = OptimizationHistory()
    ParticleSwarmOptimization().optimize(problem, config, SimulationRunManager(), history)
    best = history.best_feasible(problem.objectives[0])
    assert best is not None
    assert best.objective_values["f"] < 0.1


def test_particle_swarm_initial_positions_are_within_bounds() -> None:
    problem = _sphere_problem()
    config = OptimizationConfig(
        algorithm="particle_swarm", population_size=10, max_evaluations=10, seed=1
    )
    history = OptimizationHistory()
    ParticleSwarmOptimization().optimize(problem, config, SimulationRunManager(), history)
    # the first `population_size` evaluations are the initial swarm (generation 0)
    for evaluation in history.evaluations:
        assert evaluation.generation == 0
        for name, value in evaluation.design_variables.items():
            variable = problem.variable(name)
            assert variable.lower_bound <= value <= variable.upper_bound


def test_particle_swarm_respects_bounds_after_velocity_updates() -> None:
    problem = _sphere_problem()
    config = OptimizationConfig(
        algorithm="particle_swarm", population_size=8, max_evaluations=150, seed=2
    )
    history = OptimizationHistory()
    ParticleSwarmOptimization().optimize(problem, config, SimulationRunManager(), history)
    for evaluation in history.evaluations:
        assert evaluation.status != DesignStatus.INVALID
        for name, value in evaluation.design_variables.items():
            variable = problem.variable(name)
            assert variable.lower_bound <= value <= variable.upper_bound


def test_particle_swarm_is_reproducible_with_same_seed() -> None:
    problem = _sphere_problem()
    config = OptimizationConfig(
        algorithm="particle_swarm", population_size=8, max_evaluations=60, seed=5
    )

    history_a = OptimizationHistory()
    ParticleSwarmOptimization().optimize(problem, config, SimulationRunManager(), history_a)
    history_b = OptimizationHistory()
    ParticleSwarmOptimization().optimize(problem, config, SimulationRunManager(), history_b)

    values_a = [e.design_variables for e in history_a.evaluations]
    values_b = [e.design_variables for e in history_b.evaluations]
    assert values_a == values_b


def test_particle_swarm_never_exceeds_max_evaluations() -> None:
    problem = _sphere_problem()
    config = OptimizationConfig(
        algorithm="particle_swarm", population_size=6, max_evaluations=12, seed=3
    )
    history = OptimizationHistory()
    ParticleSwarmOptimization().optimize(problem, config, SimulationRunManager(), history)
    assert history.n_evaluations <= 12


def test_particle_swarm_stops_at_max_generations() -> None:
    problem = _sphere_problem()
    config = OptimizationConfig(
        algorithm="particle_swarm",
        population_size=4,
        max_evaluations=1000,
        max_generations=2,
        tolerance=1e-300,
        patience=100000,
        seed=6,
    )
    history = OptimizationHistory()
    stop_reason = ParticleSwarmOptimization().optimize(
        problem, config, SimulationRunManager(), history
    )
    assert stop_reason == StopReason.MAX_GENERATIONS
