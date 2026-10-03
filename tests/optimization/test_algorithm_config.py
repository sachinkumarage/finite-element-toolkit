"""Tests for the Version 33 extensions to femtoolkit.optimization.algorithms.base.

Covers OptimizationConfig's new population-based/robust-adjacent
fields, the new StopReason members, and should_stop's generation/
target-objective checks -- the V32 fields and behavior are already
covered by tests/optimization/test_random_search.py and
test_coordinate_search.py and are not repeated here.
"""

from __future__ import annotations

import pytest

from femtoolkit.exceptions import ValidationError
from femtoolkit.optimization.algorithms.base import OptimizationConfig, StopReason, should_stop
from femtoolkit.optimization.evaluation import DesignEvaluation, DesignStatus
from femtoolkit.optimization.history import OptimizationHistory
from femtoolkit.optimization.objectives import Objective, ObjectiveDirection


def _objective() -> Objective:
    return Objective(name="f", direction=ObjectiveDirection.MINIMIZE, evaluate=lambda ctx: None)


def _feasible(design_id: str, value: float) -> DesignEvaluation:
    return DesignEvaluation(
        design_id=design_id, design_variables={"x": value}, run_id=design_id,
        status=DesignStatus.FEASIBLE, objective_values={"f": value},
    )


# --- OptimizationConfig: population-based fields ---


@pytest.mark.parametrize(
    "algorithm", ["differential_evolution", "genetic_algorithm", "particle_swarm", "nsga2"]
)
def test_optimization_config_accepts_every_population_algorithm(algorithm: str) -> None:
    config = OptimizationConfig(algorithm=algorithm, population_size=10)
    assert config.algorithm == algorithm


def test_differential_evolution_requires_population_of_at_least_four() -> None:
    with pytest.raises(ValidationError):
        OptimizationConfig(algorithm="differential_evolution", population_size=3)
    OptimizationConfig(algorithm="differential_evolution", population_size=4)  # must not raise


def test_other_population_algorithms_require_at_least_two() -> None:
    with pytest.raises(ValidationError):
        OptimizationConfig(algorithm="particle_swarm", population_size=1)
    OptimizationConfig(algorithm="particle_swarm", population_size=2)  # must not raise


def test_mutation_factor_must_lie_in_zero_two_range() -> None:
    with pytest.raises(ValidationError):
        OptimizationConfig(algorithm="differential_evolution", mutation_factor=0.0)
    with pytest.raises(ValidationError):
        OptimizationConfig(algorithm="differential_evolution", mutation_factor=2.5)
    OptimizationConfig(algorithm="differential_evolution", mutation_factor=2.0)


def test_crossover_and_mutation_probability_must_lie_in_unit_interval() -> None:
    with pytest.raises(ValidationError):
        OptimizationConfig(algorithm="genetic_algorithm", crossover_probability=-0.1)
    with pytest.raises(ValidationError):
        OptimizationConfig(algorithm="genetic_algorithm", mutation_probability=1.1)


def test_elite_count_and_tournament_size_bounded_by_population_for_ga_and_nsga2() -> None:
    with pytest.raises(ValidationError):
        OptimizationConfig(algorithm="genetic_algorithm", population_size=5, elite_count=5)
    with pytest.raises(ValidationError):
        OptimizationConfig(algorithm="nsga2", population_size=5, tournament_size=6)
    OptimizationConfig(algorithm="genetic_algorithm", population_size=5, elite_count=4)


def test_pso_coefficients_must_be_non_negative() -> None:
    with pytest.raises(ValidationError):
        OptimizationConfig(algorithm="particle_swarm", cognitive_coefficient=-1.0)
    with pytest.raises(ValidationError):
        OptimizationConfig(algorithm="particle_swarm", social_coefficient=-1.0)


def test_inertia_weight_must_lie_in_zero_two_range() -> None:
    with pytest.raises(ValidationError):
        OptimizationConfig(algorithm="particle_swarm", inertia_weight=2.5)
    OptimizationConfig(algorithm="particle_swarm", inertia_weight=0.0)


def test_velocity_limit_must_lie_in_zero_one_range() -> None:
    with pytest.raises(ValidationError):
        OptimizationConfig(algorithm="particle_swarm", velocity_limit=0.0)
    with pytest.raises(ValidationError):
        OptimizationConfig(algorithm="particle_swarm", velocity_limit=1.5)


def test_max_generations_none_disables_the_generation_check() -> None:
    config = OptimizationConfig(algorithm="differential_evolution", max_generations=None)
    assert config.max_generations is None


def test_max_generations_must_be_positive_if_given() -> None:
    with pytest.raises(ValidationError):
        OptimizationConfig(algorithm="differential_evolution", max_generations=0)


def test_target_objective_must_be_finite_if_given() -> None:
    with pytest.raises(ValidationError):
        OptimizationConfig(algorithm="random_search", target_objective=float("inf"))
    OptimizationConfig(algorithm="random_search", target_objective=0.001)  # must not raise


# --- should_stop: generation and target_objective checks ---


def test_should_stop_max_generations() -> None:
    history = OptimizationHistory()
    history.add(_feasible("a", 10.0))
    config = OptimizationConfig(algorithm="differential_evolution", max_generations=3)
    assert should_stop(history, _objective(), config, 0, generation=3) == StopReason.MAX_GENERATIONS
    assert should_stop(history, _objective(), config, 0, generation=2) is None


def test_should_stop_ignores_generation_when_none_passed() -> None:
    history = OptimizationHistory()
    history.add(_feasible("a", 10.0))
    config = OptimizationConfig(algorithm="random_search", max_generations=1)
    # V32 algorithms never pass `generation`, so this must never trigger MAX_GENERATIONS
    assert should_stop(history, _objective(), config, 0) != StopReason.MAX_GENERATIONS


def test_should_stop_target_objective_reached_for_minimize() -> None:
    history = OptimizationHistory()
    history.add(_feasible("a", 5.0))
    config = OptimizationConfig(algorithm="random_search", target_objective=10.0)
    assert should_stop(history, _objective(), config, 0) == StopReason.TARGET_REACHED


def test_should_stop_target_objective_not_yet_reached() -> None:
    history = OptimizationHistory()
    history.add(_feasible("a", 50.0))
    config = OptimizationConfig(
        algorithm="random_search", target_objective=10.0, tolerance=1e-300, patience=10**9
    )
    assert should_stop(history, _objective(), config, 0) is None


def test_should_stop_target_objective_reached_for_maximize() -> None:
    maximize_objective = Objective(
        name="f", direction=ObjectiveDirection.MAXIMIZE, evaluate=lambda ctx: None
    )
    history = OptimizationHistory()
    history.add(_feasible("a", 50.0))
    config = OptimizationConfig(algorithm="random_search", target_objective=10.0)
    assert should_stop(history, maximize_objective, config, 0) == StopReason.TARGET_REACHED
