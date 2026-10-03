"""A modular genetic algorithm: selection, crossover, mutation, elitism (Version 33).

.. code-block:: text

    Initialize a population of candidate vectors, uniformly random in bounds
            |
            v
    Evaluate every candidate (generation 0)
            |
            v
    For each generation:
        Elitism: copy the best `elite_count` individuals unchanged
        Fill the rest of the next generation:
            Select two parents (tournament selection)
            Crossover (uniform, with probability crossover_probability)
            Mutation (per-gene random reset, with probability mutation_probability)
            Evaluate the resulting child
        Replace the population with elites + children
        Repeat until a stopping condition is reached

**The five building blocks, briefly.** A *population* is the set of
candidate designs considered together at one generation; *fitness
evaluation* scores every candidate (here: the same feasibility-first
comparison every algorithm in this package already uses, never a
separate "fitness function"). *Selection* chooses which individuals get
to reproduce -- this implementation uses **tournament selection**: draw
``tournament_size`` individuals at random and keep the best one, a
simple mechanism whose selection pressure is tuned by one parameter.
*Crossover* combines two parents' genes into a child (here: **uniform
crossover**, each gene independently taken from one parent or the
other). *Mutation* randomly perturbs a child's genes to maintain
diversity the population might otherwise lose (here: **random-reset
mutation**, a mutated gene is redrawn uniformly within its own bounds).
*Elitism* guarantees the best individuals found so far are never lost
to an unlucky generation of crossover/mutation.

Deliberately **one** selection strategy and **one** crossover/mutation
pair are implemented -- a clean, well-tested foundation rather than a
large configurable library of interchangeable operators.

Mixed-type design variables share the same real-valued box encoding
every population-based algorithm in this package uses (see
:mod:`femtoolkit.optimization.algorithms._encoding`). Reproducible via
``config.seed``. No mathematical global-optimality guarantee is made.
"""

from __future__ import annotations

import functools

import numpy as np

from femtoolkit.optimization.algorithms._encoding import (
    bounds_arrays,
    decode_vector,
    sample_vector,
)
from femtoolkit.optimization.algorithms.base import (
    OptimizationAlgorithm,
    OptimizationConfig,
    StopReason,
    should_stop,
)
from femtoolkit.optimization.evaluation import (
    DesignEvaluation,
    DesignStatus,
    evaluate_design,
    is_better_evaluation,
)
from femtoolkit.optimization.history import OptimizationHistory
from femtoolkit.optimization.problems import OptimizationProblem
from femtoolkit.runs.manager import SimulationRunManager


class GeneticAlgorithm(OptimizationAlgorithm):
    """A real-valued-encoded genetic algorithm with tournament selection and elitism."""

    def optimize(
        self,
        problem: OptimizationProblem,
        config: OptimizationConfig,
        run_manager: SimulationRunManager,
        history: OptimizationHistory,
        starting_evaluation: DesignEvaluation | None = None,
    ) -> StopReason:
        del starting_evaluation  # the genetic algorithm initializes its own random population
        rng = np.random.default_rng(config.seed)
        primary_objective = problem.objectives[0]
        design_variables = problem.design_variables
        lower, upper = bounds_arrays(design_variables)
        population_size = config.population_size
        consecutive_failures = 0
        evaluation_index = 0

        def _evaluate(vector: np.ndarray, generation: int) -> DesignEvaluation:
            nonlocal evaluation_index, consecutive_failures
            values = decode_vector(design_variables, vector)
            evaluation = evaluate_design(
                design_id=f"{problem.name}-ga-{evaluation_index}",
                values=values,
                design_variables=design_variables,
                base_project=problem.base_project,
                objectives=problem.objectives,
                constraints=problem.constraints,
                run_manager=run_manager,
                generation=generation,
            )
            history.add(evaluation)
            evaluation_index += 1
            failed = evaluation.status in (DesignStatus.FAILED, DesignStatus.INVALID)
            consecutive_failures = consecutive_failures + 1 if failed else 0
            return evaluation

        def _tournament_select(
            population: list[np.ndarray], evaluations: list[DesignEvaluation]
        ) -> int:
            contestants = rng.choice(len(population), size=config.tournament_size, replace=False)
            best_index = contestants[0]
            for index in contestants[1:]:
                if is_better_evaluation(
                    evaluations[index], evaluations[best_index], primary_objective
                ):
                    best_index = index
            return int(best_index)

        population = [sample_vector(design_variables, rng) for _ in range(population_size)]
        evaluations = [_evaluate(vector, generation=0) for vector in population]

        generation = 0
        while True:
            stop_reason = should_stop(
                history, primary_objective, config, consecutive_failures, generation=generation
            )
            if stop_reason is not None:
                return stop_reason

            generation += 1

            def _compare(i: int, j: int, _evaluations: list[DesignEvaluation] = evaluations) -> int:
                if is_better_evaluation(_evaluations[i], _evaluations[j], primary_objective):
                    return -1
                if is_better_evaluation(_evaluations[j], _evaluations[i], primary_objective):
                    return 1
                return 0

            # Rank by repeated pairwise "is better" comparison (feasibility-first,
            # then objective, then total violation) -- never a separately
            # computed scalar fitness score.
            ranked_indices = sorted(range(population_size), key=functools.cmp_to_key(_compare))
            elite_indices = ranked_indices[: config.elite_count]
            next_population = [population[index].copy() for index in elite_indices]
            next_evaluations = [evaluations[index] for index in elite_indices]

            while len(next_population) < population_size:
                stop_reason = should_stop(
                    history, primary_objective, config, consecutive_failures, generation=generation
                )
                if stop_reason is not None:
                    return stop_reason

                parent_a = population[_tournament_select(population, evaluations)]
                parent_b = population[_tournament_select(population, evaluations)]

                if rng.random() < config.crossover_probability:
                    mask = rng.random(len(design_variables)) < 0.5
                    child = np.where(mask, parent_a, parent_b)
                else:
                    child = parent_a.copy()

                mutation_mask = rng.random(len(design_variables)) < config.mutation_probability
                if np.any(mutation_mask):
                    random_vector = sample_vector(design_variables, rng)
                    child = np.where(mutation_mask, random_vector, child)
                child = np.clip(child, lower, upper)

                child_evaluation = _evaluate(child, generation=generation)
                next_population.append(child)
                next_evaluations.append(child_evaluation)

            population = next_population
            evaluations = next_evaluations


__all__ = ["GeneticAlgorithm"]
