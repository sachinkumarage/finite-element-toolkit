"""NSGA-II: a practical multi-objective evolutionary algorithm (Version 33).

.. code-block:: text

    Initialize a population of candidate vectors, uniformly random in bounds
            |
            v
    Evaluate every candidate (generation 0); rank into fronts
    (femtoolkit.optimization.pareto.fast_non_dominated_sort) and compute
    each front's crowding distance
            |
            v
    For each generation:
        Build an offspring population the same size as the parent population:
            Select two parents by binary tournament (lower rank wins;
            ties broken by larger crowding distance -- "less crowded")
            Crossover (uniform) + mutation (per-gene random reset)
            Evaluate the child
        Combine parents + offspring; re-rank into fronts; keep the best
        `population_size` individuals, filling the last admitted front
        by crowding distance (never arbitrarily)
        Repeat until a stopping condition is reached

**Why NSGA-II, briefly.** A single-objective algorithm needs one
scalar "better than" comparison; a *multi-objective* problem has no
single such comparison (minimizing mass and minimizing displacement
can disagree about which of two designs is "better"). NSGA-II solves
this with two ideas working together: **non-dominated sorting**
partitions a population into successive Pareto fronts (the first front
is every design nothing else in the population dominates -- see
:func:`~femtoolkit.optimization.pareto.dominates`: design *A*
dominates *B* if *A* is no worse on every objective and strictly
better on at least one), giving every individual a *rank*; **crowding
distance** then measures how isolated each individual is within its
own front, so that when only part of a front can survive into the next
generation, the survivors are spread across the whole trade-off curve
rather than clustering in one region. Both this algorithm's variation
operators (crossover, mutation) and its constraint handling reuse
exactly the same building blocks the rest of this package already
uses -- :func:`~femtoolkit.optimization.pareto.constrained_dominates`
folds the feasibility-first principle
(:func:`~femtoolkit.optimization.evaluation.is_better_evaluation`)
directly into the dominance check used for ranking, so an infeasible
design is never preferred over a feasible one regardless of objective
values.

**This implementation never selects one "best" solution.** The
algorithm's job is to produce a good approximation of the Pareto
front; which trade-off to actually build is an engineering judgment
this package leaves entirely to the caller (see
:meth:`~femtoolkit.optimization.results.OptimizationResult.pareto_front`).

Mixed-type design variables share the same real-valued box encoding
every population-based algorithm in this package uses (see
:mod:`femtoolkit.optimization.algorithms._encoding`). Reproducible via
``config.seed``. No mathematical global-optimality guarantee is made.
"""

from __future__ import annotations

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
from femtoolkit.optimization.evaluation import DesignEvaluation, DesignStatus, evaluate_design
from femtoolkit.optimization.history import OptimizationHistory
from femtoolkit.optimization.pareto import crowding_distance, fast_non_dominated_sort
from femtoolkit.optimization.problems import OptimizationProblem
from femtoolkit.runs.manager import SimulationRunManager


class NSGA2(OptimizationAlgorithm):
    """NSGA-II: non-dominated sorting + crowding distance, with feasibility-first ranking."""

    def optimize(
        self,
        problem: OptimizationProblem,
        config: OptimizationConfig,
        run_manager: SimulationRunManager,
        history: OptimizationHistory,
        starting_evaluation: DesignEvaluation | None = None,
    ) -> StopReason:
        del starting_evaluation  # NSGA-II initializes its own random population
        rng = np.random.default_rng(config.seed)
        primary_objective = problem.objectives[0]
        objectives = problem.objectives
        design_variables = problem.design_variables
        lower, upper = bounds_arrays(design_variables)
        population_size = config.population_size
        consecutive_failures = 0
        evaluation_index = 0

        def _evaluate(vector: np.ndarray, generation: int) -> DesignEvaluation:
            nonlocal evaluation_index, consecutive_failures
            values = decode_vector(design_variables, vector)
            evaluation = evaluate_design(
                design_id=f"{problem.name}-nsga2-{evaluation_index}",
                values=values,
                design_variables=design_variables,
                base_project=problem.base_project,
                objectives=objectives,
                constraints=problem.constraints,
                run_manager=run_manager,
                generation=generation,
            )
            history.add(evaluation)
            evaluation_index += 1
            failed = evaluation.status in (DesignStatus.FAILED, DesignStatus.INVALID)
            consecutive_failures = consecutive_failures + 1 if failed else 0
            return evaluation

        def _rank_and_crowding(
            evaluations: list[DesignEvaluation],
        ) -> tuple[list[list[DesignEvaluation]], dict[str, int], dict[str, float]]:
            fronts = fast_non_dominated_sort(evaluations, objectives)
            rank_of: dict[str, int] = {}
            crowding_of: dict[str, float] = {}
            for rank, front in enumerate(fronts):
                front_crowding = crowding_distance(front, objectives)
                for evaluation in front:
                    rank_of[evaluation.design_id] = rank
                    crowding_of[evaluation.design_id] = front_crowding[evaluation.design_id]
            return fronts, rank_of, crowding_of

        def _crowded_tournament(
            vectors: list[np.ndarray],
            evaluations: list[DesignEvaluation],
            rank_of: dict[str, int],
            crowding_of: dict[str, float],
        ) -> np.ndarray:
            i, j = rng.choice(len(vectors), size=2, replace=False)
            evaluation_i, evaluation_j = evaluations[i], evaluations[j]
            rank_i, rank_j = rank_of[evaluation_i.design_id], rank_of[evaluation_j.design_id]
            if rank_i != rank_j:
                return vectors[i] if rank_i < rank_j else vectors[j]
            crowd_i = crowding_of[evaluation_i.design_id]
            crowd_j = crowding_of[evaluation_j.design_id]
            return vectors[i] if crowd_i >= crowd_j else vectors[j]

        population_vectors = [sample_vector(design_variables, rng) for _ in range(population_size)]
        population_evaluations = [
            _evaluate(vector, generation=0) for vector in population_vectors
        ]
        _, rank_of, crowding_of = _rank_and_crowding(population_evaluations)

        generation = 0
        while True:
            stop_reason = should_stop(
                history, primary_objective, config, consecutive_failures, generation=generation
            )
            if stop_reason is not None:
                return stop_reason

            generation += 1
            offspring_vectors: list[np.ndarray] = []
            offspring_evaluations: list[DesignEvaluation] = []
            while len(offspring_vectors) < population_size:
                stop_reason = should_stop(
                    history, primary_objective, config, consecutive_failures, generation=generation
                )
                if stop_reason is not None:
                    return stop_reason

                parent_a = _crowded_tournament(
                    population_vectors, population_evaluations, rank_of, crowding_of
                )
                parent_b = _crowded_tournament(
                    population_vectors, population_evaluations, rank_of, crowding_of
                )
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

                offspring_vectors.append(child)
                offspring_evaluations.append(_evaluate(child, generation=generation))

            combined_vectors = population_vectors + offspring_vectors
            combined_evaluations = population_evaluations + offspring_evaluations
            vector_by_id = {
                evaluation.design_id: vector
                for vector, evaluation in zip(combined_vectors, combined_evaluations, strict=True)
            }

            fronts = fast_non_dominated_sort(combined_evaluations, objectives)
            next_evaluations: list[DesignEvaluation] = []
            rank_of = {}
            crowding_of = {}
            for rank, front in enumerate(fronts):
                front_crowding = crowding_distance(front, objectives)
                if len(next_evaluations) + len(front) <= population_size:
                    selected_front = front
                else:
                    remaining = population_size - len(next_evaluations)
                    selected_front = sorted(
                        front, key=lambda e: -front_crowding[e.design_id]
                    )[:remaining]
                for evaluation in selected_front:
                    rank_of[evaluation.design_id] = rank
                    crowding_of[evaluation.design_id] = front_crowding[evaluation.design_id]
                    next_evaluations.append(evaluation)
                if len(next_evaluations) >= population_size:
                    break

            population_evaluations = next_evaluations
            population_vectors = [
                vector_by_id[evaluation.design_id] for evaluation in population_evaluations
            ]


__all__ = ["NSGA2"]
