"""Differential evolution: a population-based, derivative-free search (Version 33).

.. code-block:: text

    Initialize a population of NP candidate vectors, uniformly random in bounds
            |
            v
    Evaluate every candidate (generation 0)
            |
            v
    For each generation:
        For each target vector x_i in the population:
            Pick three OTHER distinct population members x_r1, x_r2, x_r3
            Mutant vector:  v = x_r1 + F * (x_r2 - x_r3), clipped to bounds
            Binomial crossover between v and x_i -> trial vector u
            Evaluate u; keep whichever of {x_i, u} is better
                (feasibility-first, see
                 femtoolkit.optimization.evaluation.is_better_evaluation)
        Repeat until a stopping condition is reached

**Why a population, and why this mutation scheme.** A single-point
search (coordinate search, Version 32) can only ever look at the
neighborhood of its current design. A *population* of candidates lets
the search explore several regions of the design space at once
(exploration) while the mutation/crossover/selection cycle below
concentrates the population toward better regions over successive
generations (exploitation) -- without ever needing a gradient, which
may not exist or may be expensive/impossible to compute for a real FEA
model. Differential evolution's mutation vector ``v = x_r1 + F*(x_r2 -
x_r3)`` uses the population's own spread as a self-adapting step size:
early on, when candidates are spread out, steps are large
(exploration); as the population converges, the same formula
automatically takes smaller steps (exploitation) -- no step-size
schedule needs to be hand-tuned.

Mixed-type design variables (continuous, integer, categorical) are all
searched in one shared real-valued "box" encoding (see
:mod:`femtoolkit.optimization.algorithms._encoding`): a categorical
variable is treated as a position in its category list, and mutation
can move it to a different list position just like any other gene.

Reproducible via ``config.seed``. No mathematical global-optimality
guarantee is made or implied -- this is a well-known, effective
heuristic, not a convergence proof.
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
from femtoolkit.optimization.evaluation import (
    DesignEvaluation,
    DesignStatus,
    evaluate_design,
    is_better_evaluation,
)
from femtoolkit.optimization.history import OptimizationHistory
from femtoolkit.optimization.problems import OptimizationProblem
from femtoolkit.runs.manager import SimulationRunManager


class DifferentialEvolution(OptimizationAlgorithm):
    """Differential evolution (DE/rand/1/bin): population-based derivative-free search."""

    def optimize(
        self,
        problem: OptimizationProblem,
        config: OptimizationConfig,
        run_manager: SimulationRunManager,
        history: OptimizationHistory,
        starting_evaluation: DesignEvaluation | None = None,
    ) -> StopReason:
        del starting_evaluation  # DE initializes its own random population
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
                design_id=f"{problem.name}-de-{evaluation_index}",
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

        population = [sample_vector(design_variables, rng) for _ in range(population_size)]
        population_evaluations = [_evaluate(vector, generation=0) for vector in population]

        generation = 0
        while True:
            stop_reason = should_stop(
                history, primary_objective, config, consecutive_failures, generation=generation
            )
            if stop_reason is not None:
                return stop_reason

            generation += 1
            for i in range(population_size):
                stop_reason = should_stop(
                    history, primary_objective, config, consecutive_failures, generation=generation
                )
                if stop_reason is not None:
                    return stop_reason

                others = [j for j in range(population_size) if j != i]
                r1, r2, r3 = rng.choice(others, size=3, replace=False)
                mutant = population[r1] + config.mutation_factor * (population[r2] - population[r3])
                mutant = np.clip(mutant, lower, upper)

                trial = population[i].copy()
                forced_index = rng.integers(0, len(design_variables))
                crossover_mask = rng.random(len(design_variables)) < config.crossover_probability
                crossover_mask[forced_index] = True
                trial[crossover_mask] = mutant[crossover_mask]

                trial_evaluation = _evaluate(trial, generation=generation)
                if is_better_evaluation(
                    trial_evaluation, population_evaluations[i], primary_objective
                ):
                    population[i] = trial
                    population_evaluations[i] = trial_evaluation


__all__ = ["DifferentialEvolution"]
