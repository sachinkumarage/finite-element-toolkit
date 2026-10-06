"""Particle swarm optimization: velocity-driven population search (Version 33).

.. code-block:: text

    Initialize particle positions uniformly random in bounds; velocities start at zero
            |
            v
    Evaluate every particle (generation 0); personal best = own position,
    global best = the feasibility-first best particle found so far
            |
            v
    For each iteration:
        For each particle i:
            v_i <- w*v_i + c1*r1*(pbest_i - x_i) + c2*r2*(gbest - x_i)
            v_i <- clip(v_i, velocity_limit)
            x_i <- clip(x_i + v_i, bounds)
            Evaluate x_i; update pbest_i and the shared global best if improved
                (feasibility-first, see
                 femtoolkit.optimization.evaluation.is_better_evaluation)
        Repeat until a stopping condition is reached

**The velocity update, read term by term.** ``w * v_i`` (inertia)
keeps a particle moving roughly the way it already was -- without it,
particles would jitter toward the best point found so far and lose any
ability to explore new regions. ``c1 * r1 * (pbest_i - x_i)`` (the
cognitive term) pulls a particle back toward the best position *it
personally* has found. ``c2 * r2 * (g - x_i)`` (the social term) pulls
every particle toward the best position *the whole swarm* has found.
``r1``/``r2`` are independent random numbers in ``[0, 1)`` redrawn
every step, so the pull toward each attractor varies in strength each
time rather than following a fixed deterministic path. Large ``w``
favors exploration (particles roam more freely); large ``c1``/``c2``
favor exploitation (particles converge faster toward known good
regions) -- the three coefficients trade off that balance directly,
unlike differential evolution's self-adapting step size.

Mixed-type design variables share the same real-valued box encoding
every population-based algorithm in this package uses (see
:mod:`femtoolkit.optimization.algorithms._encoding`); a velocity
applies to that encoded vector, so a categorical variable's velocity
moves it between category-list positions. Reproducible via
``config.seed``. No mathematical global-optimality guarantee is made.

**Version 34 batched evaluation: generation 0 only.** This
implementation is "asynchronous": ``global_best_position`` and each
particle's personal best are updated *immediately* after that
particle's own evaluation, inside the per-particle loop -- so particle
``i + 1``'s velocity update (which reads ``global_best_position``) can
legitimately be influenced by particle ``i``'s just-computed result
within the same iteration. That is a real, intentional property of this
PSO variant, not a bug -- evaluating a whole iteration against a frozen
snapshot of the global best would be a genuine behavioral change
(different accepted updates for the same seed) that this version does
not make. Only generation 0's initial swarm positions -- independent by
construction -- are evaluated as a batch (via
:func:`~femtoolkit.optimization.algorithms._batch_support.evaluate_vector_batch`)
when ``orchestration_config`` requests parallel execution; the
per-iteration particle loop always evaluates one particle at a time,
exactly as before.
"""

from __future__ import annotations

import numpy as np

from femtoolkit.optimization.algorithms._batch_support import (
    count_consecutive_failures,
    evaluate_vector_batch,
)
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
from femtoolkit.orchestration.config import OrchestrationConfig
from femtoolkit.runs.manager import SimulationRunManager


class ParticleSwarmOptimization(OptimizationAlgorithm):
    """Particle swarm optimization with inertia, cognitive, and social coefficients."""

    def optimize(
        self,
        problem: OptimizationProblem,
        config: OptimizationConfig,
        run_manager: SimulationRunManager,
        history: OptimizationHistory,
        starting_evaluation: DesignEvaluation | None = None,
        orchestration_config: OrchestrationConfig | None = None,
    ) -> StopReason:
        del starting_evaluation  # PSO initializes its own random swarm
        rng = np.random.default_rng(config.seed)
        primary_objective = problem.objectives[0]
        design_variables = problem.design_variables
        lower, upper = bounds_arrays(design_variables)
        span = upper - lower
        max_velocity = config.velocity_limit * span
        swarm_size = config.population_size
        consecutive_failures = 0
        evaluation_index = 0

        def _evaluate(vector: np.ndarray, generation: int) -> DesignEvaluation:
            nonlocal evaluation_index, consecutive_failures
            values = decode_vector(design_variables, vector)
            evaluation = evaluate_design(
                design_id=f"{problem.name}-pso-{evaluation_index}",
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

        positions = [sample_vector(design_variables, rng) for _ in range(swarm_size)]
        velocities = [np.zeros(len(design_variables)) for _ in range(swarm_size)]
        if orchestration_config is not None:
            evaluations = evaluate_vector_batch(
                positions, 0, problem, f"{problem.name}-pso", evaluation_index, history,
                orchestration_config,
            )
            evaluation_index += len(evaluations)
            consecutive_failures = count_consecutive_failures(evaluations, consecutive_failures)
        else:
            evaluations = [_evaluate(position, generation=0) for position in positions]

        personal_best_positions = [position.copy() for position in positions]
        personal_best_evaluations = list(evaluations)

        global_best_index = 0
        for index in range(1, swarm_size):
            if is_better_evaluation(
                evaluations[index], evaluations[global_best_index], primary_objective
            ):
                global_best_index = index
        global_best_position = positions[global_best_index].copy()
        global_best_evaluation = evaluations[global_best_index]

        generation = 0
        while True:
            stop_reason = should_stop(
                history, primary_objective, config, consecutive_failures, generation=generation
            )
            if stop_reason is not None:
                return stop_reason

            generation += 1
            for i in range(swarm_size):
                stop_reason = should_stop(
                    history, primary_objective, config, consecutive_failures, generation=generation
                )
                if stop_reason is not None:
                    return stop_reason

                r1 = rng.random(len(design_variables))
                r2 = rng.random(len(design_variables))
                cognitive = config.cognitive_coefficient * r1 * (
                    personal_best_positions[i] - positions[i]
                )
                social = config.social_coefficient * r2 * (global_best_position - positions[i])
                velocities[i] = config.inertia_weight * velocities[i] + cognitive + social
                velocities[i] = np.clip(velocities[i], -max_velocity, max_velocity)
                positions[i] = np.clip(positions[i] + velocities[i], lower, upper)

                evaluation = _evaluate(positions[i], generation=generation)
                evaluations[i] = evaluation
                if is_better_evaluation(
                    evaluation, personal_best_evaluations[i], primary_objective
                ):
                    personal_best_positions[i] = positions[i].copy()
                    personal_best_evaluations[i] = evaluation
                    if is_better_evaluation(evaluation, global_best_evaluation, primary_objective):
                        global_best_position = positions[i].copy()
                        global_best_evaluation = evaluation


__all__ = ["ParticleSwarmOptimization"]
