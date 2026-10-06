"""Bounded random search: a simple, transparent baseline algorithm (Version 32).

.. code-block:: text

    Generate candidate (uniformly random within each variable's bounds)
            |
            v
    Evaluate (build scenario, run FEA, score objectives/constraints)
            |
            v
    Record in history (best-feasible tracking happens automatically)
            |
            v
    Repeat until a stopping condition is reached

No gradient, no model of the objective's shape, no assumption of
smoothness -- every candidate is drawn independently and uniformly at
random from the design space, exactly like
:func:`~femtoolkit.uncertainty.sampling.generate_random_samples`
(Version 31) samples an uncertain parameter's distribution. This makes
random search the simplest possible baseline to compare a more
sophisticated algorithm (:mod:`.coordinate_search`, or a future one)
against, and a reasonable choice when nothing is known about the
objective's structure.
"""

from __future__ import annotations

import numpy as np

from femtoolkit.optimization.algorithms.base import (
    OptimizationAlgorithm,
    OptimizationConfig,
    StopReason,
    should_stop,
)
from femtoolkit.optimization.evaluation import DesignEvaluation, DesignStatus, evaluate_design
from femtoolkit.optimization.history import OptimizationHistory
from femtoolkit.optimization.problems import OptimizationProblem
from femtoolkit.orchestration.config import OrchestrationConfig
from femtoolkit.runs.manager import SimulationRunManager


class BoundedRandomSearch(OptimizationAlgorithm):
    """Draws each candidate design uniformly at random from the design space.

    Reproducible via ``config.seed`` -- the same seed always produces
    the same sequence of candidate designs.

    **No Version 34 batched evaluation.** Every candidate here is
    already drawn independently of every other (unlike
    :class:`~femtoolkit.optimization.algorithms.coordinate_search.CoordinateSearch`,
    whose next move depends on the previous one), so batching would in
    principle be safe -- but this version's parallel-evaluation
    integration is scoped to the population-based algorithms the spec
    names explicitly (differential evolution, genetic algorithm,
    particle swarm, NSGA-II); ``orchestration_config`` is accepted for
    interface consistency and ignored.
    """

    def optimize(
        self,
        problem: OptimizationProblem,
        config: OptimizationConfig,
        run_manager: SimulationRunManager,
        history: OptimizationHistory,
        starting_evaluation: DesignEvaluation | None = None,
        orchestration_config: OrchestrationConfig | None = None,
    ) -> StopReason:
        del starting_evaluation  # random search does not need a starting point
        del orchestration_config  # not batched in this version -- see class docstring
        rng = np.random.default_rng(config.seed)
        primary_objective = problem.objectives[0]
        consecutive_failures = 0
        evaluation_index = 0

        while True:
            stop_reason = should_stop(history, primary_objective, config, consecutive_failures)
            if stop_reason is not None:
                return stop_reason

            values = {variable.name: variable.sample(rng) for variable in problem.design_variables}
            evaluation = evaluate_design(
                design_id=f"{problem.name}-rs-{evaluation_index}",
                values=values,
                design_variables=problem.design_variables,
                base_project=problem.base_project,
                objectives=problem.objectives,
                constraints=problem.constraints,
                run_manager=run_manager,
            )
            history.add(evaluation)
            failed = evaluation.status in (DesignStatus.FAILED, DesignStatus.INVALID)
            consecutive_failures = consecutive_failures + 1 if failed else 0
            evaluation_index += 1


__all__ = ["BoundedRandomSearch"]
