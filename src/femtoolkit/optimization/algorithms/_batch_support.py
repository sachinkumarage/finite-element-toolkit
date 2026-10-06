"""Shared batched-evaluation helper for population-based algorithms (Version 34).

Every population-based algorithm (differential evolution, genetic
algorithm, particle swarm, NSGA-II) evaluates generation 0's initial
population independently -- always safe to batch regardless of that
algorithm's own per-generation loop structure (see each algorithm's
module docstring for which generations, if any, *beyond* generation 0
are also safely batchable without changing the algorithm's seeded,
deterministic behavior). This module holds the one shared
"vector batch -> :class:`~femtoolkit.optimization.batch.DesignEvaluationTask`
batch -> :func:`~femtoolkit.optimization.batch.evaluate_design_batch` ->
history-recording" routine every algorithm's batched evaluation point
calls, so there is exactly one place that bridges a population-based
algorithm's real-valued box encoding to the generic batch-evaluation
layer -- not four near-duplicate ones.
"""

from __future__ import annotations

import numpy as np

from femtoolkit.optimization.algorithms._encoding import decode_vector
from femtoolkit.optimization.batch import DesignEvaluationTask, evaluate_design_batch
from femtoolkit.optimization.evaluation import DesignEvaluation, DesignStatus
from femtoolkit.optimization.history import OptimizationHistory
from femtoolkit.optimization.problems import OptimizationProblem
from femtoolkit.orchestration.config import OrchestrationConfig


def evaluate_vector_batch(
    vectors: list[np.ndarray],
    generation: int,
    problem: OptimizationProblem,
    design_id_prefix: str,
    evaluation_index_start: int,
    history: OptimizationHistory,
    orchestration_config: OrchestrationConfig | None,
) -> list[DesignEvaluation]:
    """Decode, evaluate, and record one batch of candidate vectors.

    Args:
        vectors: The candidate vectors to evaluate, in the order their
            design IDs should be assigned.
        generation: The generation index every resulting evaluation is
            tagged with.
        problem: The problem being optimized.
        design_id_prefix: Combined with a running evaluation index for
            each design's ID (e.g. ``f"{problem.name}-ga-7"``) --
            matching every algorithm's existing per-evaluation ID
            convention, so IDs stay identical whether a run used
            per-candidate or batched evaluation.
        evaluation_index_start: The next unused evaluation index, so IDs
            stay globally unique and stable regardless of which
            evaluation points in this run were batched.
        history: Appended to, in ``vectors``' order, exactly once per
            vector -- identical to what a per-candidate serial loop
            would have recorded.
        orchestration_config: How to run this batch; ``None`` evaluates
            serially, in the calling process, in submission order.

    Returns:
        One :class:`~femtoolkit.optimization.evaluation.DesignEvaluation`
        per vector, in the same order as ``vectors``.
    """
    tasks = [
        DesignEvaluationTask(
            task_id=f"{design_id_prefix}-{evaluation_index_start + index}",
            values=decode_vector(problem.design_variables, vector),
            design_variables=problem.design_variables,
            base_project=problem.base_project,
            objectives=problem.objectives,
            constraints=problem.constraints,
            generation=generation,
        )
        for index, vector in enumerate(vectors)
    ]
    evaluations = evaluate_design_batch(tasks, config=orchestration_config)
    for evaluation in evaluations:
        history.add(evaluation)
    return evaluations


def count_consecutive_failures(evaluations: list[DesignEvaluation], previous_streak: int) -> int:
    """Recompute the consecutive-failure streak after a batch, in evaluation order.

    Walks ``evaluations`` in order (the same order they were just
    recorded into history), continuing from ``previous_streak``, exactly
    as a per-candidate serial loop would have updated the streak one
    evaluation at a time.

    Args:
        evaluations: The batch's evaluations, in submission order.
        previous_streak: The consecutive-failure count immediately
            before this batch.

    Returns:
        The consecutive-failure count after this batch.
    """
    streak = previous_streak
    for evaluation in evaluations:
        failed = evaluation.status in (DesignStatus.FAILED, DesignStatus.INVALID)
        streak = streak + 1 if failed else 0
    return streak


__all__ = ["count_consecutive_failures", "evaluate_vector_batch"]
