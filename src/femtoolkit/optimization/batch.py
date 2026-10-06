"""Design-evaluation-batch integration with the generic orchestration layer (Version 34).

Builds :mod:`femtoolkit.orchestration`'s generic
:class:`~femtoolkit.orchestration.manager.ExecutionManager` on top of the
existing, unmodified :func:`~femtoolkit.optimization.evaluation.evaluate_design`
(Version 32) so a population-based algorithm
(:mod:`femtoolkit.optimization.algorithms`) can evaluate a batch of
independent candidate designs in parallel without any change to how one
single design is actually evaluated.

Extends the *evaluation layer*, not the algorithms themselves: an
algorithm still decides which candidate vectors to generate and how to
use their results (selection, crossover, mutation, environmental
selection) -- it only swaps a per-candidate loop calling
:func:`~femtoolkit.optimization.evaluation.evaluate_design` directly for
one call to :func:`evaluate_design_batch`. See each algorithm's own
module docstring for which generations are safe to batch-evaluate this
way without changing that algorithm's seeded, deterministic behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from femtoolkit.optimization.evaluation import DesignEvaluation, DesignStatus, evaluate_design
from femtoolkit.orchestration.config import OrchestrationConfig
from femtoolkit.orchestration.manager import build_execution_manager
from femtoolkit.orchestration.models import TaskStatus
from femtoolkit.orchestration.nesting import resolve_safe_project_execution
from femtoolkit.runs.manager import SimulationRunManager

if TYPE_CHECKING:
    from femtoolkit.application.project import Project
    from femtoolkit.optimization.constraints import Constraint
    from femtoolkit.optimization.objectives import Objective
    from femtoolkit.optimization.variables import DesignVariable


@dataclass(frozen=True)
class DesignEvaluationTask:
    """One independent candidate-design evaluation, as a self-contained task.

    Attributes:
        task_id: This task's unique identifier -- used as the design's
            ``design_id`` (and therefore its underlying scenario/run
            ID), so result identity always matches the originating
            candidate regardless of execution order or worker count.
        values: The candidate's variable values, keyed by design
            variable name. Must be picklable (plain floats/ints/strings
            -- true for every design variable value this toolkit
            produces).
        design_variables: The problem's design variable definitions.
        base_project: The unmodified base project every candidate
            overrides.
        objectives: The problem's objectives. Must be picklable to
            cross a process boundary for parallel evaluation -- see
            :mod:`femtoolkit.optimization.objectives`'s module
            docstring ("Picklability") for what that requires of a
            caller-supplied objective.
        constraints: The problem's constraints (same picklability note
            as ``objectives``; see
            :mod:`femtoolkit.optimization.constraints`).
        generation: The generation/iteration index this candidate
            belongs to, carried through unchanged to the resulting
            :class:`~femtoolkit.optimization.evaluation.DesignEvaluation`.
    """

    task_id: str
    values: dict[str, Any]
    design_variables: list[DesignVariable]
    base_project: Project
    objectives: list[Objective]
    constraints: list[Constraint]
    generation: int | None = None


def execute_design_evaluation_task(task: DesignEvaluationTask) -> DesignEvaluation:
    """Evaluate one :class:`DesignEvaluationTask` to completion.

    A plain, module-level function -- required for picklability when
    this runs in a worker process. Constructs a *fresh*
    :class:`~femtoolkit.runs.manager.SimulationRunManager` on every
    call (worker isolation), and applies
    :func:`~femtoolkit.orchestration.nesting.resolve_safe_project_execution`
    so a project that itself requests Version 27 parallel element
    execution never spawns a nested pool inside this worker.
    """
    project = resolve_safe_project_execution(task.base_project)
    return evaluate_design(
        task.task_id,
        task.values,
        task.design_variables,
        project,
        task.objectives,
        task.constraints,
        SimulationRunManager(),
        generation=task.generation,
    )


def _evaluation_for_incomplete_task(task: DesignEvaluationTask, message: str) -> DesignEvaluation:
    """Build a structured ``FAILED`` evaluation for a task that never produced a normal result.

    Covers what :func:`~femtoolkit.optimization.evaluation.evaluate_design`
    itself cannot report (it runs entirely in-process and only ever
    returns a value) -- a cancelled task, a timed-out task, or an
    unexpected worker failure -- keeping :func:`evaluate_design_batch`'s
    return shape uniform: always one real
    :class:`~femtoolkit.optimization.evaluation.DesignEvaluation` per
    task.
    """
    return DesignEvaluation(
        design_id=task.task_id,
        design_variables=dict(task.values),
        run_id=None,
        status=DesignStatus.FAILED,
        generation=task.generation,
        error_message=message,
    )


def evaluate_design_batch(
    tasks: list[DesignEvaluationTask],
    config: OrchestrationConfig | None = None,
) -> list[DesignEvaluation]:
    """Evaluate a batch of independent candidate designs, serially or in parallel.

    Args:
        tasks: The independent candidates to evaluate.
        config: How to run them. ``None`` (the default) evaluates
            serially, in submission order, in the calling process --
            identical to an algorithm's own per-candidate loop calling
            :func:`~femtoolkit.optimization.evaluation.evaluate_design`
            directly.

    Returns:
        One :class:`~femtoolkit.optimization.evaluation.DesignEvaluation`
        per task, in the same order as ``tasks`` when
        ``config.preserve_order`` is ``True`` (the default) -- required
        so population order (and therefore which evaluation corresponds
        to which candidate vector) is never silently reshuffled by
        parallel execution. A task that did not complete normally
        (cancelled, timed out, or an unexpected worker failure) is still
        represented by a real ``FAILED`` evaluation, never dropped or
        left as a bare exception.
    """
    manager = build_execution_manager(config)
    outcomes, _summary = manager.run_batch(
        execute_design_evaluation_task,
        tasks,
        task_ids=[task.task_id for task in tasks],
    )
    task_by_id = {task.task_id: task for task in tasks}
    evaluations: list[DesignEvaluation] = []
    for outcome in outcomes:
        task = task_by_id[outcome.task_id]
        if outcome.status is TaskStatus.COMPLETED and outcome.value is not None:
            evaluations.append(outcome.value)
            continue
        message = outcome.error_message or f"Task did not complete (status={outcome.status.value})."
        evaluations.append(_evaluation_for_incomplete_task(task, message))
    return evaluations


__all__ = [
    "DesignEvaluationTask",
    "evaluate_design_batch",
    "execute_design_evaluation_task",
]
