"""FEA simulation-task integration with the generic orchestration layer (Version 34).

Builds :mod:`femtoolkit.orchestration`'s generic
:class:`~femtoolkit.orchestration.manager.ExecutionManager` on top of the
existing, unmodified :class:`~femtoolkit.runs.manager.SimulationRunManager`
(Version 30) so parameter studies
(:mod:`femtoolkit.studies.runner`) and Monte Carlo studies
(:mod:`femtoolkit.uncertainty.monte_carlo`) can request parallel
execution without any change to how one single simulation is actually
run.

.. code-block:: text

    SerialExecutionManager   -> SimulationRunManager
    ParallelExecutionManager -> Worker -> SimulationRunManager

The same underlying execution logic (:meth:`~femtoolkit.runs.manager.SimulationRunManager.execute`)
is reused in both cases for consistency -- :class:`SimulationRunManager`
itself is not replaced or subclassed.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from femtoolkit.application.project import Project
from femtoolkit.orchestration.config import OrchestrationConfig
from femtoolkit.orchestration.manager import build_execution_manager
from femtoolkit.orchestration.models import TaskStatus
from femtoolkit.orchestration.nesting import resolve_safe_project_execution
from femtoolkit.orchestration.progress import ExecutionProgress
from femtoolkit.orchestration.summary import ExecutionSummary
from femtoolkit.runs.manager import SimulationRunManager
from femtoolkit.runs.models import RunStatus, SimulationRun, new_run_id, snapshot_configuration

ProgressCallback = Callable[[ExecutionProgress], None]


@dataclass(frozen=True)
class SimulationTask:
    """One independent whole-simulation unit of work.

    Attributes:
        task_id: This task's unique identifier -- every integration in
            this toolkit sets it equal to ``scenario_id``, so result
            identity always matches the originating scenario regardless
            of execution order or worker count.
        project: The fully-resolved project to simulate -- any scenario
            override must already be applied by the caller (see
            :func:`~femtoolkit.studies.scenarios.apply_scenario`); a
            worker performs no further resolution. Must be picklable to
            cross a process boundary for parallel execution --
            :class:`~femtoolkit.application.project.Project` and its
            sub-configs already are (plain dataclasses of primitives).
        scenario_id: Passed straight through to
            :meth:`~femtoolkit.runs.manager.SimulationRunManager.execute`,
            preserved on the resulting
            :class:`~femtoolkit.runs.models.SimulationRun`.
    """

    task_id: str
    project: Project
    scenario_id: str | None = None


def execute_simulation_task(task: SimulationTask) -> SimulationRun:
    """Run one :class:`SimulationTask` to completion.

    A plain, module-level function -- required for picklability when
    this runs in a worker process (see
    :mod:`femtoolkit.orchestration.backends.local_process`'s module
    docstring) -- that constructs a *fresh*
    :class:`~femtoolkit.runs.manager.SimulationRunManager` on every
    call, so no run-manager state is shared across tasks, let alone
    across worker processes (worker isolation). Also applies
    :func:`~femtoolkit.orchestration.nesting.resolve_safe_project_execution`
    so a project that itself requests Version 27 parallel element
    execution never spawns a nested pool inside this worker.
    """
    project = resolve_safe_project_execution(task.project)
    return SimulationRunManager().execute(project, scenario_id=task.scenario_id)


def _run_for_incomplete_task(
    task: SimulationTask, status: RunStatus, message: str
) -> SimulationRun:
    """Build a structured run for a task that never produced a normal result.

    :meth:`~femtoolkit.runs.manager.SimulationRunManager.execute` itself
    only returns a structured ``FAILED`` run for an *ordinary* solver
    failure (invalid configuration or a solver error) -- it never
    raises for those. This helper covers what that function's contract
    does not: a cancelled task, a timed-out task, or a worker crash
    reported by the orchestration layer. It exists so
    :func:`evaluate_simulation_batch` always returns one real
    :class:`~femtoolkit.runs.models.SimulationRun` per task -- never a
    bare exception or a missing entry -- keeping its return shape
    identical to :meth:`~femtoolkit.studies.runner.StudyRunner.run`'s
    existing serial behavior.
    """
    return SimulationRun(
        run_id=new_run_id(),
        project_id=task.project.project_id,
        configuration_snapshot=snapshot_configuration(task.project),
        scenario_id=task.scenario_id,
        status=status,
        error_stage=None if status is RunStatus.CANCELLED else "solve",
        error_message=message,
    )


def evaluate_simulation_batch(
    tasks: list[SimulationTask],
    config: OrchestrationConfig | None = None,
    on_progress: ProgressCallback | None = None,
) -> tuple[list[SimulationRun], ExecutionSummary]:
    """Run a batch of independent :class:`SimulationTask` objects, serially or in parallel.

    Args:
        tasks: The independent simulations to run.
        config: How to run them. ``None`` (the default) runs serially,
            in submission order, in the calling process -- identical to
            every prior version's behavior.
        on_progress: An optional progress callback.

    Returns:
        A ``(runs, summary)`` pair: one
        :class:`~femtoolkit.runs.models.SimulationRun` per task. When
        ``config.preserve_order`` is ``True`` (the default), ``runs`` is
        in the same order as ``tasks`` regardless of which worker
        finished first -- required so a parameter study's scenario
        order, or a Monte Carlo sample's index, is never silently
        reshuffled by parallel execution. A task that did not complete
        normally (cancelled, timed out, or an unexpected worker
        failure) is still represented by a real
        ``FAILED``/``CANCELLED`` :class:`~femtoolkit.runs.models.SimulationRun`
        (see :func:`_run_for_incomplete_task`), never dropped or left as
        a bare exception. Also returns the batch's
        :class:`~femtoolkit.orchestration.summary.ExecutionSummary`.
    """
    manager = build_execution_manager(config)
    outcomes, summary = manager.run_batch(
        execute_simulation_task,
        tasks,
        task_ids=[task.task_id for task in tasks],
        on_progress=on_progress,
    )
    task_by_id = {task.task_id: task for task in tasks}
    runs: list[SimulationRun] = []
    for outcome in outcomes:
        task = task_by_id[outcome.task_id]
        if outcome.status is TaskStatus.COMPLETED and outcome.value is not None:
            runs.append(outcome.value)
            continue
        message = outcome.error_message or f"Task did not complete (status={outcome.status.value})."
        status = (
            RunStatus.CANCELLED
            if outcome.status in (TaskStatus.CANCELLED, TaskStatus.SKIPPED)
            else RunStatus.FAILED
        )
        runs.append(_run_for_incomplete_task(task, status, message))
    return runs, summary


__all__ = [
    "SimulationTask",
    "evaluate_simulation_batch",
    "execute_simulation_task",
]
