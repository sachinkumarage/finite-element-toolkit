"""Execution summary and performance metrics for a completed task batch (Version 34).

Reuses Version 27's :func:`~femtoolkit.performance.benchmark.speedup`
and :func:`~femtoolkit.performance.benchmark.parallel_efficiency`
directly -- this module computes no new performance-metric formula of
its own, only assembles the inputs those existing functions need from a
completed batch's :class:`~femtoolkit.orchestration.models.TaskOutcome`
list.

.. math::

    S = T_{serial} / T_{parallel} \\qquad E = S / P

``S`` (speedup) and ``E`` (parallel efficiency, ``P`` = worker count) are
only meaningful when an honest estimate of the equivalent serial time
exists; for a batch actually run serially, no such comparison applies
(there is no "parallel" time to compare against), so
:attr:`ExecutionSummary.parallel_speedup`/:attr:`parallel_efficiency`
are ``None`` in that case rather than a fabricated ``1.0``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from femtoolkit.orchestration.config import ExecutionMode
from femtoolkit.orchestration.models import TaskOutcome, TaskStatus


@dataclass(frozen=True)
class ExecutionSummary:
    """A structured summary of one completed task batch.

    Attributes:
        total_tasks: Total number of tasks submitted.
        completed_tasks: How many completed successfully.
        failed_tasks: How many failed.
        cancelled_tasks: How many were cancelled or skipped.
        execution_mode: ``"serial"`` or ``"parallel"`` -- the mode
            actually used (after any nested-parallelism downgrade; see
            :mod:`femtoolkit.orchestration.nesting`).
        worker_count: How many workers were used (``1`` for serial).
        total_elapsed_seconds: Wall-clock time for the whole batch.
        average_task_seconds: Mean per-task duration among tasks that
            ran to completion (succeeded or failed; a cancelled/skipped
            task that never ran contributes no duration). ``None`` if no
            task ran.
        fastest_task_seconds: The shortest observed task duration, or
            ``None`` if no task ran.
        slowest_task_seconds: The longest observed task duration, or
            ``None`` if no task ran.
        serial_estimated_seconds: For a parallel batch, an estimate of
            how long the same tasks would have taken run serially
            (``average_task_seconds * total_tasks``) -- an estimate, not
            a measurement; it assumes uniform task cost and ignores
            process-startup/serialization overhead that a *real* serial
            run would not pay. ``None`` for a batch that was already run
            serially (no estimate is needed -- the real time is already
            known).
        parallel_speedup: ``S = T_serial_estimated / T_parallel``,
            via :func:`femtoolkit.performance.benchmark.speedup`.
            ``None`` unless this batch ran in parallel.
        parallel_efficiency: ``E = S / worker_count``, via
            :func:`femtoolkit.performance.benchmark.parallel_efficiency`.
            ``None`` unless this batch ran in parallel.
        notes: Free-text caveats worth surfacing alongside the numbers
            above (e.g. a nested-parallelism downgrade, or an
            unpicklable-objective fallback to serial execution) -- never
            parsed programmatically, purely for a human reading a report
            or console summary.
    """

    total_tasks: int
    completed_tasks: int
    failed_tasks: int
    cancelled_tasks: int
    execution_mode: ExecutionMode
    worker_count: int
    total_elapsed_seconds: float
    average_task_seconds: float | None
    fastest_task_seconds: float | None
    slowest_task_seconds: float | None
    serial_estimated_seconds: float | None = None
    parallel_speedup: float | None = None
    parallel_efficiency: float | None = None
    notes: list[str] = field(default_factory=list)


def build_execution_summary(
    outcomes: list[TaskOutcome],
    execution_mode: ExecutionMode,
    worker_count: int,
    total_elapsed_seconds: float,
    notes: list[str] | None = None,
) -> ExecutionSummary:
    """Assemble an :class:`ExecutionSummary` from a completed batch's outcomes.

    Args:
        outcomes: Every task's :class:`~femtoolkit.orchestration.models.TaskOutcome`,
            in any order.
        execution_mode: The execution mode actually used for this batch.
        worker_count: How many workers were used (``1`` for serial).
        total_elapsed_seconds: The whole batch's measured wall-clock time.
        notes: Optional human-readable caveats to attach (see
            :attr:`ExecutionSummary.notes`).

    Returns:
        A fully populated :class:`ExecutionSummary`.
    """
    from femtoolkit.performance.benchmark import parallel_efficiency as _parallel_efficiency
    from femtoolkit.performance.benchmark import speedup as _speedup

    completed = sum(1 for o in outcomes if o.status is TaskStatus.COMPLETED)
    failed = sum(1 for o in outcomes if o.status is TaskStatus.FAILED)
    cancelled = sum(
        1 for o in outcomes if o.status in (TaskStatus.CANCELLED, TaskStatus.SKIPPED)
    )
    durations = [o.duration_seconds for o in outcomes if o.duration_seconds is not None]

    average_task_seconds = sum(durations) / len(durations) if durations else None
    fastest_task_seconds = min(durations) if durations else None
    slowest_task_seconds = max(durations) if durations else None

    serial_estimated_seconds: float | None = None
    speedup_value: float | None = None
    efficiency_value: float | None = None
    if execution_mode == "parallel" and average_task_seconds is not None:
        serial_estimated_seconds = average_task_seconds * len(outcomes)
        if serial_estimated_seconds > 0.0 and total_elapsed_seconds > 0.0:
            speedup_value = _speedup(serial_estimated_seconds, total_elapsed_seconds)
            efficiency_value = _parallel_efficiency(speedup_value, worker_count)

    return ExecutionSummary(
        total_tasks=len(outcomes),
        completed_tasks=completed,
        failed_tasks=failed,
        cancelled_tasks=cancelled,
        execution_mode=execution_mode,
        worker_count=worker_count,
        total_elapsed_seconds=total_elapsed_seconds,
        average_task_seconds=average_task_seconds,
        fastest_task_seconds=fastest_task_seconds,
        slowest_task_seconds=slowest_task_seconds,
        serial_estimated_seconds=serial_estimated_seconds,
        parallel_speedup=speedup_value,
        parallel_efficiency=efficiency_value,
        notes=list(notes or []),
    )


def execution_summary_to_dict(summary: ExecutionSummary) -> dict[str, object]:
    """Reduce an :class:`ExecutionSummary` to the small dict Version 30 run-history persists.

    :mod:`femtoolkit.runs.history` (the simulation core's run-history
    layer) must not import this package -- see
    :mod:`femtoolkit.orchestration`'s module docstring on that layering
    -- so a caller that wants to attach a batch's execution context to a
    :class:`~femtoolkit.runs.history.RunRecord` calls this function
    first and passes the resulting plain dict to
    :func:`~femtoolkit.runs.history.record_from_run`'s
    ``execution_metadata`` argument.

    Args:
        summary: The batch's execution summary.

    Returns:
        A plain dict with keys ``execution_mode``, ``worker_count``,
        ``total_tasks``, ``completed_tasks``, ``failed_tasks``,
        ``cancelled_tasks``, and ``total_elapsed_seconds`` -- every
        value JSON-serializable, intentionally a small subset of
        ``summary``'s own fields (no per-task timing statistics or
        speedup/efficiency -- those belong to display/reporting, not
        persisted run-history metadata).
    """
    return {
        "execution_mode": summary.execution_mode,
        "worker_count": summary.worker_count,
        "total_tasks": summary.total_tasks,
        "completed_tasks": summary.completed_tasks,
        "failed_tasks": summary.failed_tasks,
        "cancelled_tasks": summary.cancelled_tasks,
        "total_elapsed_seconds": summary.total_elapsed_seconds,
    }


__all__ = ["ExecutionSummary", "build_execution_summary", "execution_summary_to_dict"]
