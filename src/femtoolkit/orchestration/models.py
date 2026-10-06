"""Generic task/outcome models for independent-task execution (Version 34).

**Engineering/computational concept.** This module defines the smallest
reusable vocabulary for describing *one independent unit of work*
(:class:`TaskStatus`, :class:`TaskOutcome`) without knowing anything
about what that work actually is -- running a finite element solve,
evaluating an optimization candidate, or any other independent
computation. :mod:`femtoolkit.orchestration.manager` executes a batch of
such tasks (serially or in parallel); domain-specific layers
(:mod:`femtoolkit.orchestration.simulation` for FEA runs,
:mod:`femtoolkit.optimization.batch` for design evaluations) build their
own richer, named result types *on top of* this shared vocabulary rather
than each reimplementing status tracking, timing, and error handling
from scratch.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import Generic, TypeVar

R = TypeVar("R")


class TaskStatus(Enum):
    """The lifecycle status of one independently-executed task.

    Attributes:
        PENDING: Created but not yet submitted for execution.
        RUNNING: Currently executing (serially in the calling process,
            or in a worker process).
        COMPLETED: Finished without raising; a result value is available.
        FAILED: Raised an exception, returned a structured simulation
            failure, or exceeded its configured timeout.
        CANCELLED: Never started, or was abandoned mid-flight, because
            the batch was cancelled or another task triggered
            ``fail_fast``.
        SKIPPED: Never submitted because the batch was already stopping
            (fail-fast or cancellation) by the time this task's turn
            came up.
    """

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"


_TERMINAL_STATUSES = frozenset(
    {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED, TaskStatus.SKIPPED}
)


def is_terminal_status(status: TaskStatus) -> bool:
    """Whether ``status`` represents a task that will not change state again."""
    return status in _TERMINAL_STATUSES


@dataclass
class TaskOutcome(Generic[R]):
    """The structured outcome of running one independent task.

    Attributes:
        task_id: The originating task's unique identifier (preserved
            from the submitted task so result identity is never lost,
            regardless of completion order or which worker ran it).
        status: The task's final :class:`TaskStatus`.
        value: The task's result, if ``status == COMPLETED``. ``None``
            otherwise -- a task that failed never has a fabricated
            result standing in for a real one.
        error_type: The failing exception's type name (e.g.
            ``"ValueError"``), if ``status == FAILED``.
        error_message: A human-readable error description, if
            ``status == FAILED``.
        started_at: ISO-8601 UTC timestamp when execution began, or
            ``None`` if the task never started (``CANCELLED``/``SKIPPED``
            before submission).
        completed_at: ISO-8601 UTC timestamp when execution ended, or
            ``None`` if it never finished.
        duration_seconds: Wall-clock execution time, or ``None`` if the
            task never ran to completion.
        worker_name: An identifying label for the worker that ran this
            task (e.g. a process name), where available. ``None`` for
            serial execution, where there is only ever the calling
            process.
    """

    task_id: str
    status: TaskStatus
    value: R | None = None
    error_type: str | None = None
    error_message: str | None = None
    started_at: str | None = None
    completed_at: str | None = None
    duration_seconds: float | None = None
    worker_name: str | None = None

    @property
    def is_successful(self) -> bool:
        """Whether this task completed successfully."""
        return self.status is TaskStatus.COMPLETED


def utc_now_iso() -> str:
    """The current UTC time as an ISO-8601 string (shared timestamp format for this package)."""
    return datetime.now(UTC).isoformat()


__all__ = ["TaskOutcome", "TaskStatus", "is_terminal_status", "utc_now_iso"]
