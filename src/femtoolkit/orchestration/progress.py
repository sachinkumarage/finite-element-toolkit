"""Structured, GUI-independent progress reporting (Version 34).

:class:`ExecutionProgress` is a plain, immutable snapshot of how a
running batch is doing -- never tied to any particular display. A caller
can render it however it needs: a GUI progress bar
(:mod:`femtoolkit.gui.workflow_pages`), a console line
(:func:`format_console_progress`), or nothing at all for a script that
only cares about the final :class:`~femtoolkit.orchestration.summary.ExecutionSummary`.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExecutionProgress:
    """A snapshot of one execution batch's progress at one point in time.

    Attributes:
        total_tasks: Total number of tasks in the batch.
        completed_tasks: How many have finished successfully so far.
        failed_tasks: How many have failed so far.
        cancelled_tasks: How many were cancelled or skipped so far.
        running_tasks: How many are currently executing.
        elapsed_seconds: Wall-clock time since the batch started.
    """

    total_tasks: int
    completed_tasks: int
    failed_tasks: int
    cancelled_tasks: int
    running_tasks: int
    elapsed_seconds: float

    @property
    def finished_tasks(self) -> int:
        """How many tasks have reached a terminal state (completed, failed, or cancelled)."""
        return self.completed_tasks + self.failed_tasks + self.cancelled_tasks

    @property
    def percentage(self) -> float:
        """What fraction of the batch has finished, as a percentage in ``[0, 100]``."""
        if self.total_tasks <= 0:
            return 100.0
        return 100.0 * self.finished_tasks / self.total_tasks

    @property
    def estimated_remaining_seconds(self) -> float | None:
        """A simple linear estimate of remaining time, or ``None`` if not yet estimable.

        Returns ``None`` until at least one task has finished (there is
        no rate to extrapolate from yet). The estimate assumes the
        remaining tasks take, on average, as long as the finished ones
        have so far -- a rough guide for a progress display, not a
        precise prediction (task cost can vary, and this says nothing
        about load imbalance across workers).
        """
        if self.finished_tasks <= 0:
            return None
        average_seconds = self.elapsed_seconds / self.finished_tasks
        remaining = self.total_tasks - self.finished_tasks
        return max(0.0, average_seconds * remaining)


def format_console_progress(progress: ExecutionProgress) -> str:
    """Render one :class:`ExecutionProgress` snapshot as a single console-friendly line.

    Args:
        progress: The progress snapshot to render.

    Returns:
        A line such as ``"Progress: 42/100 (42.0%) | Completed: 39 | Failed: 2 |
        Running: 1 | Elapsed: 18.4s"``. No external formatting/UI
        dependency is used.
    """
    return (
        f"Progress: {progress.finished_tasks}/{progress.total_tasks} "
        f"({progress.percentage:.1f}%) | Completed: {progress.completed_tasks} | "
        f"Failed: {progress.failed_tasks} | Cancelled: {progress.cancelled_tasks} | "
        f"Running: {progress.running_tasks} | Elapsed: {progress.elapsed_seconds:.1f}s"
    )


__all__ = ["ExecutionProgress", "format_console_progress"]
