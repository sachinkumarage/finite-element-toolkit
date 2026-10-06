"""The execution backend abstraction (Version 34).

.. code-block:: text

    ExecutionBackend (ABC)
    |-- SerialBackend         (femtoolkit.orchestration.backends.serial)        -- default
    `-- LocalProcessBackend   (femtoolkit.orchestration.backends.local_process) -- opt-in

A backend's only job is to run a batch of independent, already-built
tasks and report a structured outcome for each -- it knows nothing about
what a task's payload actually represents (a project to simulate, a
design to evaluate, ...), and nothing about the higher-level policy
decisions (serial-vs-parallel selection, nested-parallelism downgrading)
that belong to :class:`~femtoolkit.orchestration.manager.ExecutionManager`.
Kept deliberately generic (mirroring Version 27's
:class:`~femtoolkit.execution.executor.ElementExecutor` "one interface,
swappable strategy" pattern) so a future backend -- a thread pool, or
eventually a distributed/cluster backend -- can be added without
touching any call site built against this interface (spec section 33;
no such future backend is implemented in this version).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import TypeVar

from femtoolkit.orchestration.cancellation import CancellationToken
from femtoolkit.orchestration.config import OrchestrationConfig
from femtoolkit.orchestration.models import TaskOutcome
from femtoolkit.orchestration.progress import ExecutionProgress

T = TypeVar("T")
R = TypeVar("R")

ProgressCallback = Callable[[ExecutionProgress], None]


class ExecutionBackend(ABC):
    """Runs a batch of independent tasks, serially or in parallel, with structured outcomes."""

    @abstractmethod
    def run(
        self,
        func: Callable[[T], R],
        items: Sequence[T],
        task_ids: Sequence[str],
        config: OrchestrationConfig,
        cancellation_token: CancellationToken,
        on_progress: ProgressCallback | None = None,
    ) -> list[TaskOutcome[R]]:
        """Apply ``func`` to every item in ``items``, returning one outcome per task.

        Args:
            func: A callable of one argument. Must be a picklable,
                module-level function for any backend that runs tasks in
                separate processes (see
                :mod:`femtoolkit.orchestration.backends.local_process`'s
                module docstring).
            items: The independent inputs to process, aligned
                one-to-one with ``task_ids``.
            task_ids: Each item's unique, caller-assigned identifier --
                preserved on the returned outcome regardless of
                completion order, so result identity never depends on
                execution timing.
            config: The run's orchestration configuration (fail-fast,
                timeout, worker count, ...).
            cancellation_token: Checked between task submissions; once
                set, no further tasks are submitted and already-pending
                (not yet started) tasks are marked ``CANCELLED``.
            on_progress: An optional callback invoked with an
                :class:`~femtoolkit.orchestration.progress.ExecutionProgress`
                snapshot each time a task finishes.

        Returns:
            One :class:`~femtoolkit.orchestration.models.TaskOutcome`
            per entry in ``task_ids``. If ``config.preserve_order`` is
            ``True`` (the default), the list is in the same order as
            ``task_ids``, regardless of which worker finished first --
            required whenever result *position* carries meaning (a
            parameter study's scenario order, a Monte Carlo sample
            index). If ``False``, the list is in **completion order**
            instead; every outcome still carries its own ``task_id``, so
            identity is never lost, but callers must look up results by
            ``task_id`` rather than by position.
        """
        raise NotImplementedError


__all__ = ["ExecutionBackend", "ProgressCallback"]
