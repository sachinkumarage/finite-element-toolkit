"""The execution management layer (Version 34).

This is the layer :mod:`femtoolkit.studies.runner` (Version 30),
:mod:`femtoolkit.uncertainty.monte_carlo` (Version 31), and the
Version 32/33 optimization algorithms call directly to run a batch of
independent tasks -- it is the single place the *policy* decisions live
(which backend to use, whether a parallel request must be downgraded for
nested-parallelism safety, how to time and summarize a batch), keeping
:mod:`femtoolkit.orchestration.backends` purely mechanical (just "run
these tasks, report outcomes").

.. code-block:: text

    ExecutionManager (ABC)
    |-- SerialExecutionManager    -- always runs in the calling process
    `-- ParallelExecutionManager  -- runs in worker processes, unless
                                      nested-parallelism protection
                                      (see femtoolkit.orchestration.nesting)
                                      forces it back to serial

:func:`build_execution_manager` is the one factory every domain-specific
adapter (:mod:`femtoolkit.orchestration.simulation`,
:mod:`femtoolkit.optimization.batch`) should use to turn an
:class:`~femtoolkit.orchestration.config.OrchestrationConfig` into a
concrete manager, rather than constructing
:class:`SerialExecutionManager`/:class:`ParallelExecutionManager`
directly -- this is the only place ``config.execution_mode`` is
interpreted.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import TypeVar

from femtoolkit.orchestration.backends.base import ExecutionBackend, ProgressCallback
from femtoolkit.orchestration.backends.local_process import LocalProcessBackend
from femtoolkit.orchestration.backends.serial import SerialBackend
from femtoolkit.orchestration.cancellation import CancellationToken
from femtoolkit.orchestration.config import (
    ExecutionMode,
    OrchestrationConfig,
    resolve_max_workers,
)
from femtoolkit.orchestration.models import TaskOutcome
from femtoolkit.orchestration.nesting import resolve_safe_execution_mode
from femtoolkit.orchestration.summary import ExecutionSummary, build_execution_summary

T = TypeVar("T")
R = TypeVar("R")


class ExecutionManager(ABC):
    """Runs a batch of independent tasks and reports outcomes plus a summary."""

    @abstractmethod
    def run_batch(
        self,
        func: Callable[[T], R],
        items: Sequence[T],
        task_ids: Sequence[str] | None = None,
        on_progress: ProgressCallback | None = None,
    ) -> tuple[list[TaskOutcome[R]], ExecutionSummary]:
        """Apply ``func`` to every item in ``items``.

        Args:
            func: A callable of one argument. Must be a picklable,
                module-level function if this manager ends up running in
                worker processes (see
                :mod:`femtoolkit.orchestration.backends.local_process`).
                A serial manager places no such requirement on ``func``.
            items: The independent inputs to process.
            task_ids: Each item's unique, caller-assigned identifier,
                aligned one-to-one with ``items``. If omitted, string
                indices (``"0"``, ``"1"``, ...) are generated -- callers
                that care about identity beyond position (a scenario ID,
                a sample index) should always pass their own.
            on_progress: An optional callback invoked with an
                :class:`~femtoolkit.orchestration.progress.ExecutionProgress`
                snapshot each time a task finishes.

        Returns:
            A ``(outcomes, summary)`` pair: one
            :class:`~femtoolkit.orchestration.models.TaskOutcome` per
            item (ordering governed by
            :attr:`~femtoolkit.orchestration.config.OrchestrationConfig.preserve_order`),
            and one :class:`~femtoolkit.orchestration.summary.ExecutionSummary`
            for the whole batch.
        """
        raise NotImplementedError

    @abstractmethod
    def cancel(self) -> None:
        """Request cancellation of the current (or next) :meth:`run_batch` call.

        Cooperative, not forced -- see
        :class:`~femtoolkit.orchestration.cancellation.CancellationToken`.
        Safe to call from another thread while :meth:`run_batch` is in
        flight.
        """
        raise NotImplementedError


class _BackendExecutionManager(ExecutionManager):
    """Shared plumbing for the two concrete managers.

    Both managers reduce to the same three steps -- resolve a backend,
    time the batch, build a summary -- and differ only in *which*
    backend and worker count apply; this base class holds that common
    sequence so neither concrete subclass duplicates it.
    """

    def __init__(
        self,
        backend: ExecutionBackend,
        config: OrchestrationConfig,
        execution_mode: ExecutionMode,
        worker_count: int,
        notes: list[str] | None = None,
    ) -> None:
        self._backend = backend
        self._config = config
        self._execution_mode = execution_mode
        self._worker_count = worker_count
        self._notes = list(notes or [])
        self._cancellation_token = CancellationToken()

    def cancel(self) -> None:
        self._cancellation_token.cancel()

    def run_batch(
        self,
        func: Callable[[T], R],
        items: Sequence[T],
        task_ids: Sequence[str] | None = None,
        on_progress: ProgressCallback | None = None,
    ) -> tuple[list[TaskOutcome[R]], ExecutionSummary]:
        if task_ids is None:
            task_ids = [str(index) for index in range(len(items))]
        start_time = time.perf_counter()
        outcomes = self._backend.run(
            func, items, task_ids, self._config, self._cancellation_token, on_progress
        )
        elapsed_seconds = time.perf_counter() - start_time
        summary = build_execution_summary(
            outcomes, self._execution_mode, self._worker_count, elapsed_seconds,
            notes=self._notes,
        )
        return outcomes, summary


class SerialExecutionManager(_BackendExecutionManager):
    """Always runs every task in the calling process -- the toolkit's default.

    Reproduces every pre-Version-34 workflow's exact behavior: no
    picklability requirement, no worker processes, no change to existing
    results for callers that never opt into parallel execution.
    """

    def __init__(self, config: OrchestrationConfig | None = None) -> None:
        super().__init__(
            SerialBackend(),
            config or OrchestrationConfig(execution_mode="serial"),
            execution_mode="serial",
            worker_count=1,
        )


class ParallelExecutionManager(_BackendExecutionManager):
    """Runs tasks across local worker processes, unless nesting safety forces serial.

    Before honoring ``config.execution_mode == "parallel"``, consults
    :func:`~femtoolkit.orchestration.nesting.resolve_safe_execution_mode`;
    if the calling code is already running inside a worker process, this
    manager transparently falls back to a
    :class:`~femtoolkit.orchestration.backends.serial.SerialBackend`
    ("prefer the safest behavior" -- downgrade rather than reject), and
    records that downgrade in the resulting
    :class:`~femtoolkit.orchestration.summary.ExecutionSummary`'s
    ``notes``.
    """

    def __init__(self, config: OrchestrationConfig) -> None:
        safe_mode = resolve_safe_execution_mode(config.execution_mode)
        notes: list[str] = []
        if safe_mode != config.execution_mode:
            notes.append(
                "Parallel execution was requested but downgraded to serial because "
                "this batch was already running inside a worker process (nested-"
                "parallelism protection)."
            )
            backend: ExecutionBackend = SerialBackend()
            worker_count = 1
        else:
            backend = LocalProcessBackend()
            worker_count = resolve_max_workers(config)
        super().__init__(backend, config, safe_mode, worker_count, notes=notes)


def build_execution_manager(config: OrchestrationConfig | None = None) -> ExecutionManager:
    """Build the :class:`ExecutionManager` that ``config`` describes.

    Args:
        config: The orchestration configuration. ``None`` (the default)
            is equivalent to ``OrchestrationConfig()`` -- serial
            execution, matching every prior version's behavior.

    Returns:
        A :class:`SerialExecutionManager` if ``config.execution_mode ==
        "serial"``, otherwise a :class:`ParallelExecutionManager` (which
        may itself still run serially, if nested-parallelism protection
        applies).
    """
    resolved_config = config or OrchestrationConfig()
    if resolved_config.execution_mode == "serial":
        return SerialExecutionManager(resolved_config)
    return ParallelExecutionManager(resolved_config)


__all__ = [
    "ExecutionManager",
    "ParallelExecutionManager",
    "SerialExecutionManager",
    "build_execution_manager",
]
