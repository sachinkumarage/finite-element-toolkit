"""Multi-process execution strategy for independent simulation tasks (Version 34).

:class:`LocalProcessBackend` spreads independent tasks across several
local worker processes using the standard library's
:mod:`concurrent.futures` -- no third-party dependency, mirroring
Version 27's :class:`~femtoolkit.execution.parallel.ParallelExecutor`
(which solves the same problem one layer down, for per-element work
within a single FEA assembly; see :mod:`femtoolkit.orchestration`'s
module docstring for the distinction).

**Why processes, not threads, for this workload.** This backend targets
whole, independent simulation tasks -- each one runs real Python-level
control flow (building a scenario, orchestrating a solve) interleaved
with NumPy/SciPy calls. CPython's Global Interpreter Lock means only one
thread executes Python bytecode at a time; a thread pool would only gain
real parallelism during the portions of each task that release the GIL
(mostly the numerical kernels), not the Python-level bookkeeping around
them. A process pool sidesteps the GIL entirely -- at the cost of
needing every task's function, arguments, and result to be *picklable*
to cross the process boundary (see "Worker isolation" below).

**Worker isolation.** Each worker receives a self-contained,
serializable task (see
:mod:`femtoolkit.orchestration.simulation`/:mod:`femtoolkit.optimization.batch`
for the concrete task shapes), constructs whatever runtime state it
needs *inside* the worker, executes independently, and returns a
structured result. No GUI state, open file handle, database connection,
or other non-picklable runtime object may be part of a task -- a worker
process shares no memory with its parent beyond what was true at pool
creation time (and, on a ``spawn``-based platform, not even that).

**A fresh pool per batch.** Exactly like Version 27's
``ParallelExecutor``, a new worker pool is created for each :meth:`run`
call and always shut down before it returns (never left running after
cancellation or a timeout -- spec section 17/18's explicit "do not leave
executors or worker processes running"). This toolkit calls a batch
executor a handful of times per study, not in a tight loop, so
pool-startup overhead is not worth a persistent background pool's added
bookkeeping and crash-recovery complexity.

**Fail-fast vs. cancellation, handled differently on purpose.**
``fail_fast``'s first failure stops *future submissions* and cancels any
*not-yet-started* queued task, but lets already-running tasks finish and
their results be collected -- a deliberately softer stop that still
yields a clean, complete-as-possible result set (spec section 16: "allow
already-running tasks to finish where necessary"). An explicit
:meth:`~femtoolkit.orchestration.cancellation.CancellationToken.cancel`
call is stricter: the pool is shut down immediately
(``cancel_futures=True``), and every task that has not *already
returned* is marked ``CANCELLED`` without waiting for it (spec section
17: "do not leave ... worker processes running after cancellation").

**Timeout, approximated as "no progress."** A per-task wall-clock budget
that only that one task's worker enforces is not available from
:mod:`concurrent.futures` without extra IPC machinery this version does
not add. Instead, ``config.timeout`` bounds how long the batch may go
*without any task completing*: if an entire ``timeout``-second window
passes with zero completions, every outstanding task is treated as
timed out, the pool is shut down, and no further tasks are submitted.
This is a conservative, clearly-documented approximation, not a
per-task guarantee -- see :mod:`femtoolkit.orchestration.config`'s
``OrchestrationConfig.timeout`` docstring.

**Why the wait loop polls instead of blocking on the full timeout.**
``concurrent.futures.wait()`` is called with a short, fixed poll
interval (not ``config.timeout`` directly) so that
``cancellation_token``'s state is re-checked at least that often even
while every outstanding task is still running -- blocking on the full
``config.timeout`` (or forever, when ``config.timeout`` is ``None``)
would leave an explicit :meth:`~femtoolkit.orchestration.cancellation.CancellationToken.cancel`
call undetected until the next task happened to finish on its own,
defeating "cancel this batch now." The "no progress" timeout itself is
still measured against ``config.timeout``, accumulated in wall-clock
time since the last completion across however many poll ticks that
takes.
"""

from __future__ import annotations

import concurrent.futures
import pickle
import time
from collections.abc import Callable, Sequence
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from typing import TypeVar

from femtoolkit.exceptions import TaskSerializationError
from femtoolkit.orchestration.cancellation import CancellationToken
from femtoolkit.orchestration.config import OrchestrationConfig, resolve_max_workers
from femtoolkit.orchestration.models import TaskOutcome, TaskStatus, utc_now_iso
from femtoolkit.orchestration.progress import ExecutionProgress

T = TypeVar("T")
R = TypeVar("R")

_POLL_INTERVAL_SECONDS = 0.5
"""How often the wait loop re-checks ``cancellation_token`` while every
outstanding task is still running. Bounded above by ``config.timeout``
(when set) so a short timeout is still detected close to on time."""


class LocalProcessBackend:
    """Runs tasks across several local worker processes.

    ``func`` (and every item in ``items``, and every returned value)
    must be picklable: in practice, a plain function defined at module
    scope (importable by its ``module.qualname``), not a lambda, a
    nested function, or a bound method of an unpicklable object. NumPy
    arrays and the plain dataclasses this toolkit's projects/design
    variables are built from all pickle without special handling.
    """

    def run(
        self,
        func: Callable[[T], R],
        items: Sequence[T],
        task_ids: Sequence[str],
        config: OrchestrationConfig,
        cancellation_token: CancellationToken,
        on_progress: Callable[[ExecutionProgress], None] | None = None,
    ) -> list[TaskOutcome[R]]:
        """Apply ``func`` to every item in ``items``, across worker processes.

        See :meth:`femtoolkit.orchestration.backends.base.ExecutionBackend.run`
        for the full argument/return contract.

        Raises:
            TaskSerializationError: If ``func`` or an item cannot be
                pickled to send to a worker process. The pool is shut
                down before this is raised.
        """
        if not items:
            return []

        worker_count = resolve_max_workers(config)
        start_time = time.perf_counter()
        last_progress_time = start_time
        outcomes: dict[str, TaskOutcome[R]] = {}
        task_start_perf: dict[str, float] = {}
        queue = list(zip(task_ids, items, strict=True))
        future_to_id: dict = {}
        fail_fast_triggered = False
        pool_shutdown_already = False

        poll_interval = _POLL_INTERVAL_SECONDS
        if config.timeout is not None:
            poll_interval = min(poll_interval, config.timeout)

        pool = ProcessPoolExecutor(max_workers=worker_count)
        try:
            self._submit_up_to_capacity(
                pool, func, queue, future_to_id, outcomes, task_start_perf,
                worker_count, cancellation_token,
            )

            while future_to_id:
                if cancellation_token.is_cancelled:
                    pool.shutdown(wait=False, cancel_futures=True)
                    pool_shutdown_already = True
                    for task_id in future_to_id.values():
                        outcomes[task_id] = TaskOutcome(
                            task_id=task_id,
                            status=TaskStatus.CANCELLED,
                            error_message="Batch was cancelled.",
                        )
                    for task_id, _item in queue:
                        outcomes[task_id] = TaskOutcome(
                            task_id=task_id,
                            status=TaskStatus.CANCELLED,
                            error_message="Skipped: the batch was cancelled.",
                        )
                    future_to_id = {}
                    queue = []
                    break

                done, _pending = concurrent.futures.wait(
                    list(future_to_id), timeout=poll_interval, return_when=FIRST_COMPLETED
                )

                if not done:
                    if (
                        config.timeout is not None
                        and (time.perf_counter() - last_progress_time) >= config.timeout
                    ):
                        pool.shutdown(wait=False, cancel_futures=True)
                        pool_shutdown_already = True
                        for task_id in future_to_id.values():
                            outcomes[task_id] = TaskOutcome(
                                task_id=task_id,
                                status=TaskStatus.FAILED,
                                error_type="TaskTimeoutError",
                                error_message=(
                                    f"No task completed within the {config.timeout}s "
                                    "timeout window; the batch was stopped."
                                ),
                            )
                        for task_id, _item in queue:
                            outcomes[task_id] = TaskOutcome(
                                task_id=task_id,
                                status=TaskStatus.CANCELLED,
                                error_message="Skipped: the batch stopped due to a timeout.",
                            )
                        future_to_id = {}
                        queue = []
                        break
                    # Just a poll tick with nothing done yet -- loop back so
                    # cancellation keeps being checked even while every task
                    # is still running (see the module docstring).
                    continue

                last_progress_time = time.perf_counter()
                for future in done:
                    task_id = future_to_id.pop(future)
                    started_at = outcomes[task_id].started_at
                    outcomes[task_id] = _outcome_from_future(
                        task_id, future, started_at, task_start_perf.get(task_id)
                    )
                    if outcomes[task_id].status is TaskStatus.FAILED and config.fail_fast:
                        fail_fast_triggered = True
                    if on_progress is not None:
                        on_progress(
                            _progress_snapshot(
                                outcomes, future_to_id, len(items), start_time
                            )
                        )

                if fail_fast_triggered:
                    for task_id, _item in queue:
                        outcomes[task_id] = TaskOutcome(
                            task_id=task_id,
                            status=TaskStatus.CANCELLED,
                            error_message="Skipped: an earlier task failed (fail_fast).",
                        )
                    queue = []
                    for future, task_id in list(future_to_id.items()):
                        if future.cancel():
                            outcomes[task_id] = TaskOutcome(
                                task_id=task_id,
                                status=TaskStatus.CANCELLED,
                                error_message="Cancelled: an earlier task failed (fail_fast).",
                            )
                            del future_to_id[future]
                    # Any remaining future_to_id entries are already running; the loop
                    # continues to wait for and collect them rather than abandoning them.
                else:
                    self._submit_up_to_capacity(
                        pool, func, queue, future_to_id, outcomes, task_start_perf,
                        worker_count, cancellation_token,
                    )
        except BrokenProcessPool as error:
            pool.shutdown(wait=False, cancel_futures=True)
            pool_shutdown_already = True
            for task_id in list(future_to_id.values()):
                outcomes[task_id] = TaskOutcome(
                    task_id=task_id,
                    status=TaskStatus.FAILED,
                    error_type="BrokenProcessPool",
                    error_message=f"The worker pool crashed: {error}",
                )
            for task_id, _item in queue:
                outcomes[task_id] = TaskOutcome(
                    task_id=task_id,
                    status=TaskStatus.CANCELLED,
                    error_message="Skipped: the worker pool crashed.",
                )
        except TaskSerializationError:
            pool.shutdown(wait=False, cancel_futures=True)
            pool_shutdown_already = True
            raise
        finally:
            if not pool_shutdown_already:
                pool.shutdown(wait=True)

        if config.preserve_order:
            return [outcomes[task_id] for task_id in task_ids]
        return list(outcomes.values())

    @staticmethod
    def _submit_up_to_capacity(
        pool: ProcessPoolExecutor,
        func: Callable,
        queue: list,
        future_to_id: dict,
        outcomes: dict,
        task_start_perf: dict,
        worker_count: int,
        cancellation_token: CancellationToken,
    ) -> None:
        while queue and len(future_to_id) < worker_count:
            if cancellation_token.is_cancelled:
                return
            task_id, item = queue.pop(0)
            try:
                future = pool.submit(func, item)
            except (pickle.PicklingError, AttributeError, TypeError) as error:
                pool.shutdown(wait=False, cancel_futures=True)
                raise TaskSerializationError(
                    f"Could not send task {task_id!r} (or its function {func!r}) to a "
                    f"worker process: {error}. Both the function and every task item must "
                    "be picklable -- typically a plain module-level function and a plain "
                    "dataclass/dict of primitive values, not a lambda, closure, open file "
                    "handle, or other non-picklable runtime state."
                ) from error
            future_to_id[future] = task_id
            task_start_perf[task_id] = time.perf_counter()
            outcomes[task_id] = TaskOutcome(
                task_id=task_id, status=TaskStatus.RUNNING, started_at=utc_now_iso()
            )


def _outcome_from_future(
    task_id: str, future, started_at: str | None, task_start: float | None
) -> TaskOutcome:  # noqa: ANN001
    """Build a :class:`TaskOutcome` from a finished future.

    A pickling failure for the submitted function or one of its
    arguments does **not** surface at ``pool.submit()`` time --
    :class:`~concurrent.futures.ProcessPoolExecutor` pickles
    asynchronously, so the error only appears here, from
    ``future.result()``. Since the same unpicklable function/item would
    fail identically for *every* task in the batch, this is a batch-wide
    configuration problem, not one task's own failure -- it is re-raised
    as :exc:`~femtoolkit.exceptions.TaskSerializationError` rather than
    recorded as an ordinary per-task failure (mirroring Version 27's
    ``ParallelExecutor.map`` treatment of the same three exception
    types).

    Args:
        task_id: The task's identifier.
        future: The finished (or failed) future.
        started_at: The ISO timestamp recorded when this task was
            submitted (carried over from the ``RUNNING`` outcome so it
            is not lost on the terminal outcome).
        task_start: The :func:`time.perf_counter` reading taken at
            submission time, used to compute ``duration_seconds``.
            ``None`` only if the task's submission-time bookkeeping
            entry was somehow missing, in which case no duration is
            reported rather than a fabricated one.

    Raises:
        TaskSerializationError: If ``future.result()`` raised
            :class:`pickle.PicklingError`, :class:`AttributeError`, or
            :class:`TypeError` -- the same three types
            :mod:`concurrent.futures` surfaces a pickling failure as.
    """
    completed_at = utc_now_iso()
    duration_seconds = time.perf_counter() - task_start if task_start is not None else None
    try:
        value = future.result()
        return TaskOutcome(
            task_id=task_id,
            status=TaskStatus.COMPLETED,
            value=value,
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=duration_seconds,
        )
    except (pickle.PicklingError, AttributeError, TypeError) as error:
        raise TaskSerializationError(
            f"Could not send task {task_id!r} to, or retrieve its result from, a worker "
            f"process: {error}. The function and every task item/result must be "
            "picklable -- typically a plain module-level function and a plain "
            "dataclass/dict of primitive values, not a lambda, closure, open file handle, "
            "or other non-picklable runtime state."
        ) from error
    except Exception as error:  # noqa: BLE001 -- a worker-side failure is recorded, not raised.
        return TaskOutcome(
            task_id=task_id,
            status=TaskStatus.FAILED,
            error_type=type(error).__name__,
            error_message=str(error),
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=duration_seconds,
        )


def _progress_snapshot(
    outcomes: dict, future_to_id: dict, total: int, start_time: float
) -> ExecutionProgress:
    completed = sum(1 for o in outcomes.values() if o.status is TaskStatus.COMPLETED)
    failed = sum(1 for o in outcomes.values() if o.status is TaskStatus.FAILED)
    cancelled = sum(1 for o in outcomes.values() if o.status is TaskStatus.CANCELLED)
    return ExecutionProgress(
        total_tasks=total,
        completed_tasks=completed,
        failed_tasks=failed,
        cancelled_tasks=cancelled,
        running_tasks=len(future_to_id),
        elapsed_seconds=time.perf_counter() - start_time,
    )


__all__ = ["LocalProcessBackend"]
