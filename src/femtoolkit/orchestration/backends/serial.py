"""The default, zero-overhead execution strategy (Version 34).

:class:`SerialBackend` runs every task in the calling process, in
submission order -- exactly what every workflow in this toolkit already
did through Version 33. It exists so code written against
:class:`~femtoolkit.orchestration.backends.base.ExecutionBackend` has a
real implementation to use by default, without an ``if parallel: ...
else: ...`` branch at every call site, mirroring Version 27's
:class:`~femtoolkit.execution.serial.SerialExecutor`.
"""

from __future__ import annotations

import time
import traceback
from collections.abc import Callable, Sequence
from typing import TypeVar

from femtoolkit.orchestration.cancellation import CancellationToken
from femtoolkit.orchestration.config import OrchestrationConfig
from femtoolkit.orchestration.models import TaskOutcome, TaskStatus, utc_now_iso
from femtoolkit.orchestration.progress import ExecutionProgress

T = TypeVar("T")
R = TypeVar("R")


class SerialBackend:
    """Runs every task in the calling process, one at a time, in order.

    Places no picklability requirement on ``func`` or its items -- unlike
    :class:`~femtoolkit.orchestration.backends.local_process.LocalProcessBackend`,
    nothing here crosses a process boundary.
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
        """Apply ``func`` to every item in ``items``, serially, in order.

        See :meth:`femtoolkit.orchestration.backends.base.ExecutionBackend.run`
        for the full argument/return contract. ``config.preserve_order``
        has no effect here -- serial execution is always in submission
        order, which is also always completion order.
        """
        if not items:
            return []

        start_time = time.perf_counter()
        outcomes: list[TaskOutcome[R]] = []
        fail_fast_triggered = False

        for task_id, item in zip(task_ids, items, strict=True):
            if cancellation_token.is_cancelled or fail_fast_triggered:
                outcomes.append(
                    TaskOutcome(
                        task_id=task_id,
                        status=TaskStatus.CANCELLED,
                        error_message=(
                            "Batch was cancelled."
                            if cancellation_token.is_cancelled
                            else "Skipped: an earlier task failed (fail_fast)."
                        ),
                    )
                )
                continue

            started_at = utc_now_iso()
            task_start = time.perf_counter()
            try:
                value = func(item)
                outcomes.append(
                    TaskOutcome(
                        task_id=task_id,
                        status=TaskStatus.COMPLETED,
                        value=value,
                        started_at=started_at,
                        completed_at=utc_now_iso(),
                        duration_seconds=time.perf_counter() - task_start,
                    )
                )
            except Exception as error:  # noqa: BLE001 -- a task's own failure must never
                # crash the whole batch; it is recorded, not propagated (spec section 15).
                outcomes.append(
                    TaskOutcome(
                        task_id=task_id,
                        status=TaskStatus.FAILED,
                        error_type=type(error).__name__,
                        error_message=f"{error}\n{traceback.format_exc()}",
                        started_at=started_at,
                        completed_at=utc_now_iso(),
                        duration_seconds=time.perf_counter() - task_start,
                    )
                )
                if config.fail_fast:
                    fail_fast_triggered = True

            if on_progress is not None:
                completed = sum(1 for o in outcomes if o.status is TaskStatus.COMPLETED)
                failed = sum(1 for o in outcomes if o.status is TaskStatus.FAILED)
                cancelled = sum(1 for o in outcomes if o.status is TaskStatus.CANCELLED)
                on_progress(
                    ExecutionProgress(
                        total_tasks=len(items),
                        completed_tasks=completed,
                        failed_tasks=failed,
                        cancelled_tasks=cancelled,
                        running_tasks=0,
                        elapsed_seconds=time.perf_counter() - start_time,
                    )
                )

        return outcomes


__all__ = ["SerialBackend"]
