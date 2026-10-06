"""Tests for femtoolkit.orchestration.backends.local_process (Version 34).

Every worker function here is module-level (required for picklability
-- see the module under test's own docstring). Timing-based assertions
(cancellation, timeout) use generous margins and short sleep durations
rather than asserting on exact elapsed time, per this version's own
"do not make tests dependent on exact execution time" requirement.
"""

from __future__ import annotations

import threading
import time

from femtoolkit.exceptions import TaskSerializationError
from femtoolkit.orchestration.backends.local_process import LocalProcessBackend
from femtoolkit.orchestration.cancellation import CancellationToken
from femtoolkit.orchestration.config import OrchestrationConfig
from femtoolkit.orchestration.models import TaskStatus


def _square(x: int) -> int:
    return x * x


def _sleep_and_return(seconds: float) -> float:
    time.sleep(seconds)
    return seconds


def _raise_for_value(value: int, bad_value: int) -> int:
    if value == bad_value:
        raise ValueError(f"synthetic failure at {value}")
    return value


def _fail_on_odd(x: int) -> int:
    if x % 2 == 1:
        raise ValueError(f"odd value {x}")
    return x * 2


def _fail_on_two(x: int) -> int:
    return _raise_for_value(x, 2)


def test_empty_batch_returns_empty_list() -> None:
    outcomes = LocalProcessBackend().run(
        _square, [], [], OrchestrationConfig(execution_mode="parallel"), CancellationToken()
    )
    assert outcomes == []


def test_basic_parallel_execution() -> None:
    config = OrchestrationConfig(execution_mode="parallel", max_workers=2)
    outcomes = LocalProcessBackend().run(
        _square, [1, 2, 3, 4], ["a", "b", "c", "d"], config, CancellationToken()
    )
    assert [o.status for o in outcomes] == [TaskStatus.COMPLETED] * 4
    assert [o.value for o in outcomes] == [1, 4, 9, 16]
    assert [o.task_id for o in outcomes] == ["a", "b", "c", "d"]


def test_preserves_submission_order_by_default() -> None:
    config = OrchestrationConfig(execution_mode="parallel", max_workers=4)
    outcomes = LocalProcessBackend().run(
        _sleep_and_return, [0.2, 0.01, 0.1], ["slow", "fast", "medium"], config, CancellationToken()
    )
    assert [o.task_id for o in outcomes] == ["slow", "fast", "medium"]


def test_completion_order_when_preserve_order_disabled() -> None:
    config = OrchestrationConfig(execution_mode="parallel", max_workers=1, preserve_order=False)
    outcomes = LocalProcessBackend().run(
        _sleep_and_return, [0.2, 0.01], ["slow", "fast"], config, CancellationToken()
    )
    # With one worker, "fast" is submitted second but there's only one slot;
    # identity must still be correct regardless of position.
    assert {o.task_id for o in outcomes} == {"slow", "fast"}


def test_records_timing_fields() -> None:
    config = OrchestrationConfig(execution_mode="parallel", max_workers=2)
    outcomes = LocalProcessBackend().run(
        _square, [3], ["a"], config, CancellationToken()
    )
    outcome = outcomes[0]
    assert outcome.started_at is not None
    assert outcome.completed_at is not None
    assert outcome.duration_seconds is not None
    assert outcome.duration_seconds >= 0.0


def test_failures_are_recorded_without_stopping_the_batch() -> None:
    config = OrchestrationConfig(execution_mode="parallel", max_workers=2)
    outcomes = LocalProcessBackend().run(
        _fail_on_odd, [1, 2, 3, 4], ["a", "b", "c", "d"], config, CancellationToken()
    )
    statuses = {o.task_id: o.status for o in outcomes}
    assert statuses["a"] is TaskStatus.FAILED
    assert statuses["b"] is TaskStatus.COMPLETED
    assert statuses["c"] is TaskStatus.FAILED
    assert statuses["d"] is TaskStatus.COMPLETED
    failed = next(o for o in outcomes if o.task_id == "a")
    assert failed.error_type == "ValueError"


def test_fail_fast_cancels_queued_tasks() -> None:
    config = OrchestrationConfig(execution_mode="parallel", max_workers=1, fail_fast=True)
    outcomes = LocalProcessBackend().run(
        _fail_on_two, [1, 2, 3, 4], ["a", "b", "c", "d"], config, CancellationToken()
    )
    statuses = {o.task_id: o.status for o in outcomes}
    assert statuses["a"] is TaskStatus.COMPLETED
    assert statuses["b"] is TaskStatus.FAILED
    assert statuses["c"] is TaskStatus.CANCELLED
    assert statuses["d"] is TaskStatus.CANCELLED


def test_cancellation_stops_the_batch_promptly() -> None:
    config = OrchestrationConfig(execution_mode="parallel", max_workers=2)
    token = CancellationToken()

    def cancel_soon() -> None:
        time.sleep(0.3)
        token.cancel()

    thread = threading.Thread(target=cancel_soon)
    thread.start()
    start = time.perf_counter()
    outcomes = LocalProcessBackend().run(
        _sleep_and_return, [5.0, 5.0, 5.0, 5.0], ["a", "b", "c", "d"], config, token
    )
    elapsed = time.perf_counter() - start
    thread.join()

    assert all(o.status is TaskStatus.CANCELLED for o in outcomes)
    # Should stop well before any 5-second task would finish naturally.
    assert elapsed < 3.0


def test_timeout_marks_outstanding_tasks() -> None:
    config = OrchestrationConfig(execution_mode="parallel", max_workers=1, timeout=0.5)
    outcomes = LocalProcessBackend().run(
        _sleep_and_return, [5.0, 5.0], ["a", "b"], config, CancellationToken()
    )
    statuses = {o.task_id: o.status for o in outcomes}
    assert statuses["a"] is TaskStatus.FAILED
    assert outcomes[0].error_type == "TaskTimeoutError"
    assert statuses["b"] is TaskStatus.CANCELLED


def test_unpicklable_function_raises_task_serialization_error() -> None:
    config = OrchestrationConfig(execution_mode="parallel", max_workers=2)
    try:
        LocalProcessBackend().run(
            lambda x: x, [1, 2], ["a", "b"], config, CancellationToken()
        )
        raised = False
    except TaskSerializationError:
        raised = True
    assert raised


def test_max_workers_respected_for_submission_capacity() -> None:
    # Indirect check: with max_workers=1 and a timeout shorter than one
    # task's duration, only one task should have been dispatched (the
    # other stays queued and is marked CANCELLED, not FAILED/timed-out).
    config = OrchestrationConfig(execution_mode="parallel", max_workers=1, timeout=0.3)
    outcomes = LocalProcessBackend().run(
        _sleep_and_return, [5.0, 5.0, 5.0], ["a", "b", "c"], config, CancellationToken()
    )
    by_id = {o.task_id: o for o in outcomes}
    assert by_id["a"].status is TaskStatus.FAILED  # was dispatched, timed out
    assert by_id["b"].status is TaskStatus.CANCELLED  # never dispatched
    assert by_id["c"].status is TaskStatus.CANCELLED  # never dispatched
