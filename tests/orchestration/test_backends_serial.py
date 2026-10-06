"""Tests for femtoolkit.orchestration.backends.serial (Version 34)."""

from __future__ import annotations

from femtoolkit.orchestration.backends.serial import SerialBackend
from femtoolkit.orchestration.cancellation import CancellationToken
from femtoolkit.orchestration.config import OrchestrationConfig
from femtoolkit.orchestration.models import TaskStatus
from femtoolkit.orchestration.progress import ExecutionProgress


def _square(x: int) -> int:
    return x * x


def _raise_for_value(value: int, bad_value: int) -> int:
    if value == bad_value:
        raise ValueError(f"synthetic failure at {value}")
    return value


def _fail_on_three(x: int) -> int:
    return _raise_for_value(x, 3)


def test_empty_batch_returns_empty_list() -> None:
    outcomes = SerialBackend().run(
        _square, [], [], OrchestrationConfig(), CancellationToken()
    )
    assert outcomes == []


def test_basic_execution_matches_plain_map() -> None:
    items = [1, 2, 3, 4]
    outcomes = SerialBackend().run(
        _square, items, ["a", "b", "c", "d"], OrchestrationConfig(), CancellationToken()
    )
    assert [o.status for o in outcomes] == [TaskStatus.COMPLETED] * 4
    assert [o.value for o in outcomes] == [1, 4, 9, 16]
    assert [o.task_id for o in outcomes] == ["a", "b", "c", "d"]


def test_preserves_submission_order() -> None:
    outcomes = SerialBackend().run(
        _square, [5, 1, 3], ["x", "y", "z"], OrchestrationConfig(), CancellationToken()
    )
    assert [o.task_id for o in outcomes] == ["x", "y", "z"]


def test_records_timing_fields() -> None:
    outcomes = SerialBackend().run(
        _square, [2], ["a"], OrchestrationConfig(), CancellationToken()
    )
    outcome = outcomes[0]
    assert outcome.started_at is not None
    assert outcome.completed_at is not None
    assert outcome.duration_seconds is not None
    assert outcome.duration_seconds >= 0.0


def test_failure_is_recorded_not_raised() -> None:
    outcomes = SerialBackend().run(
        _fail_on_three, [1, 2, 3, 4], ["a", "b", "c", "d"], OrchestrationConfig(),
        CancellationToken(),
    )
    statuses = [o.status for o in outcomes]
    assert statuses == [
        TaskStatus.COMPLETED, TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.COMPLETED,
    ]
    assert outcomes[2].error_type == "ValueError"
    assert "synthetic failure" in outcomes[2].error_message


def test_fail_fast_cancels_remaining_tasks() -> None:
    config = OrchestrationConfig(fail_fast=True)
    outcomes = SerialBackend().run(
        _fail_on_three, [1, 2, 3, 4, 5], ["a", "b", "c", "d", "e"], config, CancellationToken()
    )
    statuses = [o.status for o in outcomes]
    assert statuses == [
        TaskStatus.COMPLETED, TaskStatus.COMPLETED, TaskStatus.FAILED,
        TaskStatus.CANCELLED, TaskStatus.CANCELLED,
    ]


def test_fail_fast_disabled_runs_every_task() -> None:
    config = OrchestrationConfig(fail_fast=False)
    outcomes = SerialBackend().run(
        _fail_on_three, [1, 2, 3, 4, 5], ["a", "b", "c", "d", "e"], config, CancellationToken()
    )
    statuses = [o.status for o in outcomes]
    assert statuses.count(TaskStatus.FAILED) == 1
    assert statuses.count(TaskStatus.COMPLETED) == 4


def test_cancellation_before_execution_cancels_everything() -> None:
    token = CancellationToken()
    token.cancel()
    outcomes = SerialBackend().run(
        _square, [1, 2, 3], ["a", "b", "c"], OrchestrationConfig(), token
    )
    assert all(o.status is TaskStatus.CANCELLED for o in outcomes)


def test_on_progress_called_once_per_task() -> None:
    snapshots: list[ExecutionProgress] = []
    SerialBackend().run(
        _square, [1, 2, 3], ["a", "b", "c"], OrchestrationConfig(), CancellationToken(),
        on_progress=snapshots.append,
    )
    assert len(snapshots) == 3
    assert snapshots[-1].completed_tasks == 3
    assert snapshots[-1].total_tasks == 3
