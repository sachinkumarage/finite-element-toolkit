"""Tests for femtoolkit.orchestration.summary (Version 34)."""

from __future__ import annotations

from femtoolkit.orchestration.models import TaskOutcome, TaskStatus
from femtoolkit.orchestration.summary import build_execution_summary


def _outcome(task_id: str, status: TaskStatus, duration: float | None = None) -> TaskOutcome:
    return TaskOutcome(task_id=task_id, status=status, duration_seconds=duration)


def test_serial_summary_has_no_speedup_fields() -> None:
    outcomes = [_outcome("a", TaskStatus.COMPLETED, 1.0), _outcome("b", TaskStatus.COMPLETED, 1.0)]
    summary = build_execution_summary(outcomes, "serial", worker_count=1, total_elapsed_seconds=2.0)
    assert summary.execution_mode == "serial"
    assert summary.serial_estimated_seconds is None
    assert summary.parallel_speedup is None
    assert summary.parallel_efficiency is None


def test_parallel_summary_computes_speedup_and_efficiency() -> None:
    outcomes = [_outcome(str(i), TaskStatus.COMPLETED, 1.0) for i in range(8)]
    summary = build_execution_summary(
        outcomes, "parallel", worker_count=4, total_elapsed_seconds=2.0
    )
    # average_task_seconds = 1.0, serial_estimated = 8 * 1.0 = 8.0
    assert summary.serial_estimated_seconds == 8.0
    assert summary.parallel_speedup == 4.0  # 8.0 / 2.0
    assert summary.parallel_efficiency == 1.0  # 4.0 / 4 workers


def test_counts_completed_failed_and_cancelled() -> None:
    outcomes = [
        _outcome("a", TaskStatus.COMPLETED, 1.0),
        _outcome("b", TaskStatus.FAILED, 0.5),
        _outcome("c", TaskStatus.CANCELLED),
        _outcome("d", TaskStatus.SKIPPED),
    ]
    summary = build_execution_summary(outcomes, "serial", worker_count=1, total_elapsed_seconds=1.5)
    assert summary.total_tasks == 4
    assert summary.completed_tasks == 1
    assert summary.failed_tasks == 1
    assert summary.cancelled_tasks == 2


def test_fastest_and_slowest_task_seconds() -> None:
    outcomes = [
        _outcome("a", TaskStatus.COMPLETED, 2.0),
        _outcome("b", TaskStatus.COMPLETED, 0.5),
        _outcome("c", TaskStatus.FAILED, 1.0),
    ]
    summary = build_execution_summary(outcomes, "serial", worker_count=1, total_elapsed_seconds=3.5)
    assert summary.fastest_task_seconds == 0.5
    assert summary.slowest_task_seconds == 2.0


def test_no_duration_data_yields_none_averages() -> None:
    outcomes = [_outcome("a", TaskStatus.CANCELLED), _outcome("b", TaskStatus.SKIPPED)]
    summary = build_execution_summary(outcomes, "serial", worker_count=1, total_elapsed_seconds=0.0)
    assert summary.average_task_seconds is None
    assert summary.fastest_task_seconds is None
    assert summary.slowest_task_seconds is None


def test_notes_are_carried_through() -> None:
    outcomes = [_outcome("a", TaskStatus.COMPLETED, 1.0)]
    summary = build_execution_summary(
        outcomes, "serial", worker_count=1, total_elapsed_seconds=1.0, notes=["downgraded"]
    )
    assert summary.notes == ["downgraded"]


def test_notes_default_to_empty_list() -> None:
    outcomes = [_outcome("a", TaskStatus.COMPLETED, 1.0)]
    summary = build_execution_summary(outcomes, "serial", worker_count=1, total_elapsed_seconds=1.0)
    assert summary.notes == []


def test_execution_summary_to_dict_has_expected_keys() -> None:
    from femtoolkit.orchestration.summary import execution_summary_to_dict

    outcomes = [_outcome(str(i), TaskStatus.COMPLETED, 1.0) for i in range(4)]
    summary = build_execution_summary(
        outcomes, "parallel", worker_count=2, total_elapsed_seconds=2.0
    )
    data = execution_summary_to_dict(summary)
    assert data == {
        "execution_mode": "parallel",
        "worker_count": 2,
        "total_tasks": 4,
        "completed_tasks": 4,
        "failed_tasks": 0,
        "cancelled_tasks": 0,
        "total_elapsed_seconds": 2.0,
    }
