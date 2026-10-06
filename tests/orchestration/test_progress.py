"""Tests for femtoolkit.orchestration.progress (Version 34)."""

from __future__ import annotations

from femtoolkit.orchestration.progress import ExecutionProgress, format_console_progress


def test_percentage_zero_tasks_is_complete() -> None:
    progress = ExecutionProgress(
        total_tasks=0, completed_tasks=0, failed_tasks=0, cancelled_tasks=0,
        running_tasks=0, elapsed_seconds=0.0,
    )
    assert progress.percentage == 100.0
    assert progress.finished_tasks == 0


def test_percentage_partial_progress() -> None:
    progress = ExecutionProgress(
        total_tasks=10, completed_tasks=3, failed_tasks=1, cancelled_tasks=0,
        running_tasks=2, elapsed_seconds=4.0,
    )
    assert progress.finished_tasks == 4
    assert progress.percentage == 40.0


def test_estimated_remaining_none_before_any_finish() -> None:
    progress = ExecutionProgress(
        total_tasks=10, completed_tasks=0, failed_tasks=0, cancelled_tasks=0,
        running_tasks=3, elapsed_seconds=1.0,
    )
    assert progress.estimated_remaining_seconds is None


def test_estimated_remaining_linear_extrapolation() -> None:
    progress = ExecutionProgress(
        total_tasks=10, completed_tasks=5, failed_tasks=0, cancelled_tasks=0,
        running_tasks=5, elapsed_seconds=10.0,
    )
    # 5 finished in 10s -> 2s/task average; 5 remaining -> 10s estimate.
    assert progress.estimated_remaining_seconds == 10.0


def test_format_console_progress_contains_key_fields() -> None:
    progress = ExecutionProgress(
        total_tasks=10, completed_tasks=4, failed_tasks=1, cancelled_tasks=0,
        running_tasks=5, elapsed_seconds=3.5,
    )
    line = format_console_progress(progress)
    assert "5/10" in line
    assert "50.0%" in line
    assert "Completed: 4" in line
    assert "Failed: 1" in line
    assert "Running: 5" in line
