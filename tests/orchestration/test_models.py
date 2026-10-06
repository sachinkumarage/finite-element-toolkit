"""Tests for femtoolkit.orchestration.models (Version 34)."""

from __future__ import annotations

from femtoolkit.orchestration.models import TaskOutcome, TaskStatus, is_terminal_status, utc_now_iso


def test_terminal_statuses() -> None:
    assert is_terminal_status(TaskStatus.COMPLETED)
    assert is_terminal_status(TaskStatus.FAILED)
    assert is_terminal_status(TaskStatus.CANCELLED)
    assert is_terminal_status(TaskStatus.SKIPPED)


def test_non_terminal_statuses() -> None:
    assert not is_terminal_status(TaskStatus.PENDING)
    assert not is_terminal_status(TaskStatus.RUNNING)


def test_task_outcome_defaults() -> None:
    outcome = TaskOutcome(task_id="a", status=TaskStatus.PENDING)
    assert outcome.value is None
    assert outcome.error_type is None
    assert outcome.error_message is None
    assert outcome.started_at is None
    assert outcome.completed_at is None
    assert outcome.duration_seconds is None
    assert outcome.worker_name is None


def test_is_successful_true_only_for_completed() -> None:
    assert TaskOutcome(task_id="a", status=TaskStatus.COMPLETED, value=1).is_successful
    assert not TaskOutcome(task_id="a", status=TaskStatus.FAILED).is_successful
    assert not TaskOutcome(task_id="a", status=TaskStatus.CANCELLED).is_successful


def test_utc_now_iso_is_a_parseable_iso_string() -> None:
    from datetime import datetime

    timestamp = utc_now_iso()
    parsed = datetime.fromisoformat(timestamp)
    assert parsed.tzinfo is not None
