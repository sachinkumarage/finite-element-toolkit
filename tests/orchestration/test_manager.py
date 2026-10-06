"""Tests for femtoolkit.orchestration.manager (Version 34)."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor

from femtoolkit.orchestration.config import OrchestrationConfig
from femtoolkit.orchestration.manager import (
    ParallelExecutionManager,
    SerialExecutionManager,
    build_execution_manager,
)
from femtoolkit.orchestration.models import TaskStatus


def _square(x: int) -> int:
    return x * x


def _run_nested_parallel_batch(_unused: int) -> str:
    manager = build_execution_manager(OrchestrationConfig(execution_mode="parallel", max_workers=2))
    _, summary = manager.run_batch(_square, [1, 2, 3])
    return summary.execution_mode


def _check_summary_notes(_unused: int) -> list[str]:
    manager = build_execution_manager(OrchestrationConfig(execution_mode="parallel", max_workers=2))
    _, summary = manager.run_batch(_square, [1, 2, 3])
    return summary.notes


def test_build_execution_manager_default_is_serial() -> None:
    manager = build_execution_manager()
    assert isinstance(manager, SerialExecutionManager)


def test_build_execution_manager_serial_config() -> None:
    manager = build_execution_manager(OrchestrationConfig(execution_mode="serial"))
    assert isinstance(manager, SerialExecutionManager)


def test_build_execution_manager_parallel_config() -> None:
    manager = build_execution_manager(OrchestrationConfig(execution_mode="parallel"))
    assert isinstance(manager, ParallelExecutionManager)


def test_serial_manager_run_batch_basic() -> None:
    manager = SerialExecutionManager()
    outcomes, summary = manager.run_batch(_square, [1, 2, 3], task_ids=["a", "b", "c"])
    assert [o.value for o in outcomes] == [1, 4, 9]
    assert summary.execution_mode == "serial"
    assert summary.worker_count == 1
    assert summary.total_tasks == 3


def test_run_batch_generates_default_task_ids() -> None:
    manager = SerialExecutionManager()
    outcomes, _summary = manager.run_batch(_square, [1, 2, 3])
    assert [o.task_id for o in outcomes] == ["0", "1", "2"]


def test_parallel_manager_run_batch_basic() -> None:
    manager = build_execution_manager(OrchestrationConfig(execution_mode="parallel", max_workers=2))
    outcomes, summary = manager.run_batch(_square, [1, 2, 3, 4], task_ids=["a", "b", "c", "d"])
    assert [o.value for o in outcomes] == [1, 4, 9, 16]
    assert summary.execution_mode == "parallel"
    assert summary.worker_count == 2


def test_cancel_before_run_batch_cancels_everything() -> None:
    manager = SerialExecutionManager()
    manager.cancel()
    outcomes, _summary = manager.run_batch(_square, [1, 2, 3])
    assert all(o.status is TaskStatus.CANCELLED for o in outcomes)


def test_nested_parallel_request_downgrades_to_serial() -> None:
    outer_config = OrchestrationConfig(execution_mode="parallel", max_workers=2)
    outer_manager = build_execution_manager(outer_config)
    outcomes, outer_summary = outer_manager.run_batch(_run_nested_parallel_batch, [0, 1])
    assert outer_summary.execution_mode == "parallel"
    # Each worker's own nested attempt must have been downgraded to serial.
    assert [o.value for o in outcomes] == ["serial", "serial"]


def test_nested_downgrade_is_noted_in_summary() -> None:
    with ProcessPoolExecutor(max_workers=1) as pool:
        notes = pool.submit(_check_summary_notes, 0).result()
    assert any("downgrad" in note.lower() for note in notes)
