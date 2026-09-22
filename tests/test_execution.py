"""Tests for femtoolkit.execution (Version 27)."""

from __future__ import annotations

import pytest

from femtoolkit.exceptions import (
    InvalidExecutionConfigurationError,
    TaskSerializationError,
    WorkerExecutionError,
)
from femtoolkit.execution import (
    ExecutionConfig,
    ParallelExecutor,
    SerialExecutor,
    create_executor,
    resolve_chunk_size,
)
from femtoolkit.execution.config import resolve_workers


def _square(x: int) -> int:
    return x * x


def _identity(x: int) -> int:
    return x


def _raise_for_value(value: int, *, bad_value: int) -> int:
    if value == bad_value:
        raise ValueError(f"synthetic failure at {value}")
    return value


def _raise_for_three(x: int) -> int:
    return _raise_for_value(x, bad_value=3)


# --- ExecutionConfig ---------------------------------------------------


def test_execution_config_defaults_are_serial() -> None:
    config = ExecutionConfig()
    assert config.mode == "serial"
    assert config.workers is None
    assert config.chunk_size == "auto"
    assert config.backend == "process"


def test_execution_config_rejects_invalid_mode() -> None:
    with pytest.raises(InvalidExecutionConfigurationError):
        ExecutionConfig(mode="bogus")


def test_execution_config_rejects_non_positive_workers() -> None:
    with pytest.raises(InvalidExecutionConfigurationError):
        ExecutionConfig(mode="parallel", workers=0)
    with pytest.raises(InvalidExecutionConfigurationError):
        ExecutionConfig(mode="parallel", workers=-1)


def test_execution_config_rejects_invalid_chunk_size() -> None:
    with pytest.raises(InvalidExecutionConfigurationError):
        ExecutionConfig(chunk_size=0)
    with pytest.raises(InvalidExecutionConfigurationError):
        ExecutionConfig(chunk_size="not-auto")


def test_execution_config_rejects_invalid_backend() -> None:
    with pytest.raises(InvalidExecutionConfigurationError):
        ExecutionConfig(backend="gpu")


def test_resolve_workers_explicit() -> None:
    assert resolve_workers(ExecutionConfig(mode="parallel", workers=3)) == 3


def test_resolve_workers_automatic_is_capped_and_positive() -> None:
    workers = resolve_workers(ExecutionConfig())
    assert 1 <= workers <= 8


# --- resolve_chunk_size --------------------------------------------------


def test_resolve_chunk_size_explicit() -> None:
    config = ExecutionConfig(mode="parallel", chunk_size=5)
    assert resolve_chunk_size(config, item_count=100, worker_count=4) == 5


def test_resolve_chunk_size_auto_is_positive() -> None:
    config = ExecutionConfig(mode="parallel")
    assert resolve_chunk_size(config, item_count=1000, worker_count=4) >= 1


def test_resolve_chunk_size_auto_handles_empty() -> None:
    config = ExecutionConfig(mode="parallel")
    assert resolve_chunk_size(config, item_count=0, worker_count=4) == 1


# --- create_executor ------------------------------------------------------


def test_create_executor_default_is_serial() -> None:
    assert isinstance(create_executor(), SerialExecutor)
    assert isinstance(create_executor(None), SerialExecutor)


def test_create_executor_parallel_config_builds_parallel_executor() -> None:
    executor = create_executor(ExecutionConfig(mode="parallel", workers=2))
    assert isinstance(executor, ParallelExecutor)


# --- SerialExecutor --------------------------------------------------------


def test_serial_executor_matches_plain_map() -> None:
    items = list(range(10))
    assert SerialExecutor().map(_square, items) == [x * x for x in items]


def test_serial_executor_empty_workload() -> None:
    assert SerialExecutor().map(_square, []) == []


def test_serial_executor_preserves_order() -> None:
    items = [5, 4, 3, 2, 1]
    assert SerialExecutor().map(_identity, items) == items


def test_serial_executor_propagates_exceptions() -> None:
    with pytest.raises(ValueError, match="synthetic failure"):
        SerialExecutor().map(_raise_for_three, [1, 2, 3, 4])


# --- ParallelExecutor (process backend) ------------------------------------


def test_parallel_process_executor_matches_serial() -> None:
    items = list(range(30))
    serial = SerialExecutor().map(_square, items)
    parallel = create_executor(
        ExecutionConfig(mode="parallel", workers=2, backend="process")
    ).map(_square, items)
    assert serial == parallel


def test_parallel_process_executor_preserves_order() -> None:
    items = list(range(20))
    executor = create_executor(ExecutionConfig(mode="parallel", workers=3, backend="process"))
    assert executor.map(_identity, items) == items


def test_parallel_process_executor_empty_workload() -> None:
    executor = create_executor(ExecutionConfig(mode="parallel", workers=2, backend="process"))
    assert executor.map(_square, []) == []


def test_parallel_process_executor_wraps_worker_exception() -> None:
    executor = create_executor(ExecutionConfig(mode="parallel", workers=2, backend="process"))
    with pytest.raises(WorkerExecutionError, match="synthetic failure"):
        executor.map(_raise_for_three, [1, 2, 3, 4])


def test_parallel_process_executor_rejects_unpicklable_function() -> None:
    executor = create_executor(ExecutionConfig(mode="parallel", workers=2, backend="process"))
    with pytest.raises(TaskSerializationError):
        executor.map(lambda x: x, [1, 2, 3])


# --- ParallelExecutor (thread backend) -------------------------------------


def test_parallel_thread_executor_matches_serial() -> None:
    items = list(range(30))
    serial = SerialExecutor().map(_square, items)
    parallel = create_executor(
        ExecutionConfig(mode="parallel", workers=2, backend="thread")
    ).map(_square, items)
    assert serial == parallel


def test_parallel_thread_executor_wraps_worker_exception() -> None:
    executor = create_executor(ExecutionConfig(mode="parallel", workers=2, backend="thread"))
    with pytest.raises(WorkerExecutionError, match="synthetic failure"):
        executor.map(_raise_for_three, [1, 2, 3, 4])


def test_parallel_thread_executor_allows_lambda() -> None:
    executor = create_executor(ExecutionConfig(mode="parallel", workers=2, backend="thread"))
    assert executor.map(lambda x: x * 2, [1, 2, 3]) == [2, 4, 6]
